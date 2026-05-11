"""LLM orchestration for DataVaidya.

Streams analytical summaries of tabular data using Groq as the primary
provider with Gemini as a fallback. Implements per-session call budgeting,
context truncation under a token budget, and graceful degradation when
providers fail.

Public surface:
    - PROMPT_VERSION
    - estimate_tokens
    - build_context
    - check_call_budget / increment_call_count / reset_call_count
    - stream_summary (top-level orchestrator)

Internal helpers prefixed with `_` are not part of the stable API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator, Literal

import pandas as pd

from utils.constants import (
    GROQ_MODEL,
    GROQ_FALLBACK_MODELS,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODEL,
    LLM_TIMEOUT_S,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_SESSION_CAP,
    LLM_CONTEXT_TOKEN_BUDGET,
    LLM_TEMPERATURE,
    LLM_FALLBACK_MESSAGE,
    PROMPT_V1,
)
from core.pii import scrub_for_llm

logger = logging.getLogger(__name__)

PROMPT_VERSION: str = "PROMPT_V1"

# System message is held constant across calls so Groq's prefix cache can
# match. Do not interpolate per-request data into SYSTEM.
SYSTEM: str = PROMPT_V1

# Refusal text shown when the session call cap has been hit.
_REFUSAL_TEMPLATE: str = (
    "Session question limit reached ({cap}). Refresh app to reset."
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Estimate token count from character length.

    Uses the standard chars/4 heuristic. Cheap and provider-agnostic; we
    only need an order-of-magnitude estimate to keep context under budget.

    Args:
        text: The input string. ``None`` and empty strings are treated as 0.

    Returns:
        Approximate token count (always non-negative).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def _format_schema_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Render a Markdown schema table for the given columns.

    Args:
        df: The source dataframe.
        cols: Optional restricted column list. Defaults to all df columns.

    Returns:
        A Markdown table string. Empty string if no columns.
    """
    if cols is None:
        cols = list(df.columns)
    if not cols:
        return ""

    n_rows = len(df) if len(df) > 0 else 1
    lines = ["| Column | Type | Non-Null % |", "|---|---|---|"]
    for col in cols:
        try:
            dtype = str(df[col].dtype)
            non_null_pct = 100.0 * df[col].notna().sum() / n_rows
        except Exception:  # pragma: no cover - defensive
            dtype = "unknown"
            non_null_pct = 0.0
        lines.append(f"| {col} | {dtype} | {non_null_pct:.1f}% |")
    return "\n".join(lines)


def _format_numeric_summary(df: pd.DataFrame) -> str:
    """Return df.describe() as a Markdown table, rounded to 2dp.

    Returns an empty string when there are no numeric columns.
    """
    try:
        numeric = df.select_dtypes(include="number")
    except Exception:
        return ""
    if numeric.empty or numeric.shape[1] == 0:
        return ""
    try:
        desc = numeric.describe().round(2)
    except Exception:
        return ""
    # Use to_markdown if available, else fall back to a manual render.
    try:
        return desc.to_markdown()
    except Exception:
        # Manual fallback: header row + data rows.
        cols = list(desc.columns)
        header = "| stat | " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        rows = [header, sep]
        for idx in desc.index:
            vals = [f"{desc.loc[idx, c]}" for c in cols]
            rows.append(f"| {idx} | " + " | ".join(vals) + " |")
        return "\n".join(rows)


def _format_top_correlations(df: pd.DataFrame, top_n: int = 5) -> str:
    """Return the top ``top_n`` pairwise correlations by absolute value.

    Returns an empty string if there are fewer than two numeric columns.
    """
    try:
        numeric = df.select_dtypes(include="number")
    except Exception:
        return ""
    if numeric.shape[1] < 2:
        return ""
    try:
        corr = numeric.corr().abs()
    except Exception:
        return ""

    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            pairs.append((a, b, float(val)))

    if not pairs:
        return ""

    pairs.sort(key=lambda t: t[2], reverse=True)
    pairs = pairs[:top_n]

    lines = ["| Column A | Column B | |corr| |", "|---|---|---|"]
    for a, b, v in pairs:
        lines.append(f"| {a} | {b} | {v:.2f} |")
    return "\n".join(lines)


def _format_quality_flags(health_report: dict) -> str:
    """Render the quality flags bullet list from ``health_report['reasons']``."""
    if not health_report:
        return ""
    reasons = health_report.get("reasons") or []
    if not reasons:
        return ""
    lines = []
    for r in reasons:
        lines.append(f"- {r}")
    return "\n".join(lines)


def _format_sample(df: pd.DataFrame, n: int = 3) -> str:
    """Return a PII-scrubbed Markdown sample of ``n`` rows."""
    if df is None or df.empty:
        return ""
    try:
        sample = df.head(n).copy()
    except Exception:
        return ""

    # Scrub each cell. scrub_for_llm operates on strings.
    def _scrub_cell(v: Any) -> Any:
        if v is None:
            return v
        try:
            if isinstance(v, str):
                return scrub_for_llm(v)
            # Convert non-strings to string then scrub, but preserve numerics
            # to keep the sample useful for analytical signal.
            return v
        except Exception:
            return v

    try:
        sample = sample.applymap(_scrub_cell)
    except Exception:
        pass

    try:
        return sample.to_markdown(index=False)
    except Exception:
        # Manual fallback.
        cols = list(sample.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = [header, sep]
        for _, row in sample.iterrows():
            rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(rows)


def _rank_cols_by_missing(df: pd.DataFrame) -> list[str]:
    """Return columns sorted by descending missing-pct (most-missing first)."""
    if df is None or df.empty:
        return list(df.columns) if df is not None else []
    try:
        miss = df.isna().mean().sort_values(ascending=False)
        return list(miss.index)
    except Exception:
        return list(df.columns)


def build_context(
    df: pd.DataFrame,
    health_report: dict,
    dataset_name: str = "Uploaded dataset",
    max_tokens: int = LLM_CONTEXT_TOKEN_BUDGET,
) -> str:
    """Assemble a Markdown context block under ``max_tokens``.

    Template sections (in order):
        ## Dataset: {name}
        ### Schema ({n_cols} cols × {n_rows} rows)
        ### Numeric Summary
        ### Top Correlations
        ### Quality Flags
        ### Sample

    Truncation order when over budget:
        1. Trim schema to top 20 cols by missing pct; append a note.
        2. Drop numeric summary entirely; replace with omission marker.

    Quality flags are never dropped — they are the analytical signal.

    Args:
        df: The dataframe to summarize.
        health_report: A dict possibly containing a ``reasons`` list of
            quality-flag strings.
        dataset_name: Human-readable dataset name.
        max_tokens: Token budget for the rendered context.

    Returns:
        A Markdown string within the token budget, or ``"<empty dataframe>"``
        if ``df`` is None or empty.
    """
    if df is None or df.empty:
        return "<empty dataframe>"

    n_cols = df.shape[1]
    n_rows = df.shape[0]

    # Build each section once. Truncation operates by swapping sections.
    schema_full = _format_schema_table(df)
    numeric_summary = _format_numeric_summary(df)
    correlations = _format_top_correlations(df)
    quality_flags = _format_quality_flags(health_report or {})
    sample = _format_sample(df, n=3)

    def _assemble(schema: str, numeric: str, schema_note: str = "") -> str:
        parts: list[str] = []
        parts.append(f"## Dataset: {dataset_name}")
        parts.append(f"### Schema ({n_cols} cols × {n_rows} rows)")
        if schema:
            parts.append(schema)
        if schema_note:
            parts.append(schema_note)
        if numeric:
            parts.append("### Numeric Summary")
            parts.append(numeric)
        else:
            # When numeric was deliberately dropped, the omission marker is
            # supplied by the caller via the `numeric` argument. An empty
            # `numeric` here simply means no numeric columns existed; emit
            # nothing in that case.
            pass
        if correlations:
            parts.append("### Top Correlations")
            parts.append(correlations)
        if quality_flags:
            parts.append("### Quality Flags")
            parts.append(quality_flags)
        if sample:
            parts.append("### Sample")
            parts.append(sample)
        return "\n\n".join(parts)

    # Stage 0: full assembly.
    context = _assemble(schema_full, numeric_summary)
    if estimate_tokens(context) <= max_tokens:
        return context

    # Stage 1: trim schema to top 20 cols by missing pct.
    ranked = _rank_cols_by_missing(df)
    keep = ranked[:20]
    omitted = max(0, len(ranked) - len(keep))
    schema_trimmed = _format_schema_table(df, cols=keep)
    note = (
        f"*... and {omitted} more columns omitted*" if omitted > 0 else ""
    )
    context = _assemble(schema_trimmed, numeric_summary, schema_note=note)
    if estimate_tokens(context) <= max_tokens:
        return context

    # Stage 2: drop numeric summary section.
    omitted_marker = "*Numeric summary omitted (budget exceeded)*"
    context = _assemble(schema_trimmed, omitted_marker, schema_note=note)
    return context


# ---------------------------------------------------------------------------
# Session call budget
# ---------------------------------------------------------------------------
def check_call_budget() -> tuple[bool, str | None]:
    """Check whether another LLM call is permitted this session.

    Reads ``st.session_state['llm_call_count']`` (default 0) and compares
    against ``LLM_SESSION_CAP``.

    Returns:
        Tuple of ``(allowed, refusal_message)``. When allowed, the second
        element is ``None``.
    """
    try:
        import streamlit as st  # local import per spec
    except Exception:
        # If streamlit is unavailable, default to allowing the call.
        return True, None

    try:
        count = int(st.session_state.get("llm_call_count", 0))
    except Exception:
        count = 0

    if count >= LLM_SESSION_CAP:
        return False, _REFUSAL_TEMPLATE.format(cap=LLM_SESSION_CAP)
    return True, None


def increment_call_count() -> None:
    """Increment the session-scoped LLM call counter by one."""
    try:
        import streamlit as st  # local import per spec
    except Exception:
        return
    try:
        current = int(st.session_state.get("llm_call_count", 0))
    except Exception:
        current = 0
    st.session_state["llm_call_count"] = current + 1


def reset_call_count() -> None:
    """Reset the session-scoped LLM call counter to zero."""
    try:
        import streamlit as st  # local import per spec
    except Exception:
        return
    st.session_state["llm_call_count"] = 0


# ---------------------------------------------------------------------------
# Secret access
# ---------------------------------------------------------------------------
def _get_secret(name: str) -> str | None:
    """Read a secret from ``st.secrets``. Returns ``None`` if missing.

    KeyError and any access error are treated as missing.
    """
    try:
        import streamlit as st  # local import per spec
    except Exception:
        return None
    try:
        val = st.secrets[name]
        if not val:
            return None
        return str(val)
    except KeyError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Provider: Groq
# ---------------------------------------------------------------------------
def _call_groq_stream(messages: list[dict], model: str) -> Iterator[str]:
    """Stream completion tokens from Groq for the given model.

    Args:
        messages: Chat-format message list (system + user).
        model: Groq model identifier.

    Yields:
        Successive content fragments from ``delta.content``.

    Raises:
        RuntimeError: If the Groq API key is missing.
        groq.* errors: Propagated for the retry loop to classify.
    """
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")

    import groq  # third-party

    client = groq.Groq(api_key=api_key, timeout=LLM_TIMEOUT_S)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        try:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content
        except Exception:
            # Be tolerant of unexpected chunk shapes; just skip.
            continue


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------
def _call_gemini_stream(messages: list[dict], model: str) -> Iterator[str]:
    """Stream completion tokens from Gemini for the given model.

    Converts a system+user chat layout to Gemini's
    ``system_instruction`` + ``contents`` parameters.

    Args:
        messages: Chat-format message list (system + user).
        model: Gemini model identifier.

    Yields:
        Successive ``chunk.text`` fragments.

    Raises:
        RuntimeError: If the Gemini API key is missing.
        google.genai.errors.APIError: Propagated for the caller.
    """
    api_key = _get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    from google import genai  # third-party

    # Extract system and user text. Anything beyond the first user message
    # is concatenated to keep this generator simple and predictable.
    system_text = SYSTEM
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            # Spec calls for a fixed system instruction; prefer SYSTEM but
            # honor an overridden system message if provided.
            system_text = content or system_text
        elif role == "user":
            user_parts.append(content)
        else:
            user_parts.append(content)

    user_prompt = "\n\n".join(p for p in user_parts if p)

    client = genai.Client(api_key=api_key)
    stream = client.models.generate_content_stream(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_text,
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        },
    )
    for chunk in stream:
        try:
            text = getattr(chunk, "text", None)
            if text:
                yield text
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Retry / fallback machinery
# ---------------------------------------------------------------------------
def _is_retryable_groq_error(exc: BaseException) -> bool:
    """Return True if the exception is a known transient Groq error."""
    try:
        import groq
    except Exception:
        return False
    retryable = (
        getattr(groq, "RateLimitError", ()),
        getattr(groq, "APITimeoutError", ()),
        getattr(groq, "APIConnectionError", ()),
        getattr(groq, "InternalServerError", ()),
    )
    # Filter out empty tuples (missing attrs) before isinstance.
    retryable = tuple(t for t in retryable if isinstance(t, type))
    if not retryable:
        return False
    return isinstance(exc, retryable)


def _is_auth_or_missing_groq(exc: BaseException) -> bool:
    """Return True for Groq auth errors or missing-key sentinels."""
    if isinstance(exc, RuntimeError) and "GROQ_API_KEY" in str(exc):
        return True
    try:
        import groq
    except Exception:
        return False
    auth_cls = getattr(groq, "AuthenticationError", None)
    if auth_cls is not None and isinstance(exc, auth_cls):
        return True
    return False


def _retry_loop(
    fn_factory: Callable[[str], Iterator[str]],
    models: list[str],
    yield_collector: list,
    *,
    is_retryable: Callable[[BaseException], bool],
    max_retries: int,
) -> Iterator[str]:
    """Try each model in sequence, yielding tokens as they stream.

    The first model that produces at least one token "wins"; this helper
    will not switch providers after a token has been yielded. Mid-stream
    interruptions are surfaced by the caller (an interruption note is
    appended) rather than transparently retried.

    Args:
        fn_factory: Callable that takes a model name and returns a token
            iterator (e.g. ``_call_groq_stream`` bound with messages).
        models: Ordered model identifiers to try.
        yield_collector: Mutable list used as a flag — appended to on the
            first emitted token. Lets the orchestrator detect whether any
            token escaped this provider.
        is_retryable: Classifier deciding whether to advance to the next
            model on the given exception (vs. abort the whole loop).
        max_retries: Maximum attempts across the chain.

    Yields:
        Token fragments from the first working model. May also yield an
        "*(stream interrupted)*" marker if a mid-stream error occurs.
    """
    attempts = 0
    backoff = 1.5
    any_token_in_loop = False

    for model in models:
        if attempts >= max_retries:
            break
        attempts += 1
        local_emitted = False
        try:
            for token in fn_factory(model):
                if not local_emitted:
                    local_emitted = True
                    any_token_in_loop = True
                    yield_collector.append(True)
                yield token
            # Stream completed normally — done.
            return
        except BaseException as exc:  # noqa: BLE001
            if local_emitted or any_token_in_loop:
                # We already streamed something for this provider; do not
                # silently switch to another model. Signal the interruption.
                logger.warning(
                    "Mid-stream error from model=%s: %s", model, exc
                )
                yield "\n\n*(stream interrupted)*"
                return

            # Decide whether to advance.
            if _is_auth_or_missing_groq(exc):
                logger.info("Skipping Groq due to auth/missing key: %s", exc)
                # Abort the Groq chain entirely so the orchestrator can move on.
                return
            if not is_retryable(exc):
                logger.warning(
                    "Non-retryable error from model=%s: %s", model, exc
                )
                # Bail out of this provider's chain.
                return

            logger.info(
                "Retryable error from model=%s (attempt %d/%d): %s",
                model,
                attempts,
                max_retries,
                exc,
            )
            # Exponential backoff before the next model.
            try:
                time.sleep(backoff)
            except Exception:
                pass
            backoff *= 1.5
            continue


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True for known Gemini transient errors.

    Falls back to ``True`` for ``google.genai.errors.APIError`` so the
    chain can advance, and for the missing-key sentinel returns False so
    we abort the Gemini chain immediately.
    """
    if isinstance(exc, RuntimeError) and "GEMINI_API_KEY" in str(exc):
        return False
    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, getattr(genai_errors, "APIError", Exception)):
            return True
    except Exception:
        pass
    # Last-resort: treat unknown exceptions as retryable so the next
    # Gemini model gets a chance.
    return True


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def stream_summary(
    df: pd.DataFrame,
    health_report: dict,
    dataset_name: str = "Uploaded dataset",
) -> Iterator[str]:
    """Stream an analytical summary of ``df`` to the caller.

    Orchestrates the full call:
        1. Verify the session call budget. If exhausted, yield the refusal
           message and return without incrementing.
        2. Build the context block.
        3. Try Groq models (primary + fallbacks) until one starts streaming.
        4. If all Groq models fail before any token, try Gemini models.
        5. If everything fails before any token, yield
           ``LLM_FALLBACK_MESSAGE`` once.
        6. Increment the session call count exactly once, the moment the
           first token is yielded.

    The generator is safe to iterate exactly once. On mid-stream failure
    after at least one token has been emitted, an
    ``*(stream interrupted)*`` marker is appended and no provider switch
    occurs.

    Args:
        df: The dataframe to summarize.
        health_report: Data-quality metadata produced upstream.
        dataset_name: Human-readable dataset name.

    Yields:
        Summary tokens (strings).
    """
    # 1. Budget check.
    allowed, refusal = check_call_budget()
    if not allowed:
        yield refusal or _REFUSAL_TEMPLATE.format(cap=LLM_SESSION_CAP)
        return

    # 2. Context assembly.
    context = build_context(df, health_report, dataset_name=dataset_name)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": context},
    ]

    # Track first-token emission across providers to gate the increment
    # and to suppress the fallback message when something already streamed.
    emitted: list[bool] = []
    incremented = False

    def _emit(token: str) -> Iterator[str]:
        """Yield ``token`` and bump the call count on the first emission."""
        nonlocal incremented
        if not incremented:
            try:
                increment_call_count()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("increment_call_count failed: %s", exc)
            incremented = True
        yield token

    # 3. Groq chain.
    groq_models = [GROQ_MODEL] + list(GROQ_FALLBACK_MODELS or [])
    groq_collector: list = []
    try:
        for token in _retry_loop(
            lambda m: _call_groq_stream(messages, m),
            groq_models,
            groq_collector,
            is_retryable=_is_retryable_groq_error,
            max_retries=LLM_MAX_RETRIES,
        ):
            for t in _emit(token):
                yield t
            emitted.append(True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected Groq orchestration error: %s", exc)

    if emitted:
        return

    # 4. Gemini chain — only reached if Groq emitted nothing.
    gemini_models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL:
        gemini_models.append(GEMINI_FALLBACK_MODEL)

    gemini_collector: list = []
    try:
        for token in _retry_loop(
            lambda m: _call_gemini_stream(messages, m),
            gemini_models,
            gemini_collector,
            is_retryable=_is_retryable_gemini_error,
            max_retries=LLM_MAX_RETRIES,
        ):
            for t in _emit(token):
                yield t
            emitted.append(True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected Gemini orchestration error: %s", exc)

    if emitted:
        return

    # 5. Total failure — emit fallback once. Do NOT increment.
    yield LLM_FALLBACK_MESSAGE
