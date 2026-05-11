"""PII detection, masking, and LLM-safe data scrubbing for DataVaidya.

This module provides utilities to:
    * Detect PII columns in a pandas DataFrame using a combination of regex
      pattern matching and column-name heuristics.
    * Mask detected PII via three strategies: synthetic values (Faker),
      SHA-256 hashing, or partial redaction.
    * Produce LLM-safe Markdown summaries of a DataFrame with all PII
      placeholders inserted.

The detection strategy is intentionally conservative: high-precision regexes
(Aadhaar, PAN, IFSC, GSTIN, credit card) are paired with header heuristics
that surface free-text PII (names, addresses) which evade regex matching.

Public API:
    detect_pii(df, sample_size, min_hit_ratio) -> dict[str, list[str]]
    mask_pii(df, plan, method, seed) -> pd.DataFrame
    scrub_for_llm(df, n_sample_rows) -> str
    get_pii_summary(df) -> dict
"""

from __future__ import annotations

import hashlib
import logging
import re
import string
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #

PII_PATTERNS: dict[str, re.Pattern] = {
    "mobile":      re.compile(r'(?<!\d)(?:\+?91[\s\-]?|0)?[6-9](?:[\s\-]?\d){9}(?!\d)'),
    "aadhaar":     re.compile(r'(?<!\d)[2-9](?:[\s\-]?\d){11}(?!\d)'),
    "pan":         re.compile(r'\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b'),
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r'(?<!\d)(?:[3-6]\d{3})(?:[\s\-]?\d{4}){2}[\s\-]?\d{1,4}(?!\d)'),
    "pincode":     re.compile(r'(?<!\d)[1-9]\d{5}(?!\d)'),  # low confidence
    "ifsc":        re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),
    "gstin":       re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b'),
}

LOW_CONFIDENCE_CATEGORIES: frozenset[str] = frozenset({"pincode"})

# Column-name heuristic — boost detection for free-text PII regex misses
COLUMN_HEURISTICS: dict[str, set[str]] = {
    "email":       {"email", "e-mail", "mail", "email_id", "emailaddress", "email address"},
    "mobile":      {"mobile", "phone", "phone_no", "contact", "contact_no", "mobile_no",
                    "cell", "telephone", "tel"},
    "aadhaar":     {"aadhaar", "aadhar", "uid", "uidai", "aadhaar_no", "aadhaar number"},
    "pan":         {"pan", "pan_no", "pan_number", "pancard"},
    "credit_card": {"card", "card_no", "credit_card", "cc", "cc_number", "card_number"},
    "pincode":     {"pin", "pincode", "pin_code", "postal_code", "postcode", "zip", "zipcode"},
    "ifsc":        {"ifsc", "ifsc_code", "bank_code"},
    "gstin":       {"gst", "gstin", "gst_no", "gst_number", "tax_id"},
    "name":        {"name", "full_name", "first_name", "last_name", "customer_name"},
    "address":     {"address", "addr", "street", "location"},
}

SAMPLE_SIZE: int = 200
HIT_THRESHOLD: float = 0.20
LOW_CONF_HIT_THRESHOLD: float = 0.50

# Map category -> placeholder used by scrub_for_llm
_LLM_PLACEHOLDERS: dict[str, str] = {
    "email":       "<EMAIL>",
    "mobile":      "<PHONE>",
    "aadhaar":     "<AADHAAR>",
    "pan":         "<PAN>",
    "gstin":       "<GSTIN>",
    "ifsc":        "<IFSC>",
    "credit_card": "<CARD>",
    "pincode":     "<PINCODE>",
    "name":        "<NAME>",
    "address":     "<ADDRESS>",
}

# Free-text categories that have no reliable regex — header heuristic only.
_HEADER_ONLY_CATEGORIES: frozenset[str] = frozenset({"name", "address"})

# PAN 4th-character valid set (entity type)
_PAN_ENTITY_CHARS: str = "PCHFATBLJG"

_LLM_TOKEN_BUDGET_CHARS: int = 3500 * 4  # ~3500 tokens at ~4 chars/token


# --------------------------------------------------------------------------- #
# Internal helpers — normalization & detection
# --------------------------------------------------------------------------- #

def _norm_header(h: str) -> str:
    """Normalize a column header for heuristic matching.

    Lowercases, strips whitespace, and collapses dashes/spaces to underscores.

    Args:
        h: Raw column header.

    Returns:
        Normalized header string.
    """
    if h is None:
        return ""
    return str(h).strip().lower().replace("-", "_").replace(" ", "_")


def _header_hint(col: str) -> str | None:
    """Return a PII category suggested by a column name, else None.

    Performs an exact match first against ``COLUMN_HEURISTICS``, then falls
    back to a substring match so that headers like ``"customer_email_id"``
    still resolve to ``"email"``.

    Args:
        col: Column name (raw, not pre-normalized).

    Returns:
        Category key (e.g. ``"email"``) or None when no hint applies.
    """
    norm = _norm_header(col)
    if not norm:
        return None

    # Exact match wins
    for category, names in COLUMN_HEURISTICS.items():
        normed_names = {_norm_header(n) for n in names}
        if norm in normed_names:
            return category

    # Substring fallback — prefer longer keyword to avoid spurious matches
    # (e.g. "pin" inside "pincode" shouldn't shadow "pincode").
    best: tuple[str, int] | None = None
    for category, names in COLUMN_HEURISTICS.items():
        for n in names:
            kw = _norm_header(n)
            if kw and kw in norm:
                if best is None or len(kw) > best[1]:
                    best = (category, len(kw))
    return best[0] if best else None


def _match_column(series: pd.Series, pattern: re.Pattern, sample_size: int) -> float:
    """Return the fraction of sampled non-null values that match ``pattern``.

    Values are cast to string in memory only — the source series is not
    mutated. Empty samples return 0.0.

    Args:
        series: Column to scan.
        pattern: Compiled regex applied via ``pattern.search``.
        sample_size: Maximum number of non-null values to scan.

    Returns:
        Hit ratio in ``[0.0, 1.0]``.
    """
    if series is None or len(series) == 0:
        return 0.0

    non_null = series.dropna()
    if non_null.empty:
        return 0.0

    if len(non_null) > sample_size:
        sample = non_null.sample(n=sample_size, random_state=0)
    else:
        sample = non_null

    try:
        as_str = sample.astype(str)
    except (TypeError, ValueError):
        as_str = sample.map(lambda x: str(x))

    hits = sum(1 for v in as_str if pattern.search(v))
    return hits / len(as_str) if len(as_str) else 0.0


# --------------------------------------------------------------------------- #
# Public — detection
# --------------------------------------------------------------------------- #

def detect_pii(
    df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    min_hit_ratio: float = HIT_THRESHOLD,
) -> dict[str, list[str]]:
    """Detect PII categories present in each column of ``df``.

    Strategy:
        1. Skip columns that are >99% null.
        2. Sample up to ``sample_size`` non-null values per column.
        3. Apply every regex in :data:`PII_PATTERNS`; a category is recorded
           when the hit ratio meets ``min_hit_ratio`` (or
           :data:`LOW_CONF_HIT_THRESHOLD` for low-confidence categories).
        4. Apply :func:`_header_hint`; this is the only way free-text PII
           (names, addresses) is surfaced.

    Args:
        df: Input DataFrame. May be empty.
        sample_size: Max non-null values to inspect per column.
        min_hit_ratio: Minimum match fraction to accept a high-confidence
            category. Low-confidence categories use ``LOW_CONF_HIT_THRESHOLD``.

    Returns:
        Mapping ``{column_name: [categories]}`` for columns where at least
        one category was detected. Columns with no detections are omitted.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return {}

    detections: dict[str, list[str]] = {}

    for col in df.columns:
        series = df[col]
        categories: list[str] = []

        # 1. Header heuristic — always cheap, surfaces free-text PII.
        hint = _header_hint(str(col))
        if hint is not None:
            categories.append(hint)

        # 2. Regex scan — skip mostly-null columns.
        non_null_ratio = series.notna().mean() if len(series) else 0.0
        if non_null_ratio >= 0.01:
            for category, pattern in PII_PATTERNS.items():
                if category in categories:
                    continue
                threshold = (
                    LOW_CONF_HIT_THRESHOLD
                    if category in LOW_CONFIDENCE_CATEGORIES
                    else min_hit_ratio
                )
                ratio = _match_column(series, pattern, sample_size)
                if ratio >= threshold:
                    categories.append(category)

        if categories:
            # Deduplicate preserving order
            seen: set[str] = set()
            unique = [c for c in categories if not (c in seen or seen.add(c))]
            detections[str(col)] = unique

    return detections


# --------------------------------------------------------------------------- #
# Internal helpers — synthetic / format-preserving generators
# --------------------------------------------------------------------------- #

def _fake_pan(faker_instance) -> str:
    """Generate a format-preserving synthetic PAN (``AAAPA1234A``)."""
    letters = "".join(faker_instance.random_choices(elements=string.ascii_uppercase, length=3))
    entity = faker_instance.random_element(elements=tuple(_PAN_ENTITY_CHARS))
    fourth_last = faker_instance.random_element(elements=tuple(string.ascii_uppercase))
    digits = "".join(str(faker_instance.random_digit()) for _ in range(4))
    last = faker_instance.random_element(elements=tuple(string.ascii_uppercase))
    return f"{letters}{entity}{fourth_last}{digits}{last}"


def _fake_aadhaar(faker_instance) -> str:
    """Generate a synthetic 12-digit Aadhaar starting with 2-9."""
    first = str(faker_instance.random_int(min=2, max=9))
    rest = "".join(str(faker_instance.random_digit()) for _ in range(11))
    return f"{first}{rest}"


def _fake_gstin(faker_instance) -> str:
    """Generate a synthetic GSTIN (``99AAAAA9999A1Z9``)."""
    state = f"{faker_instance.random_int(min=10, max=37):02d}"
    pan_5 = "".join(faker_instance.random_choices(elements=string.ascii_uppercase, length=5))
    pan_digits = "".join(str(faker_instance.random_digit()) for _ in range(4))
    pan_letter = faker_instance.random_element(elements=tuple(string.ascii_uppercase))
    entity = faker_instance.random_element(
        elements=tuple(string.ascii_uppercase + string.digits)
    )
    checksum = faker_instance.random_element(
        elements=tuple(string.ascii_uppercase + string.digits)
    )
    return f"{state}{pan_5}{pan_digits}{pan_letter}{entity}Z{checksum}"


def _fake_ifsc(faker_instance) -> str:
    """Generate a synthetic IFSC code (``AAAA0XXXXXX``)."""
    bank = "".join(faker_instance.random_choices(elements=string.ascii_uppercase, length=4))
    branch = "".join(
        faker_instance.random_choices(
            elements=string.ascii_uppercase + string.digits, length=6
        )
    )
    return f"{bank}0{branch}"


def _fake_credit_card(faker_instance) -> str:
    """Generate a synthetic 16-digit card number starting with 3-6."""
    first = str(faker_instance.random_int(min=3, max=6))
    rest = "".join(str(faker_instance.random_digit()) for _ in range(15))
    return f"{first}{rest}"


def _fake_pincode(faker_instance) -> str:
    """Generate a synthetic 6-digit Indian pincode starting with 1-9."""
    first = str(faker_instance.random_int(min=1, max=9))
    rest = "".join(str(faker_instance.random_digit()) for _ in range(5))
    return f"{first}{rest}"


# --------------------------------------------------------------------------- #
# Internal helpers — redaction / hashing
# --------------------------------------------------------------------------- #

def _hash_value(value: str, salt: str = "datavaidya") -> str:
    """Return a 10-character SHA-256 digest of ``value`` with ``salt`` prefixed.

    Args:
        value: Source string. Non-strings are coerced via ``str()``.
        salt: Application-wide salt; constant by design so hashes are stable
            across runs within the same deployment.

    Returns:
        Lowercase 10-character hex prefix of the SHA-256 digest.
    """
    if value is None:
        value = ""
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _redact_value(value: str, category: str) -> str:
    """Partially mask ``value`` based on its PII ``category``.

    Examples:
        PAN ``ABCDE1234F`` -> ``ABCDE****F``
        email ``user@domain.com`` -> ``u***@domain.com``
        mobile ``9876543210`` -> ``98******10``
        aadhaar ``234567890123`` -> ``XXXX-XXXX-0123``

    Args:
        value: Original PII string.
        category: PII category key.

    Returns:
        Redacted string. Empty / None values pass through unchanged.
    """
    if value is None:
        return ""
    s = str(value)
    if not s:
        return s

    if category == "email":
        if "@" not in s:
            return "***"
        local, _, domain = s.partition("@")
        if len(local) <= 1:
            masked_local = "***"
        else:
            masked_local = f"{local[0]}***"
        return f"{masked_local}@{domain}"

    if category == "pan":
        # Preserve first 5 + last 1, mask the 4 middle digits
        if len(s) == 10:
            return f"{s[:5]}****{s[9]}"
        return "*" * len(s)

    if category == "mobile":
        digits_only = re.sub(r"\D", "", s)
        if len(digits_only) >= 10:
            tail = digits_only[-10:]
            return f"{tail[:2]}{'*' * 6}{tail[-2:]}"
        return "*" * len(s)

    if category == "aadhaar":
        digits_only = re.sub(r"\D", "", s)
        if len(digits_only) == 12:
            return f"XXXX-XXXX-{digits_only[-4:]}"
        return "*" * len(s)

    if category == "credit_card":
        digits_only = re.sub(r"\D", "", s)
        if len(digits_only) >= 13:
            return f"{'*' * (len(digits_only) - 4)}{digits_only[-4:]}"
        return "*" * len(s)

    if category == "gstin":
        if len(s) == 15:
            return f"{s[:2]}*****{s[7:11]}***"
        return "*" * len(s)

    if category == "ifsc":
        if len(s) == 11:
            return f"{s[:4]}0******"
        return "*" * len(s)

    if category == "pincode":
        digits_only = re.sub(r"\D", "", s)
        if len(digits_only) == 6:
            return f"{digits_only[:2]}****"
        return "*" * len(s)

    if category == "name":
        parts = s.split()
        if not parts:
            return "***"
        return " ".join(p[0] + "***" if p else "***" for p in parts)

    if category == "address":
        if len(s) <= 4:
            return "*" * len(s)
        return f"{s[:3]}{'*' * (len(s) - 3)}"

    # Unknown category — generic masking
    if len(s) <= 2:
        return "*" * len(s)
    return f"{s[0]}{'*' * (len(s) - 2)}{s[-1]}"


# --------------------------------------------------------------------------- #
# Public — masking
# --------------------------------------------------------------------------- #

def _build_faker_generator(faker_instance, category: str):
    """Return a zero-arg callable that produces a synthetic value for ``category``.

    Args:
        faker_instance: A configured ``faker.Faker`` instance.
        category: PII category key.

    Returns:
        Callable returning a string.
    """
    if category == "email":
        return faker_instance.email
    if category == "mobile":
        return lambda: faker_instance.msisdn()[-10:]
    if category == "name":
        return faker_instance.name
    if category == "address":
        return lambda: faker_instance.address().replace("\n", ", ")
    if category == "aadhaar":
        return lambda: _fake_aadhaar(faker_instance)
    if category == "pan":
        return lambda: _fake_pan(faker_instance)
    if category == "gstin":
        return lambda: _fake_gstin(faker_instance)
    if category == "ifsc":
        return lambda: _fake_ifsc(faker_instance)
    if category == "credit_card":
        return lambda: _fake_credit_card(faker_instance)
    if category == "pincode":
        return lambda: _fake_pincode(faker_instance)
    # Unknown category: opaque token
    return lambda: faker_instance.pystr(min_chars=8, max_chars=12)


def mask_pii(
    df: pd.DataFrame,
    plan: dict[str, list[str]],
    method: Literal["faker", "hash", "redact"] = "faker",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Return a copy of ``df`` with PII columns masked according to ``plan``.

    Args:
        df: Source DataFrame. Not mutated.
        plan: Mapping of column name to a list of PII categories (the first
            category in the list drives the masking strategy for that column).
        method: Masking strategy.
            * ``"faker"`` – replace with synthetic values; format-preserving
              for Aadhaar / PAN / GSTIN / IFSC / credit card / pincode.
            * ``"hash"`` – replace with a salted SHA-256 prefix (10 chars).
            * ``"redact"`` – partial masking that retains visual shape.
        seed: Optional deterministic seed for the Faker instance.

    Returns:
        A new DataFrame with the targeted columns rewritten.

    Raises:
        KeyError: If ``plan`` references a column not present in ``df``.
        ValueError: If ``method`` is not one of the supported values.
    """
    if method not in {"faker", "hash", "redact"}:
        raise ValueError(
            f"Unsupported method {method!r}; expected 'faker', 'hash', or 'redact'."
        )

    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    # Validate plan up-front so we fail loud before doing any work.
    missing = [c for c in plan.keys() if c not in df.columns]
    if missing:
        raise KeyError(
            f"mask_pii: columns in plan not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    effective_method = method
    faker_instance = None
    if effective_method == "faker":
        try:
            from faker import Faker  # type: ignore[import-not-found]

            faker_instance = Faker("en_IN")
            if seed is not None:
                Faker.seed(seed)
                faker_instance.seed_instance(seed)
        except ImportError:
            logger.warning(
                "mask_pii: Faker is not installed; falling back to 'redact' method."
            )
            effective_method = "redact"
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "mask_pii: failed to initialize Faker (%s); falling back to 'redact'.",
                exc,
            )
            effective_method = "redact"

    out = df.copy()

    for col, categories in plan.items():
        if not categories:
            continue
        primary = categories[0]

        if effective_method == "hash":
            out[col] = out[col].map(
                lambda v: _hash_value(v) if pd.notna(v) else v
            )
        elif effective_method == "redact":
            out[col] = out[col].map(
                lambda v, cat=primary: _redact_value(v, cat) if pd.notna(v) else v
            )
        else:  # faker
            gen = _build_faker_generator(faker_instance, primary)
            out[col] = out[col].map(lambda v, g=gen: g() if pd.notna(v) else v)

    return out


# --------------------------------------------------------------------------- #
# Public — LLM-safe scrubbing
# --------------------------------------------------------------------------- #

def _redact_freeform_text(text: str, categories: list[str]) -> str:
    """Replace every regex hit in ``text`` with its category placeholder.

    Args:
        text: Free-form string from a DataFrame cell.
        categories: Categories of interest for this cell (drives the order
            in which substitutions are attempted). Header-only categories
            (``name``, ``address``) cause the entire value to be replaced.

    Returns:
        Scrubbed text.
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s

    # For header-only categories (name/address), the cell is replaced wholesale.
    for cat in categories:
        if cat in _HEADER_ONLY_CATEGORIES:
            return _LLM_PLACEHOLDERS.get(cat, "<REDACTED>")

    # Apply category-specific regexes first (preserves placeholder fidelity),
    # then sweep with every pattern for anything missed elsewhere in the cell.
    for cat in categories:
        pattern = PII_PATTERNS.get(cat)
        placeholder = _LLM_PLACEHOLDERS.get(cat, "<REDACTED>")
        if pattern is not None:
            s = pattern.sub(placeholder, s)

    # Defensive sweep: any cell may still carry PII for an unrelated category
    # (e.g. an email accidentally inside an address column).
    for cat, pattern in PII_PATTERNS.items():
        placeholder = _LLM_PLACEHOLDERS.get(cat, "<REDACTED>")
        s = pattern.sub(placeholder, s)

    return s


def _format_schema_table(df: pd.DataFrame) -> str:
    """Render a Markdown schema table (column, dtype, non_null %)."""
    rows: list[str] = [
        "| Column | Dtype | Non-null % |",
        "| --- | --- | --- |",
    ]
    total = len(df)
    for col in df.columns:
        dtype = str(df[col].dtype)
        if total > 0:
            non_null_pct = f"{(df[col].notna().mean() * 100):.1f}%"
        else:
            non_null_pct = "0.0%"
        rows.append(f"| {col} | {dtype} | {non_null_pct} |")
    return "\n".join(rows)


def _format_sample_rows(
    df: pd.DataFrame,
    detections: dict[str, list[str]],
    n_rows: int,
) -> str:
    """Render ``n_rows`` sample rows as Markdown with PII redacted."""
    if df.empty or n_rows <= 0:
        return ""

    sample = df.head(n_rows).copy()

    # Scrub each cell. PII columns get aggressive replacement; other columns
    # still get the defensive regex sweep in case PII leaked into them.
    for col in sample.columns:
        cats = detections.get(str(col), [])
        sample[col] = sample[col].map(
            lambda v, c=cats: _redact_freeform_text(v, c) if pd.notna(v) else ""
        )

    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    separator = "| " + " | ".join("---" for _ in sample.columns) + " |"
    body_lines: list[str] = []
    for _, row in sample.iterrows():
        cells = []
        for v in row.values:
            cell = "" if v is None else str(v)
            # Escape pipes so we don't break Markdown table rendering.
            cell = cell.replace("|", "\\|").replace("\n", " ")
            cells.append(cell)
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator, *body_lines])


def scrub_for_llm(df: pd.DataFrame, n_sample_rows: int = 3) -> str:
    """Return a Markdown summary of ``df`` safe to send to an LLM.

    The output contains:
        * A shape line (``X rows × Y columns``).
        * A schema table (column, dtype, non-null %).
        * Up to ``n_sample_rows`` sample rows with PII fully redacted to
          placeholders such as ``<EMAIL>``, ``<PHONE>``, ``<AADHAAR>``,
          ``<PAN>``, ``<GSTIN>``, ``<IFSC>``, ``<CARD>``.

    If the rendered output exceeds ~3500 tokens (estimated as chars / 4),
    sample rows are trimmed down (eventually to zero) until the budget is
    respected. The schema is preserved.

    Args:
        df: Source DataFrame.
        n_sample_rows: Maximum sample rows to include.

    Returns:
        Markdown string. Empty/None DataFrame yields ``""``.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return ""

    detections = detect_pii(df)
    shape_line = f"**Shape:** {len(df)} rows × {len(df.columns)} columns"
    schema_md = _format_schema_table(df)

    def _assemble(n: int) -> str:
        parts = [shape_line, "", "### Schema", schema_md]
        if n > 0:
            sample_md = _format_sample_rows(df, detections, n)
            if sample_md:
                parts += ["", f"### Sample rows (n={min(n, len(df))}, PII redacted)", sample_md]
        return "\n".join(parts)

    output = _assemble(n_sample_rows)
    current_n = n_sample_rows
    while len(output) > _LLM_TOKEN_BUDGET_CHARS and current_n > 0:
        current_n -= 1
        output = _assemble(current_n)

    return output


# --------------------------------------------------------------------------- #
# Public — convenience summary
# --------------------------------------------------------------------------- #

def get_pii_summary(df: pd.DataFrame) -> dict:
    """Summarize PII detections across ``df``.

    Args:
        df: Source DataFrame.

    Returns:
        Dict with three keys:
            * ``total_pii_columns`` – number of columns with at least one
              detected PII category.
            * ``by_category`` – mapping of category -> count of columns in
              which it was detected.
            * ``detections`` – the raw output of :func:`detect_pii`.
    """
    detections = detect_pii(df) if df is not None else {}
    by_category: dict[str, int] = {}
    for cats in detections.values():
        for c in cats:
            by_category[c] = by_category.get(c, 0) + 1
    return {
        "total_pii_columns": len(detections),
        "by_category": by_category,
        "detections": detections,
    }
