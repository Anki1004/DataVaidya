"""Memory introspection, guarding, and DataFrame-footprint utilities.

This module centralizes everything DataVaidya needs to know about process
memory: live RSS sampling via :mod:`psutil`, DataFrame footprint estimation
and dtype-downcast suggestions, viz-time sampling, a context manager that
logs RSS deltas around expensive operations, and a decorator that aborts
function calls if RSS is already above a configurable threshold.

The Streamlit widget intentionally imports ``streamlit`` lazily so the rest
of the module remains importable in headless test environments where
Streamlit may not be installed.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
)

import numpy as np
import pandas as pd
import psutil

from utils.constants import (
    ERROR_RED,
    MEMORY_BUDGET_MB,
    MEMORY_CRITICAL_PCT,
    MEMORY_GUARD_MB,
    MEMORY_WARN_PCT,
    SUCCESS_GREEN,
    TEXT_MUTED,
    VIZ_SAMPLE_ROWS,
    WARN_FILE_MB,
    WARNING_AMBER,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

ThresholdStatus = Literal["ok", "warning", "critical"]

# Float32 can represent values up to ~3.4e38 but only ~7 decimal digits of
# precision. We use this magnitude bound when deciding if a float64 column
# can be safely downcast.
_FLOAT32_MAX_ABS: float = float(np.finfo(np.float32).max)
# Cardinality ratio under which we recommend converting an object column to
# pandas ``category`` dtype. Empirically, below ~0.5 the savings dominate.
_CATEGORY_RATIO_THRESHOLD: float = 0.5


class MemoryStats(TypedDict):
    """Snapshot of process memory relative to the configured budget.

    Attributes:
        rss_mb: Resident set size of the current process, in megabytes.
        available_mb: Remaining headroom under :data:`MEMORY_BUDGET_MB`,
            clamped at zero.
        percent_used: ``rss_mb / MEMORY_BUDGET_MB * 100``.
        threshold_status: ``"ok"`` (<70%), ``"warning"`` (70-85%),
            ``"critical"`` (>=85%).
    """

    rss_mb: float
    available_mb: float
    percent_used: float
    threshold_status: ThresholdStatus


class MemoryGuardError(RuntimeError):
    """Raised when a memory-guarded operation is refused pre-flight."""


def _classify(percent_used: float) -> ThresholdStatus:
    """Map a percent-used reading to a threshold status string.

    Args:
        percent_used: Memory usage as a percentage of the budget.

    Returns:
        ``"critical"`` if at or above :data:`MEMORY_CRITICAL_PCT`,
        ``"warning"`` if at or above :data:`MEMORY_WARN_PCT`, else ``"ok"``.
    """
    if percent_used >= MEMORY_CRITICAL_PCT:
        return "critical"
    if percent_used >= MEMORY_WARN_PCT:
        return "warning"
    return "ok"


def get_memory_stats() -> MemoryStats:
    """Sample the current process's memory usage against the configured budget.

    Returns:
        A :class:`MemoryStats` dict with RSS in MB, remaining headroom
        in MB, percent used, and a classified threshold status.

    Raises:
        psutil.NoSuchProcess: If the current process cannot be inspected
            (extremely rare; typically only during interpreter shutdown).
    """
    process = psutil.Process(os.getpid())
    rss_bytes = process.memory_info().rss
    rss_mb = rss_bytes / (1024.0 * 1024.0)
    available_mb = max(0.0, MEMORY_BUDGET_MB - rss_mb)
    percent_used = (rss_mb / MEMORY_BUDGET_MB) * 100.0 if MEMORY_BUDGET_MB else 0.0
    return MemoryStats(
        rss_mb=rss_mb,
        available_mb=available_mb,
        percent_used=percent_used,
        threshold_status=_classify(percent_used),
    )


def check_memory_pressure() -> tuple[str, float]:
    """Return a quick ``(status, percent_used)`` reading.

    This is a lightweight helper for hot paths that need to decide whether
    to defer work without unpacking the full :class:`MemoryStats` dict.

    Returns:
        Tuple of ``(threshold_status, percent_used)``.
    """
    stats = get_memory_stats()
    return stats["threshold_status"], stats["percent_used"]


def estimate_df_memory(df: pd.DataFrame, *, deep: bool = True) -> float:
    """Estimate a DataFrame's in-memory footprint in megabytes.

    Args:
        df: The DataFrame to measure.
        deep: If ``True`` (default), introspect object dtypes for an accurate
            string-size accounting. Set to ``False`` for a cheap upper-bound.

    Returns:
        Memory footprint in megabytes.

    Raises:
        TypeError: If ``df`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")
    total_bytes = int(df.memory_usage(deep=deep).sum())
    return total_bytes / 1024.0 / 1024.0


def _suggest_dtype(series: pd.Series) -> str | None:
    """Suggest a more compact dtype for a single column, or ``None``.

    Args:
        series: The column to analyze.

    Returns:
        The name of a recommended dtype (e.g., ``"int32"``, ``"float32"``,
        ``"category"``), or ``None`` if the column is already optimal.
    """
    dtype = series.dtype

    # Integer downcast: int64 -> int32 if range fits.
    if pd.api.types.is_integer_dtype(dtype) and dtype == np.int64:
        non_null = series.dropna()
        if non_null.empty:
            return "int32"
        try:
            col_min = int(non_null.min())
            col_max = int(non_null.max())
        except (ValueError, OverflowError):
            return None
        if np.iinfo(np.int32).min <= col_min and col_max <= np.iinfo(np.int32).max:
            return "int32"
        return None

    # Float downcast: float64 -> float32 if magnitude fits. We accept the
    # ~7-digit precision loss as a deliberate tradeoff for the savings.
    if pd.api.types.is_float_dtype(dtype) and dtype == np.float64:
        non_null = series.dropna()
        if non_null.empty:
            return "float32"
        try:
            max_abs = float(non_null.abs().max())
        except (ValueError, OverflowError):
            return None
        if max_abs <= _FLOAT32_MAX_ABS:
            return "float32"
        return None

    # Object -> category if low cardinality.
    if dtype == object:
        length = len(series)
        if length == 0:
            return None
        try:
            nunique = int(series.nunique(dropna=True))
        except TypeError:
            # Unhashable values (e.g., dicts in cells) — skip silently.
            return None
        if nunique / length < _CATEGORY_RATIO_THRESHOLD:
            return "category"
        return None

    return None


def _estimate_savings_mb(series: pd.Series, target_dtype: str) -> float:
    """Estimate MB saved by converting ``series`` to ``target_dtype``.

    Args:
        series: The source column.
        target_dtype: The proposed dtype name.

    Returns:
        Estimated savings in megabytes. Never negative.
    """
    current = float(series.memory_usage(deep=True))
    try:
        if target_dtype == "category":
            converted = series.astype("category")
        else:
            converted = series.astype(target_dtype)
        projected = float(converted.memory_usage(deep=True))
    except (ValueError, TypeError, OverflowError) as exc:
        logger.debug(
            "Could not estimate savings for dtype %s on column %s: %s",
            target_dtype,
            series.name,
            exc,
        )
        return 0.0
    return max(0.0, (current - projected) / 1024.0 / 1024.0)


def df_memory_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a per-column memory summary with dtype-downcast suggestions.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A dict with keys:

        * ``"total_mb"`` (float): current total footprint.
        * ``"columns"`` (list[dict]): one entry per column with keys
          ``column``, ``mb``, ``dtype``, and ``suggested_dtype`` (may be
          ``None``).
        * ``"suggested_savings_mb"`` (float): sum of per-column savings if
          all suggestions were applied.

    Raises:
        TypeError: If ``df`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")

    per_column_usage = df.memory_usage(deep=True)
    total_mb = float(per_column_usage.sum()) / 1024.0 / 1024.0

    columns: list[dict[str, Any]] = []
    total_savings_mb = 0.0

    for col_name in df.columns:
        series = df[col_name]
        col_bytes = float(per_column_usage.get(col_name, 0))
        col_mb = col_bytes / 1024.0 / 1024.0
        suggested = _suggest_dtype(series)
        savings = _estimate_savings_mb(series, suggested) if suggested else 0.0
        total_savings_mb += savings
        columns.append(
            {
                "column": col_name,
                "mb": col_mb,
                "dtype": str(series.dtype),
                "suggested_dtype": suggested,
            }
        )

    return {
        "total_mb": total_mb,
        "columns": columns,
        "suggested_savings_mb": total_savings_mb,
    }


def sample_for_viz(
    df: pd.DataFrame,
    max_rows: int = VIZ_SAMPLE_ROWS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deterministic random sample suitable for visualization.

    If the DataFrame already has at most ``max_rows`` rows, the original
    object is returned unchanged (no copy). Otherwise a reproducible random
    sample is taken using ``random_state``.

    Args:
        df: The source DataFrame.
        max_rows: Maximum number of rows to return. Must be >= 1.
        random_state: Seed for reproducibility.

    Returns:
        The original DataFrame, or a random sample of ``max_rows`` rows.

    Raises:
        TypeError: If ``df`` is not a :class:`pandas.DataFrame`.
        ValueError: If ``max_rows`` is less than 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")

    if len(df) <= max_rows:
        return df

    logger.info(
        "Sampling DataFrame for viz: %d -> %d rows (seed=%d)",
        len(df),
        max_rows,
        random_state,
    )
    return df.sample(n=max_rows, random_state=random_state)


def should_sample(file_size_mb: float) -> bool:
    """Return whether a file of the given size warrants pre-emptive sampling.

    Args:
        file_size_mb: File size in megabytes. Negative values are treated
            as zero.

    Returns:
        ``True`` if ``file_size_mb`` exceeds :data:`WARN_FILE_MB`.
    """
    return max(0.0, file_size_mb) > WARN_FILE_MB


def force_gc() -> int:
    """Run a full garbage-collection cycle and return the count of unreachable objects.

    Returns:
        The number of unreachable objects found by :func:`gc.collect`.
    """
    before = get_memory_stats()["rss_mb"]
    collected = gc.collect()
    after = get_memory_stats()["rss_mb"]
    logger.debug(
        "force_gc: collected %d objects, RSS %.2f -> %.2f MB",
        collected,
        before,
        after,
    )
    return collected


def render_memory_widget() -> None:
    """Render a memory-usage progress bar in the Streamlit sidebar.

    The widget shows a progress bar tinted by threshold status and a caption
    with the raw RSS / budget figures. Streamlit is imported lazily so the
    rest of this module remains importable in headless environments. Callers
    that want auto-refresh should wrap this function with
    ``@st.fragment(run_every="10s")`` at the call site.

    This function is a no-op if Streamlit is not installed (a warning is
    logged once).
    """
    try:
        import streamlit as st  # noqa: PLC0415 (intentional lazy import)
    except ImportError:
        logger.warning(
            "Streamlit is not installed; render_memory_widget() is a no-op."
        )
        return

    stats = get_memory_stats()
    status = stats["threshold_status"]
    percent = stats["percent_used"]
    rss_mb = stats["rss_mb"]
    available_mb = stats["available_mb"]

    if status == "critical":
        color = ERROR_RED
        label = "Critical"
    elif status == "warning":
        color = WARNING_AMBER
        label = "High"
    else:
        color = SUCCESS_GREEN
        label = "OK"

    # st.progress expects 0..100 int; clamp for safety.
    progress_value = int(max(0.0, min(100.0, percent)))

    st.markdown(
        f"<div style='color:{color};font-weight:600;'>Memory: {label}</div>",
        unsafe_allow_html=True,
    )
    st.progress(progress_value, text=f"{percent:.1f}% of {MEMORY_BUDGET_MB} MB")
    st.caption(
        f"<span style='color:{TEXT_MUTED};'>"
        f"RSS {rss_mb:.0f} MB &middot; {available_mb:.0f} MB free"
        f"</span>",
        unsafe_allow_html=True,
    )


@contextmanager
def memory_guard(operation: str) -> Iterator[None]:
    """Context manager that logs RSS deltas and contextualizes :class:`MemoryError`.

    Args:
        operation: Short label for the operation being guarded, used in log
            messages and exception context.

    Yields:
        ``None``.

    Raises:
        MemoryGuardError: If the guarded block raises :class:`MemoryError`,
            wrapped to preserve the original traceback and add RSS context.
    """
    before = get_memory_stats()
    logger.debug(
        "memory_guard[%s] entered: RSS=%.2f MB (%s)",
        operation,
        before["rss_mb"],
        before["threshold_status"],
    )
    try:
        yield
    except MemoryError as exc:
        after = get_memory_stats()
        msg = (
            f"MemoryError during '{operation}': "
            f"RSS {before['rss_mb']:.0f} -> {after['rss_mb']:.0f} MB "
            f"({after['percent_used']:.1f}% of {MEMORY_BUDGET_MB} MB budget)."
        )
        logger.exception(msg)
        raise MemoryGuardError(msg) from exc
    else:
        after = get_memory_stats()
        delta = after["rss_mb"] - before["rss_mb"]
        logger.debug(
            "memory_guard[%s] exited: RSS=%.2f MB (delta %+.2f MB)",
            operation,
            after["rss_mb"],
            delta,
        )


def with_memory_guard(
    threshold_mb: float = MEMORY_GUARD_MB,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory that aborts calls if RSS is already above a threshold.

    The check happens *before* the wrapped function is invoked, so it
    protects against piling more work onto a process that is already near
    its limit.

    Args:
        threshold_mb: RSS threshold in megabytes. Defaults to
            :data:`MEMORY_GUARD_MB`.

    Returns:
        A decorator that wraps a function with a pre-flight memory check.

    Raises:
        ValueError: If ``threshold_mb`` is not positive.
    """
    if threshold_mb <= 0:
        raise ValueError(f"threshold_mb must be positive, got {threshold_mb}")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            stats = get_memory_stats()
            if stats["rss_mb"] > threshold_mb:
                msg = (
                    f"Refusing to invoke '{func.__qualname__}': "
                    f"RSS {stats['rss_mb']:.0f} MB exceeds guard threshold "
                    f"{threshold_mb:.0f} MB "
                    f"({stats['percent_used']:.1f}% of budget)."
                )
                logger.error(msg)
                raise MemoryGuardError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
