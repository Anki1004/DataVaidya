"""Dataset profiling and health scoring.

This module computes data quality diagnostics for a ``pandas.DataFrame``,
including missingness, duplicates, outliers, class imbalance, correlation,
distribution summaries, cardinality, constant columns, mixed dtypes,
date-string consistency, and memory footprint. The ``compute_health_score``
function aggregates these into a single 0-100 score with categorized
deductions and a health zone.

All functions are defensive: empty frames, single-row frames, all-NaN
columns, zero-variance numerics, and unhashable cell contents are handled
without raising. ``compute_health_score`` itself never raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.constants import (
    CARDINALITY_MAX,
    CONSTANT_MAX,
    DATE_INCONSISTENCY_MAX,
    DUPLICATES_MAX,
    HIGH_CARDINALITY_MIN_UNIQUE,
    HIGH_CARDINALITY_RATIO,
    IMBALANCE_GINI_THRESHOLD,
    IMBALANCE_MAX,
    IQR_MULTIPLIER,
    MISSING_MAX,
    MIXED_DTYPE_MAX,
    OUTLIERS_MAX,
    ZSCORE_THRESHOLD,
    health_zone,
)
from utils.memory import memory_guard

logger = logging.getLogger(__name__)


class HealthReport(TypedDict):
    """Structured result returned by :func:`compute_health_score`.

    Attributes:
        score: Integer 0-100 (floored) representing overall data health.
        zone: Categorical health zone string derived from ``health_zone``.
        deductions: Per-category deduction amounts, rounded to 2 decimals.
        reasons: Human-readable explanation per category.
        affected_columns: Column names implicated per category.
    """

    score: int
    zone: str
    deductions: dict[str, float]
    reasons: dict[str, str]
    affected_columns: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric column names, excluding booleans.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with a numeric dtype (int/float).
    """
    if df is None or df.empty:
        return []
    cols: list[str] = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                cols.append(str(col))
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping column %s in numeric detection: %s", col, exc)
    return cols


def _safe_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Return categorical-like column names (object, category, bool).

    Args:
        df: Input DataFrame.

    Returns:
        List of column names that look categorical.
    """
    if df is None or df.empty:
        return []
    cols: list[str] = []
    for col in df.columns:
        try:
            series = df[col]
            if (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)
            ):
                cols.append(str(col))
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping column %s in categorical detection: %s", col, exc)
    return cols


def _iqr_bounds(s: pd.Series) -> tuple[float, float]:
    """Compute IQR lower/upper outlier bounds.

    Args:
        s: Numeric Series.

    Returns:
        Tuple ``(lower, upper)``. Returns ``(nan, nan)`` for too-small series.
    """
    s_clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(s_clean) < 4:
        return (float("nan"), float("nan"))
    q1 = float(s_clean.quantile(0.25))
    q3 = float(s_clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    return (lower, upper)


def _iqr_outlier_count(s: pd.Series) -> int:
    """Count IQR outliers in a numeric series.

    Args:
        s: Numeric Series.

    Returns:
        Number of values outside [Q1 - k*IQR, Q3 + k*IQR].
    """
    s_clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(s_clean) < 4:
        return 0
    lower, upper = _iqr_bounds(s_clean)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return 0
    mask = (s_clean < lower) | (s_clean > upper)
    return int(mask.sum())


def _zscore_outlier_count(s: pd.Series) -> int:
    """Count z-score outliers, guarding against zero variance.

    Args:
        s: Numeric Series.

    Returns:
        Number of values with ``|z| > ZSCORE_THRESHOLD``. Zero if std==0.
    """
    s_clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(s_clean) < 2:
        return 0
    std = float(s_clean.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return 0
    mean = float(s_clean.mean())
    z = (s_clean - mean).abs() / std
    return int((z > ZSCORE_THRESHOLD).sum())


def _inequality_gini(counts: np.ndarray) -> float:
    """Inequality Gini coefficient for non-negative counts.

    Implements ``G = (2 * sum(i * x_i_sorted_asc) / (k * sum(x_i))) - (k + 1) / k``.

    Args:
        counts: 1-D array of non-negative class counts.

    Returns:
        Gini coefficient in approximately ``[0, 1)``. Returns 0 on degenerate input.
    """
    arr = np.asarray(counts, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    k = arr.size
    total = arr.sum()
    if k <= 1 or total <= 0:
        return 0.0
    sorted_arr = np.sort(arr)
    indices = np.arange(1, k + 1, dtype=float)
    gini = (2.0 * np.sum(indices * sorted_arr) / (k * total)) - (k + 1.0) / k
    return float(max(0.0, min(1.0, gini)))


def _is_mixed_dtype(s: pd.Series) -> bool:
    """Detect mixed-type object columns via ``pd.api.types.infer_dtype``.

    Args:
        s: Series (typically object-dtype).

    Returns:
        ``True`` when inferred dtype is mixed-ish.
    """
    if not pd.api.types.is_object_dtype(s):
        return False
    try:
        inferred = pd.api.types.infer_dtype(s, skipna=True)
    except (TypeError, ValueError) as exc:
        logger.debug("infer_dtype failed for %s: %s", s.name, exc)
        return False
    return inferred in {"mixed", "mixed-integer", "mixed-integer-float"}


def _is_constant(s: pd.Series) -> bool:
    """Return True if column has at most one unique non-null and is not all NaN.

    Args:
        s: Series to inspect.

    Returns:
        ``True`` when ``nunique(dropna=True) <= 1`` and the column has at
        least one non-null value.
    """
    try:
        non_null = s.notna().sum()
        if non_null == 0:
            return False
        return int(s.nunique(dropna=True)) <= 1
    except (TypeError, ValueError) as exc:
        logger.debug("Constant check failed for %s: %s", s.name, exc)
        return False


def _is_high_cardinality(s: pd.Series) -> bool:
    """Detect high-cardinality categorical-like columns.

    A column is high-cardinality when it has at least
    ``HIGH_CARDINALITY_MIN_UNIQUE`` unique values AND
    ``nunique / len >= HIGH_CARDINALITY_RATIO``.

    Args:
        s: Series to inspect.

    Returns:
        ``True`` when both conditions are met.
    """
    try:
        n = len(s)
        if n == 0:
            return False
        try:
            unique = int(s.nunique(dropna=True))
        except TypeError:
            unique = int(s.astype(str).nunique(dropna=True))
        if unique < HIGH_CARDINALITY_MIN_UNIQUE:
            return False
        return (unique / n) >= HIGH_CARDINALITY_RATIO
    except (TypeError, ValueError) as exc:
        logger.debug("High-cardinality check failed for %s: %s", s.name, exc)
        return False


# ---------------------------------------------------------------------------
# Public summaries
# ---------------------------------------------------------------------------


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns ``[column, missing_count, missing_pct]``
        sorted by ``missing_pct`` descending.
    """
    cols = ["column", "missing_count", "missing_pct"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    n = len(df)
    rows: list[dict[str, object]] = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        pct = (miss / n * 100.0) if n > 0 else 0.0
        rows.append({"column": str(col), "missing_count": miss, "missing_pct": round(pct, 4)})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("missing_pct", ascending=False, kind="mergesort").reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Summarize fully-duplicated rows.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with keys ``total`` (int), ``pct`` (float, percentage of rows
        flagged as duplicates), and ``sample`` (head of duplicate rows).
    """
    empty = {"total": 0, "pct": 0.0, "sample": pd.DataFrame()}
    if df is None or df.empty:
        return empty
    n = len(df)
    try:
        dup_mask = df.duplicated(keep=False)
    except TypeError:
        try:
            dup_mask = df.astype(str).duplicated(keep=False)
        except (TypeError, ValueError) as exc:
            logger.warning("Duplicate detection fallback failed: %s", exc)
            return empty
    total = int(dup_mask.sum())
    pct = (total / n * 100.0) if n > 0 else 0.0
    sample = df.loc[dup_mask].head(5).copy() if total > 0 else pd.DataFrame()
    return {"total": total, "pct": round(pct, 4), "sample": sample}


def outlier_summary(
    df: pd.DataFrame,
    method: Literal["iqr", "zscore", "both"] = "iqr",
) -> pd.DataFrame:
    """Per-numeric-column outlier counts.

    Args:
        df: Input DataFrame.
        method: ``"iqr"``, ``"zscore"``, or ``"both"``. Both counts are
            always computed; the parameter is retained for API stability.

    Returns:
        DataFrame with columns
        ``[column, iqr_outliers, zscore_outliers, lower_bound, upper_bound]``.
        Columns with ``len < 4`` or ``std == 0`` are skipped.
    """
    cols = ["column", "iqr_outliers", "zscore_outliers", "lower_bound", "upper_bound"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    if method not in {"iqr", "zscore", "both"}:
        raise ValueError(f"Unsupported method: {method!r}")
    numeric_cols = _safe_numeric_cols(df)
    rows: list[dict[str, object]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        std = float(series.std(ddof=0))
        if std == 0 or not np.isfinite(std):
            # zero-variance: skip per spec
            continue
        lower, upper = _iqr_bounds(series)
        iqr_count = _iqr_outlier_count(series)
        z_count = _zscore_outlier_count(series)
        rows.append(
            {
                "column": col,
                "iqr_outliers": iqr_count,
                "zscore_outliers": z_count,
                "lower_bound": round(lower, 6) if np.isfinite(lower) else float("nan"),
                "upper_bound": round(upper, 6) if np.isfinite(upper) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def imbalance_summary(df: pd.DataFrame, max_cardinality: int = 50) -> pd.DataFrame:
    """Class-imbalance metrics for low-cardinality categorical columns.

    For each categorical column with ``nunique <= max_cardinality``, compute
    the inequality Gini of class counts and a chi-squared p-value against a
    uniform null. ``is_imbalanced`` is ``True`` when ``gini > IMBALANCE_GINI_THRESHOLD``.

    Args:
        df: Input DataFrame.
        max_cardinality: Skip columns above this distinct-value count.

    Returns:
        DataFrame with columns
        ``[column, n_classes, gini, chi2_p_value, is_imbalanced]``.
    """
    cols = ["column", "n_classes", "gini", "chi2_p_value", "is_imbalanced"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    cat_cols = _safe_categorical_cols(df)
    rows: list[dict[str, object]] = []
    for col in cat_cols:
        try:
            vc = df[col].dropna().value_counts()
        except TypeError:
            vc = df[col].astype(str).dropna().value_counts()
        k = int(vc.size)
        if k < 2 or k > max_cardinality:
            continue
        counts = vc.to_numpy(dtype=float)
        gini = _inequality_gini(counts)
        p_val = float("nan")
        total = float(counts.sum())
        if total > 0 and k > 1:
            expected = np.full(k, total / k, dtype=float)
            try:
                chi2_res = scipy_stats.chisquare(f_obs=counts, f_exp=expected)
                p_val = float(chi2_res.pvalue)
            except (ValueError, ZeroDivisionError) as exc:
                logger.debug("Chi-squared failed for %s: %s", col, exc)
                p_val = float("nan")
        rows.append(
            {
                "column": col,
                "n_classes": k,
                "gini": round(gini, 6),
                "chi2_p_value": round(p_val, 6) if np.isfinite(p_val) else float("nan"),
                "is_imbalanced": bool(gini > IMBALANCE_GINI_THRESHOLD),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def correlation_matrix(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """Compute a numeric-only correlation matrix.

    Args:
        df: Input DataFrame.
        method: ``"pearson"`` or ``"spearman"``.

    Returns:
        Square correlation DataFrame, or an empty DataFrame when fewer than
        two numeric columns are present.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"Unsupported correlation method: {method!r}")
    if df is None or df.empty:
        return pd.DataFrame()
    num_cols = _safe_numeric_cols(df)
    if len(num_cols) < 2:
        return pd.DataFrame()
    try:
        return df[num_cols].corr(method=method, numeric_only=True)
    except (ValueError, TypeError) as exc:
        logger.warning("Correlation computation failed: %s", exc)
        return pd.DataFrame()


def top_correlations(df: pd.DataFrame, top_n: int = 5) -> list[tuple[str, str, float]]:
    """Return strongest unique pairwise correlations by absolute value.

    Args:
        df: Input DataFrame.
        top_n: Maximum number of pairs to return.

    Returns:
        List of ``(col_a, col_b, correlation)`` triples, sorted descending
        by ``abs(correlation)``. Self-pairs are excluded; pairs with NaN
        correlation are filtered.
    """
    if top_n <= 0:
        return []
    corr = correlation_matrix(df)
    if corr.empty:
        return []
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.at[a, b]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fval):
                continue
            pairs.append((str(a), str(b), fval))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[:top_n]


def distribution_summary(df: pd.DataFrame) -> dict[str, dict]:
    """Compute descriptive distribution stats for numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        Mapping of column -> stats dict with keys ``mean``, ``std``,
        ``min``, ``max``, ``q25``, ``q50``, ``q75``, ``skew``, ``kurtosis``.
    """
    out: dict[str, dict] = {}
    if df is None or df.empty:
        return out
    for col in _safe_numeric_cols(df):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        try:
            stats_dict = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.50)),
                "q75": float(series.quantile(0.75)),
                "skew": float(series.skew()) if len(series) > 2 else 0.0,
                "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
            }
        except (ValueError, TypeError) as exc:
            logger.debug("Distribution stats failed for %s: %s", col, exc)
            continue
        out[col] = stats_dict
    return out


def cardinality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column cardinality and uniqueness diagnostics.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns
        ``[column, unique_count, uniqueness_ratio, dtype, is_high_cardinality]``.
    """
    cols = ["column", "unique_count", "uniqueness_ratio", "dtype", "is_high_cardinality"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    n = len(df)
    rows: list[dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        try:
            unique = int(series.nunique(dropna=True))
        except TypeError:
            unique = int(series.astype(str).nunique(dropna=True))
        ratio = (unique / n) if n > 0 else 0.0
        rows.append(
            {
                "column": str(col),
                "unique_count": unique,
                "uniqueness_ratio": round(ratio, 6),
                "dtype": str(series.dtype),
                "is_high_cardinality": _is_high_cardinality(series),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def constant_columns(df: pd.DataFrame) -> list[str]:
    """List columns with at most one unique non-null value.

    Args:
        df: Input DataFrame.

    Returns:
        Column names where ``nunique(dropna=True) <= 1`` and the column has
        at least one non-null value.
    """
    if df is None or df.empty:
        return []
    return [str(c) for c in df.columns if _is_constant(df[c])]


def mixed_dtype_columns(df: pd.DataFrame) -> list[str]:
    """List object columns with mixed inferred dtype.

    Args:
        df: Input DataFrame.

    Returns:
        Column names where ``pd.api.types.infer_dtype`` returns
        ``mixed``, ``mixed-integer``, or ``mixed-integer-float``.
    """
    if df is None or df.empty:
        return []
    return [str(c) for c in df.columns if _is_mixed_dtype(df[c])]


def date_consistency_check(df: pd.DataFrame) -> dict[str, list[str]]:
    """Detect inconsistently formatted date-string columns.

    A column is treated as a suspected date column when more than 50% of
    non-null values successfully parse via ``pd.to_datetime`` with
    ``errors='coerce'``. The column is reported when more than 10% of
    non-null values fail to parse, along with up to 5 sample unparseable
    values.

    Args:
        df: Input DataFrame.

    Returns:
        Mapping of column name -> list of unparseable sample strings.
    """
    out: dict[str, list[str]] = {}
    if df is None or df.empty:
        return out
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        n = len(non_null)
        if n == 0:
            continue
        try:
            parsed = pd.to_datetime(non_null, errors="coerce")
        except (ValueError, TypeError) as exc:
            logger.debug("to_datetime failed for %s: %s", col, exc)
            continue
        parsed_ok = parsed.notna().sum()
        if parsed_ok / n <= 0.5:
            continue
        failed_mask = parsed.isna()
        failed_count = int(failed_mask.sum())
        if failed_count / n <= 0.1:
            continue
        samples = (
            non_null[failed_mask.values].astype(str).head(5).tolist()
            if failed_count > 0
            else []
        )
        out[str(col)] = samples
    return out


def memory_footprint(df: pd.DataFrame) -> dict[str, float]:
    """Return per-column memory usage in megabytes (deep).

    Args:
        df: Input DataFrame.

    Returns:
        Mapping of column name -> memory in MB. Includes ``Index`` when
        ``pandas`` reports it.
    """
    if df is None or df.empty:
        return {}
    try:
        usage = df.memory_usage(deep=True)
    except (ValueError, TypeError) as exc:
        logger.warning("memory_usage failed: %s", exc)
        return {}
    return {str(idx): round(float(val) / (1024.0 * 1024.0), 6) for idx, val in usage.items()}


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------


@dataclass
class _DeductionResult:
    """Internal container for a single category's deduction outcome."""

    value: float
    reason: str
    affected: list[str]


def _deduct_missing(df: pd.DataFrame) -> _DeductionResult:
    """Compute the missingness deduction."""
    n_rows, n_cols = df.shape
    cells = n_rows * n_cols
    if cells == 0:
        return _DeductionResult(0.0, "", [])
    total_nan = int(df.isna().sum().sum())
    ratio = total_nan / cells
    deduction = min(float(MISSING_MAX), 60.0 * ratio)
    affected = [str(c) for c in df.columns if df[c].isna().any()]
    reason = (
        f"{total_nan} missing values ({ratio * 100:.2f}% of cells) across "
        f"{len(affected)} column(s)."
        if total_nan > 0
        else ""
    )
    return _DeductionResult(deduction, reason, affected)


def _deduct_duplicates(df: pd.DataFrame) -> _DeductionResult:
    """Compute the duplicate-rows deduction."""
    n_rows = len(df)
    if n_rows == 0:
        return _DeductionResult(0.0, "", [])
    summary = duplicate_summary(df)
    dup_count = int(summary.get("total", 0))
    deduction = min(float(DUPLICATES_MAX), 50.0 * dup_count / n_rows) if n_rows > 0 else 0.0
    reason = (
        f"{dup_count} duplicated row(s) ({summary.get('pct', 0.0):.2f}%)."
        if dup_count > 0
        else ""
    )
    return _DeductionResult(deduction, reason, [])


def _deduct_outliers(df: pd.DataFrame) -> _DeductionResult:
    """Compute the outlier deduction."""
    n_rows = len(df)
    numeric_cols = _safe_numeric_cols(df)
    if n_rows < 2 or not numeric_cols:
        return _DeductionResult(0.0, "", [])
    ratios: list[float] = []
    affected: list[str] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        std = float(series.std(ddof=0))
        if std == 0 or not np.isfinite(std):
            continue
        count = _iqr_outlier_count(series)
        n_eff = len(series)
        if n_eff == 0:
            continue
        ratio = count / n_eff
        ratios.append(ratio)
        if count > 0:
            affected.append(col)
    if not ratios:
        return _DeductionResult(0.0, "", [])
    mean_ratio = float(np.mean(ratios))
    deduction = min(float(OUTLIERS_MAX), 100.0 * mean_ratio)
    reason = (
        f"Mean IQR-outlier ratio {mean_ratio * 100:.2f}% across "
        f"{len(ratios)} numeric column(s)."
        if mean_ratio > 0
        else ""
    )
    return _DeductionResult(deduction, reason, affected)


def _deduct_imbalance(df: pd.DataFrame) -> _DeductionResult:
    """Compute the class-imbalance deduction."""
    if len(df) < 2:
        return _DeductionResult(0.0, "", [])
    cat_cols = _safe_categorical_cols(df)
    if not cat_cols:
        return _DeductionResult(0.0, "", [])
    max_g = -1.0
    affected: list[str] = []
    for col in cat_cols:
        try:
            vc = df[col].dropna().value_counts()
        except TypeError:
            vc = df[col].astype(str).dropna().value_counts()
        if vc.size < 2:
            continue
        g = _inequality_gini(vc.to_numpy(dtype=float))
        if g > 0.7:
            affected.append(col)
        if g > max_g:
            max_g = g
    if max_g <= 0.7:
        return _DeductionResult(0.0, "", [])
    scaled = (max_g - 0.7) / 0.3
    deduction = min(float(IMBALANCE_MAX), float(IMBALANCE_MAX) * scaled)
    reason = f"Max class-imbalance Gini {max_g:.3f} (threshold 0.70)."
    return _DeductionResult(deduction, reason, affected)


def _deduct_mixed_dtype(df: pd.DataFrame) -> _DeductionResult:
    """Compute the mixed-dtype deduction."""
    n_cols = df.shape[1]
    if n_cols == 0:
        return _DeductionResult(0.0, "", [])
    mixed = mixed_dtype_columns(df)
    deduction = min(float(MIXED_DTYPE_MAX), 50.0 * len(mixed) / n_cols)
    reason = f"{len(mixed)} mixed-dtype column(s)." if mixed else ""
    return _DeductionResult(deduction, reason, mixed)


def _deduct_cardinality(df: pd.DataFrame) -> _DeductionResult:
    """Compute the high-cardinality deduction."""
    cat_cols = _safe_categorical_cols(df)
    if not cat_cols:
        return _DeductionResult(0.0, "", [])
    high_card = [c for c in cat_cols if _is_high_cardinality(df[c])]
    deduction = min(float(CARDINALITY_MAX), 10.0 * len(high_card) / max(1, len(cat_cols)))
    reason = (
        f"{len(high_card)} high-cardinality categorical column(s)."
        if high_card
        else ""
    )
    return _DeductionResult(deduction, reason, high_card)


def _deduct_constant(df: pd.DataFrame) -> _DeductionResult:
    """Compute the constant-column deduction."""
    n_cols = df.shape[1]
    if n_cols == 0:
        return _DeductionResult(0.0, "", [])
    constants = constant_columns(df)
    deduction = min(float(CONSTANT_MAX), 50.0 * len(constants) / n_cols)
    reason = f"{len(constants)} constant column(s)." if constants else ""
    return _DeductionResult(deduction, reason, constants)


def _deduct_date_inconsistency(df: pd.DataFrame) -> _DeductionResult:
    """Compute the date-format-inconsistency deduction."""
    if df is None or df.empty:
        return _DeductionResult(0.0, "", [])
    suspected = 0
    inconsistent: list[str] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        n = len(non_null)
        if n == 0:
            continue
        try:
            parsed = pd.to_datetime(non_null, errors="coerce")
        except (ValueError, TypeError):
            continue
        parsed_ok = int(parsed.notna().sum())
        if parsed_ok / n <= 0.5:
            continue
        suspected += 1
        failed = int(parsed.isna().sum())
        if failed / n > 0.1:
            inconsistent.append(str(col))
    if suspected == 0:
        return _DeductionResult(0.0, "", [])
    deduction = min(
        float(DATE_INCONSISTENCY_MAX),
        10.0 * len(inconsistent) / max(1, suspected),
    )
    reason = (
        f"{len(inconsistent)} of {suspected} suspected date column(s) "
        f"have >10% unparseable values."
        if inconsistent
        else ""
    )
    return _DeductionResult(deduction, reason, inconsistent)


def compute_health_score(df: pd.DataFrame) -> HealthReport:
    """Compute the aggregate data health score.

    The score starts at 100 and is reduced by weighted, capped deductions
    for missingness, duplicates, outliers, class imbalance, mixed dtypes,
    high cardinality, constant columns, and date-format inconsistency. The
    final score is floored to an integer in ``[0, 100]``.

    This function never raises. Empty inputs and unexpected errors return
    safe ``HealthReport`` values.

    Args:
        df: Input DataFrame.

    Returns:
        A :class:`HealthReport` with score, zone, per-category deductions,
        reasons, and affected columns.
    """
    try:
        if df is None or df.empty or df.shape[1] == 0:
            return HealthReport(
                score=0,
                zone="Critical",
                deductions={},
                reasons={"empty": "No data"},
                affected_columns={},
            )

        category_funcs = {
            "missing": _deduct_missing,
            "duplicates": _deduct_duplicates,
            "outliers": _deduct_outliers,
            "imbalance": _deduct_imbalance,
            "mixed_dtype": _deduct_mixed_dtype,
            "cardinality": _deduct_cardinality,
            "constant": _deduct_constant,
            "date_inconsistency": _deduct_date_inconsistency,
        }

        deductions: dict[str, float] = {}
        reasons: dict[str, str] = {}
        affected_columns: dict[str, list[str]] = {}
        total_deduction = 0.0

        for name, func in category_funcs.items():
            try:
                res = func(df)
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                logger.warning("Deduction '%s' failed: %s", name, exc)
                continue
            value = float(max(0.0, res.value))
            deductions[name] = round(value, 2)
            if res.reason:
                reasons[name] = res.reason
            if res.affected:
                affected_columns[name] = res.affected
            total_deduction += value

        raw_score = 100.0 - total_deduction
        score = int(max(0, min(100, np.floor(raw_score))))
        zone = str(health_zone(score))

        return HealthReport(
            score=score,
            zone=zone,
            deductions=deductions,
            reasons=reasons,
            affected_columns=affected_columns,
        )
    except Exception as exc:  # noqa: BLE001 - public API must never raise
        logger.exception("compute_health_score failed unexpectedly")
        return HealthReport(
            score=0,
            zone="Critical",
            deductions={},
            reasons={"error": str(exc)},
            affected_columns={},
        )
