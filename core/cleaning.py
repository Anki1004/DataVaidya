"""Data cleaning operations for DataVaidya.

This module exposes a registry of pure-ish cleaning functions used by the UI
dispatcher and the code-generation exporter. Each function returns a new
DataFrame and a structured change log describing what happened, so the caller
can render diffs, preview impact, and reconstruct a deterministic pipeline.

All public cleaning functions share the signature::

    func(df, mode, **kwargs) -> tuple[pd.DataFrame, dict]

where ``mode`` is either ``"preview"`` or ``"apply"``. Preview mode is
guaranteed never to mutate caller state; apply mode is identical except that
callers typically pair it with :func:`take_snapshot` to enable undo.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from utils.constants import IQR_MULTIPLIER

logger = logging.getLogger(__name__)

Mode = Literal["preview", "apply"]

MAX_UNDO_DEPTH: int = 10


class CleaningError(ValueError):
    """Raised when a cleaning operation receives invalid input or cannot proceed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _ChangeLogBuilder:
    """Lightweight accumulator for change-log dicts.

    Using a dataclass avoids repetitive boilerplate and makes it harder to
    forget a required field in the returned log.
    """

    op: str
    params: dict[str, Any]
    rows_before: int
    rows_after: int = 0
    cells_changed: int = 0
    affected_columns: list[str] = None  # type: ignore[assignment]
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.affected_columns is None:
            self.affected_columns = []
        if self.warnings is None:
            self.warnings = []

    def warn(self, msg: str) -> None:
        """Record a non-fatal warning and forward it to the module logger."""
        logger.warning("%s: %s", self.op, msg)
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        """Materialize the builder as a plain dict for the public API."""
        return {
            "op": self.op,
            "params": self.params,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cells_changed": self.cells_changed,
            "affected_columns": list(self.affected_columns),
            "warnings": list(self.warnings),
        }


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``KeyError`` with a friendly message if any column is missing.

    Args:
        df: DataFrame whose columns are checked.
        columns: Column names that must exist on ``df``.

    Raises:
        KeyError: If at least one of ``columns`` is not present.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _empty_df_log(op: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build the change-log returned when the input DataFrame is empty."""
    builder = _ChangeLogBuilder(op=op, params=params, rows_before=0, rows_after=0)
    builder.warn("DataFrame is empty; no-op.")
    return builder.to_dict()


def _count_cells_changed(before: pd.Series, after: pd.Series) -> int:
    """Return the number of cell-level differences between two aligned series.

    NaNs are treated as equal to NaNs so that filling NaN with NaN does not
    count as a change. Length mismatches return ``0`` (the caller is expected
    to have measured those changes via row counts).
    """
    if len(before) != len(after):
        return 0
    a = before.to_numpy(copy=False)
    b = after.to_numpy(copy=False)
    # Equality where both NaN counts as equal.
    both_nan = pd.isna(a) & pd.isna(b)
    try:
        eq = (a == b) | both_nan
    except (TypeError, ValueError):
        # Mixed dtypes can fail elementwise compare; fall back to objectwise.
        eq = np.array([x == y or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)])
    return int((~eq).sum())


def _resolve_columns(
    df: pd.DataFrame, columns: list[str] | None, *, default_all: bool = True
) -> list[str]:
    """Normalize a ``columns`` argument that may be ``None`` or empty.

    Args:
        df: DataFrame whose columns are the universe of valid names.
        columns: Caller-supplied column selection.
        default_all: If ``True``, treat ``None``/empty as "all columns".

    Returns:
        A concrete list of column names; validated to exist on ``df``.
    """
    if not columns:
        return list(df.columns) if default_all else []
    _validate_columns(df, columns)
    return list(columns)


def _df_fingerprint(df: pd.DataFrame) -> tuple[Any, ...]:
    """Cheap structural fingerprint for snapshot deduplication.

    The fingerprint is intentionally small (shape + columns + first row CSV)
    so it can be computed without materializing the whole frame. It is not a
    cryptographic hash and is only used to short-circuit obvious no-op
    snapshots.
    """
    try:
        head_csv = df.head(1).to_csv(index=False)
    except (ValueError, TypeError):
        head_csv = ""
    return (df.shape, tuple(map(str, df.columns)), head_csv)


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------


def fill_missing(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str] | None = None,
    strategy: Literal["mean", "median", "mode", "constant", "ffill", "bfill"] = "median",
    value: Any = None,
) -> tuple[pd.DataFrame, dict]:
    """Fill ``NaN`` values using one of several strategies.

    Args:
        df: Input DataFrame.
        mode: ``"preview"`` or ``"apply"`` (semantics are identical here; the
            argument is part of the uniform op signature).
        columns: Columns to operate on. ``None`` or empty means all columns.
        strategy: Fill strategy. ``"mean"`` and ``"median"`` are silently
            skipped for non-numeric columns. ``"mode"`` uses ``.iloc[0]`` to
            disambiguate multimodal distributions.
        value: Required when ``strategy == "constant"``; ignored otherwise.

    Returns:
        ``(new_df, change_log)``.

    Raises:
        CleaningError: If ``strategy='constant'`` and ``value`` is ``None``,
            or if an unknown strategy is provided.
        KeyError: If a requested column is missing.
    """
    params = {"columns": columns, "strategy": strategy, "value": value}
    if df.empty:
        return df.copy(), _empty_df_log("fill_missing", params)

    cols = _resolve_columns(df, columns, default_all=True)
    out = df.copy()
    builder = _ChangeLogBuilder(
        op="fill_missing", params=params, rows_before=len(df), rows_after=len(df)
    )

    if strategy == "constant" and value is None:
        raise CleaningError("fill_missing: strategy='constant' requires a non-None value.")

    valid_strategies = {"mean", "median", "mode", "constant", "ffill", "bfill"}
    if strategy not in valid_strategies:
        raise CleaningError(
            f"fill_missing: unknown strategy {strategy!r}; expected one of {sorted(valid_strategies)}."
        )

    total_changed = 0
    affected: list[str] = []
    for col in cols:
        series = out[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if strategy in {"mean", "median"} and not is_numeric:
            builder.warn(f"Skipping non-numeric column {col!r} for strategy {strategy!r}.")
            continue

        if strategy == "mean":
            fill_val: Any = series.mean()
        elif strategy == "median":
            fill_val = series.median()
        elif strategy == "mode":
            non_na = series.dropna()
            if non_na.empty:
                builder.warn(f"Column {col!r} is entirely NaN; cannot compute mode.")
                continue
            fill_val = non_na.mode(dropna=True)
            if fill_val.empty:
                builder.warn(f"Column {col!r} has no mode; skipping.")
                continue
            fill_val = fill_val.iloc[0]
        elif strategy == "constant":
            fill_val = value
        else:
            fill_val = None  # ffill / bfill handled below

        before = series.copy()
        if strategy == "ffill":
            new_series = series.ffill()
        elif strategy == "bfill":
            new_series = series.bfill()
        else:
            new_series = series.fillna(fill_val)

        changed = _count_cells_changed(before, new_series)
        if changed:
            out[col] = new_series
            total_changed += changed
            affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def drop_missing(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str] | None = None,
    how: Literal["any", "all"] = "any",
    thresh: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop rows containing missing values.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag (see module docstring).
        columns: Subset of columns to inspect. ``None`` means all columns.
        how: ``"any"`` to drop rows with any NaN, ``"all"`` only fully-NaN rows.
        thresh: If set, keep rows with at least this many non-NaN values
            (takes precedence over ``how`` per pandas semantics).

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns, "how": how, "thresh": thresh}
    if df.empty:
        return df.copy(), _empty_df_log("drop_missing", params)

    subset = _resolve_columns(df, columns, default_all=True)
    rows_before = len(df)
    if thresh is not None:
        new_df = df.dropna(subset=subset, thresh=thresh).copy()
    else:
        new_df = df.dropna(subset=subset, how=how).copy()

    builder = _ChangeLogBuilder(
        op="drop_missing",
        params=params,
        rows_before=rows_before,
        rows_after=len(new_df),
        affected_columns=list(subset),
    )
    return new_df, builder.to_dict()


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def drop_duplicates(
    df: pd.DataFrame,
    mode: Mode,
    subset: list[str] | None = None,
    keep: Literal["first", "last", "none"] = "first",
) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate rows.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        subset: Columns to consider for duplicate detection. ``None`` uses all.
        keep: ``"first"``, ``"last"``, or ``"none"``. ``"none"`` drops every
            row that participates in a duplicate group.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"subset": subset, "keep": keep}
    if df.empty:
        return df.copy(), _empty_df_log("drop_duplicates", params)

    if subset:
        _validate_columns(df, subset)

    pandas_keep: Any = False if keep == "none" else keep
    new_df = df.drop_duplicates(subset=subset, keep=pandas_keep).copy()

    builder = _ChangeLogBuilder(
        op="drop_duplicates",
        params=params,
        rows_before=len(df),
        rows_after=len(new_df),
        affected_columns=list(subset) if subset else list(df.columns),
    )
    return new_df, builder.to_dict()


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------


def _iqr_bounds(series: pd.Series, multiplier: float) -> tuple[float, float]:
    """Compute IQR-based lower/upper bounds for a numeric series."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def cap_outliers_iqr(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, dict]:
    """Clip values to ``[Q1 - k*IQR, Q3 + k*IQR]`` per column.

    Non-numeric columns are skipped with a warning recorded in the change log.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to inspect.
        multiplier: ``k`` in the IQR formula. Defaults to ``IQR_MULTIPLIER``.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns, "multiplier": multiplier}
    if df.empty:
        return df.copy(), _empty_df_log("cap_outliers_iqr", params)

    _validate_columns(df, columns)
    out = df.copy()
    builder = _ChangeLogBuilder(
        op="cap_outliers_iqr",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            builder.warn(f"Skipping non-numeric column {col!r}.")
            continue
        lower, upper = _iqr_bounds(out[col], multiplier)
        before = out[col].copy()
        clipped = out[col].clip(lower=lower, upper=upper)
        changed = _count_cells_changed(before, clipped)
        if changed:
            out[col] = clipped
            total_changed += changed
            affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def remove_outliers_iqr(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, dict]:
    """Drop rows whose value in any given numeric column falls outside the IQR fence.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to inspect (non-numeric skipped with warning).
        multiplier: IQR multiplier.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns, "multiplier": multiplier}
    if df.empty:
        return df.copy(), _empty_df_log("remove_outliers_iqr", params)

    _validate_columns(df, columns)
    rows_before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    builder = _ChangeLogBuilder(
        op="remove_outliers_iqr",
        params=params,
        rows_before=rows_before,
    )
    affected: list[str] = []

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            builder.warn(f"Skipping non-numeric column {col!r}.")
            continue
        lower, upper = _iqr_bounds(df[col], multiplier)
        col_mask = df[col].between(lower, upper) | df[col].isna()
        keep_mask &= col_mask
        affected.append(col)

    new_df = df.loc[keep_mask].copy()
    builder.rows_after = len(new_df)
    builder.affected_columns = affected
    return new_df, builder.to_dict()


# ---------------------------------------------------------------------------
# String / text cleaning
# ---------------------------------------------------------------------------


def _object_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of all object/string-dtype columns."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def strip_whitespace(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Strip leading/trailing whitespace from string-like columns.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Subset of columns; ``None`` means all object/string columns.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns}
    if df.empty:
        return df.copy(), _empty_df_log("strip_whitespace", params)

    if columns:
        _validate_columns(df, columns)
        target_cols = list(columns)
    else:
        target_cols = _object_columns(df)

    out = df.copy()
    builder = _ChangeLogBuilder(
        op="strip_whitespace",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in target_cols:
        if not (
            pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])
        ):
            builder.warn(f"Skipping non-string column {col!r}.")
            continue
        before = out[col].copy()
        stripped = out[col].astype("object").where(out[col].isna(), out[col].astype(str).str.strip())
        changed = _count_cells_changed(before, stripped)
        if changed:
            out[col] = stripped
            total_changed += changed
            affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def standardize_case(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    case: Literal["lower", "upper", "title"] = "lower",
) -> tuple[pd.DataFrame, dict]:
    """Normalize string casing on the requested columns.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to operate on.
        case: ``"lower"``, ``"upper"``, or ``"title"``.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns, "case": case}
    if df.empty:
        return df.copy(), _empty_df_log("standardize_case", params)

    _validate_columns(df, columns)
    if case not in {"lower", "upper", "title"}:
        raise CleaningError(f"standardize_case: unknown case {case!r}.")

    out = df.copy()
    builder = _ChangeLogBuilder(
        op="standardize_case",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if not (
            pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])
        ):
            builder.warn(f"Skipping non-string column {col!r}.")
            continue
        before = out[col].copy()
        s = out[col].astype("object")
        as_str = s.astype(str)
        if case == "lower":
            transformed = as_str.str.lower()
        elif case == "upper":
            transformed = as_str.str.upper()
        else:
            transformed = as_str.str.title()
        new_series = s.where(s.isna(), transformed)
        changed = _count_cells_changed(before, new_series)
        if changed:
            out[col] = new_series
            total_changed += changed
            affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


# ---------------------------------------------------------------------------
# Schema operations
# ---------------------------------------------------------------------------


def rename_columns(
    df: pd.DataFrame,
    mode: Mode,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """Rename columns according to ``mapping``.

    Mapping values containing newline or backtick characters are rejected
    outright as a defense-in-depth measure for the downstream code generator
    (it embeds identifiers in generated Python source).

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        mapping: ``{old_name: new_name}``. Keys must exist in ``df``.

    Returns:
        ``(new_df, change_log)``.

    Raises:
        CleaningError: For unsafe characters in target names or duplicate
            destinations.
        KeyError: For unknown source names.
    """
    params = {"mapping": dict(mapping)}
    if df.empty:
        return df.copy(), _empty_df_log("rename_columns", params)

    if not isinstance(mapping, dict):
        raise CleaningError("rename_columns: mapping must be a dict.")

    _validate_columns(df, list(mapping.keys()))

    for old, new in mapping.items():
        if not isinstance(new, str):
            raise CleaningError(
                f"rename_columns: target name for {old!r} must be a string, got {type(new).__name__}."
            )
        if "\n" in new or "`" in new or "\r" in new:
            raise CleaningError(
                f"rename_columns: target name {new!r} contains forbidden characters (newline/backtick)."
            )

    new_columns = [mapping.get(c, c) for c in df.columns]
    if len(set(new_columns)) != len(new_columns):
        raise CleaningError(
            "rename_columns: rename would produce duplicate column names."
        )

    out = df.copy()
    out.columns = new_columns

    builder = _ChangeLogBuilder(
        op="rename_columns",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
        affected_columns=list(mapping.keys()),
    )
    return out, builder.to_dict()


def drop_columns(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Drop the named columns from the DataFrame.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to drop. Must all exist.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns}
    if df.empty:
        return df.copy(), _empty_df_log("drop_columns", params)

    _validate_columns(df, columns)
    out = df.drop(columns=list(columns)).copy()
    builder = _ChangeLogBuilder(
        op="drop_columns",
        params=params,
        rows_before=len(df),
        rows_after=len(out),
        affected_columns=list(columns),
    )
    return out, builder.to_dict()


# ---------------------------------------------------------------------------
# Type conversions
# ---------------------------------------------------------------------------


def parse_dates(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    format: str | Literal["infer"] = "infer",
    errors: Literal["coerce", "raise"] = "coerce",
) -> tuple[pd.DataFrame, dict]:
    """Parse columns into ``datetime64[ns]``.

    Columns that are already datetime dtype are left untouched (no-op) so
    that re-running the pipeline is idempotent.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to parse.
        format: ``"infer"`` to let pandas guess, or an explicit ``strftime``
            pattern.
        errors: ``"coerce"`` (default) turns unparseable values into NaT;
            ``"raise"`` propagates the underlying ``ValueError``.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns, "format": format, "errors": errors}
    if df.empty:
        return df.copy(), _empty_df_log("parse_dates", params)

    _validate_columns(df, columns)
    out = df.copy()
    builder = _ChangeLogBuilder(
        op="parse_dates",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            builder.warn(f"Column {col!r} is already datetime; leaving as-is.")
            continue
        before = out[col].copy()
        kwargs: dict[str, Any] = {"errors": errors}
        if format != "infer":
            kwargs["format"] = format
        try:
            parsed = pd.to_datetime(out[col], **kwargs)
        except (ValueError, TypeError) as exc:
            if errors == "raise":
                raise CleaningError(
                    f"parse_dates: failed to parse column {col!r}: {exc}"
                ) from exc
            builder.warn(f"Column {col!r} parse error coerced: {exc}")
            continue
        changed = _count_cells_changed(before, parsed)
        out[col] = parsed
        if changed:
            total_changed += changed
        affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def cast_dtype(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    dtype: Literal["int", "float", "str", "category", "bool"],
) -> tuple[pd.DataFrame, dict]:
    """Cast columns to one of a handful of friendly dtype aliases.

    ``"int"`` rejects columns with NaN values, since pandas would silently
    promote them back to float. Callers are nudged toward nullable ``Int64``
    in the error message.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to cast.
        dtype: Target dtype alias.

    Returns:
        ``(new_df, change_log)``.

    Raises:
        CleaningError: If ``dtype='int'`` is requested on a column containing
            NaNs, or if ``dtype`` is unknown.
    """
    params = {"columns": columns, "dtype": dtype}
    if df.empty:
        return df.copy(), _empty_df_log("cast_dtype", params)

    _validate_columns(df, columns)
    dtype_map = {
        "int": "int64",
        "float": "float64",
        "str": "string",
        "category": "category",
        "bool": "bool",
    }
    if dtype not in dtype_map:
        raise CleaningError(
            f"cast_dtype: unknown dtype {dtype!r}; expected one of {sorted(dtype_map)}."
        )

    out = df.copy()
    builder = _ChangeLogBuilder(
        op="cast_dtype",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    affected: list[str] = []
    for col in columns:
        if dtype == "int" and out[col].isna().any():
            raise CleaningError(
                f"cast_dtype: column {col!r} contains NaN values; cannot cast to int. "
                "Fill missing values first, or cast to nullable 'Int64' explicitly."
            )
        try:
            out[col] = out[col].astype(dtype_map[dtype])
        except (ValueError, TypeError) as exc:
            raise CleaningError(
                f"cast_dtype: could not cast column {col!r} to {dtype!r}: {exc}"
            ) from exc
        affected.append(col)

    builder.affected_columns = affected
    return out, builder.to_dict()


def downcast_numeric(
    df: pd.DataFrame,
    mode: Mode,
) -> tuple[pd.DataFrame, dict]:
    """Reduce memory footprint by safely downcasting numeric and object columns.

    Rules:
        * ``int64`` -> ``int32`` if all values fit in the int32 range.
        * ``float64`` -> ``float32`` if max absolute round-trip error is below
          ``1e-6``.
        * ``object`` -> ``category`` if the unique ratio is below 0.5.

    Columns containing NaN are never downcast on the integer path because the
    promotion back to float would silently negate the optimization.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.

    Returns:
        ``(new_df, change_log)``.
    """
    params: dict[str, Any] = {}
    if df.empty:
        return df.copy(), _empty_df_log("downcast_numeric", params)

    out = df.copy()
    builder = _ChangeLogBuilder(
        op="downcast_numeric",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )
    affected: list[str] = []

    int32_info = np.iinfo(np.int32)
    for col in out.columns:
        series = out[col]
        has_nan = series.isna().any()

        if pd.api.types.is_integer_dtype(series) and not has_nan:
            if series.dtype == np.int64:
                col_min = int(series.min())
                col_max = int(series.max())
                if int32_info.min <= col_min and col_max <= int32_info.max:
                    out[col] = series.astype(np.int32)
                    affected.append(col)
        elif pd.api.types.is_float_dtype(series):
            if series.dtype == np.float64:
                candidate = series.astype(np.float32)
                # Compare ignoring NaNs (NaN != NaN under direct compare).
                diff = (candidate.astype(np.float64) - series).abs()
                max_err = float(diff.max(skipna=True)) if len(diff) else 0.0
                if not np.isnan(max_err) and max_err < 1e-6:
                    out[col] = candidate
                    affected.append(col)
        elif pd.api.types.is_object_dtype(series):
            n = len(series)
            if n > 0 and series.nunique(dropna=True) / n < 0.5:
                out[col] = series.astype("category")
                affected.append(col)

    builder.affected_columns = affected
    return out, builder.to_dict()


# ---------------------------------------------------------------------------
# Numeric transforms
# ---------------------------------------------------------------------------


def clip_range(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Clip the given numeric columns to an explicit range.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to clip.
        lower: Lower bound (``None`` to leave the lower side unbounded).
        upper: Upper bound (``None`` to leave the upper side unbounded).

    Returns:
        ``(new_df, change_log)``.

    Raises:
        CleaningError: If both bounds are ``None`` or if ``lower > upper``.
    """
    params = {"columns": columns, "lower": lower, "upper": upper}
    if df.empty:
        return df.copy(), _empty_df_log("clip_range", params)

    _validate_columns(df, columns)
    if lower is None and upper is None:
        raise CleaningError("clip_range: at least one of lower/upper must be provided.")
    if lower is not None and upper is not None and lower > upper:
        raise CleaningError(
            f"clip_range: lower ({lower}) must not exceed upper ({upper})."
        )

    out = df.copy()
    builder = _ChangeLogBuilder(
        op="clip_range",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            builder.warn(f"Skipping non-numeric column {col!r}.")
            continue
        before = out[col].copy()
        clipped = out[col].clip(lower=lower, upper=upper)
        changed = _count_cells_changed(before, clipped)
        if changed:
            out[col] = clipped
            total_changed += changed
            affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def normalize_minmax(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply min-max normalization to map each column into ``[0, 1]``.

    Constant columns are replaced with ``0.0`` (rather than ``NaN`` from
    division by zero) and a warning is recorded.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to normalize.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns}
    if df.empty:
        return df.copy(), _empty_df_log("normalize_minmax", params)

    _validate_columns(df, columns)
    out = df.copy()
    builder = _ChangeLogBuilder(
        op="normalize_minmax",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            builder.warn(f"Skipping non-numeric column {col!r}.")
            continue
        series = out[col].astype(float)
        col_min = float(series.min())
        col_max = float(series.max())
        if np.isnan(col_min) or np.isnan(col_max):
            builder.warn(f"Column {col!r} is entirely NaN; skipping.")
            continue
        rng = col_max - col_min
        before = out[col].copy()
        if rng == 0:
            new_series = pd.Series(0.0, index=series.index)
            builder.warn(
                f"Column {col!r} is constant; min-max normalization produced 0.0 for all rows."
            )
        else:
            new_series = (series - col_min) / rng
        out[col] = new_series
        changed = _count_cells_changed(before, new_series)
        if changed:
            total_changed += changed
        affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


def standardize_zscore(
    df: pd.DataFrame,
    mode: Mode,
    columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply z-score standardization (``(x - mean) / std``) per column.

    Constant columns (``std == 0``) are replaced with ``0.0`` and a warning
    is recorded.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        columns: Columns to standardize.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"columns": columns}
    if df.empty:
        return df.copy(), _empty_df_log("standardize_zscore", params)

    _validate_columns(df, columns)
    out = df.copy()
    builder = _ChangeLogBuilder(
        op="standardize_zscore",
        params=params,
        rows_before=len(df),
        rows_after=len(df),
    )

    total_changed = 0
    affected: list[str] = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            builder.warn(f"Skipping non-numeric column {col!r}.")
            continue
        series = out[col].astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        before = out[col].copy()
        if std == 0 or np.isnan(std):
            new_series = pd.Series(0.0, index=series.index)
            builder.warn(
                f"Column {col!r} has zero variance; z-score produced 0.0 for all rows."
            )
        else:
            new_series = (series - mean) / std
        out[col] = new_series
        changed = _count_cells_changed(before, new_series)
        if changed:
            total_changed += changed
        affected.append(col)

    builder.cells_changed = total_changed
    builder.affected_columns = affected
    return out, builder.to_dict()


# ---------------------------------------------------------------------------
# Row filtering / index
# ---------------------------------------------------------------------------


def filter_rows(
    df: pd.DataFrame,
    mode: Mode,
    column: str,
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "notin", "contains"],
    value: Any,
) -> tuple[pd.DataFrame, dict]:
    """Filter rows by a single column predicate.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        column: Column to evaluate.
        op: Comparison operator.
        value: Right-hand side. For ``"in"``/``"notin"`` must be a list/tuple/set.

    Returns:
        ``(new_df, change_log)``.

    Raises:
        CleaningError: For unknown operators or type mismatches.
    """
    params = {"column": column, "op": op, "value": value}
    if df.empty:
        return df.copy(), _empty_df_log("filter_rows", params)

    _validate_columns(df, [column])
    valid_ops = {"==", "!=", ">", ">=", "<", "<=", "in", "notin", "contains"}
    if op not in valid_ops:
        raise CleaningError(f"filter_rows: unknown op {op!r}; expected one of {sorted(valid_ops)}.")

    if op in {"in", "notin"} and not isinstance(value, (list, tuple, set)):
        raise CleaningError(
            f"filter_rows: op={op!r} requires value to be a list, tuple, or set; got {type(value).__name__}."
        )

    series = df[column]
    try:
        if op == "==":
            mask = series == value
        elif op == "!=":
            mask = series != value
        elif op == ">":
            mask = series > value
        elif op == ">=":
            mask = series >= value
        elif op == "<":
            mask = series < value
        elif op == "<=":
            mask = series <= value
        elif op == "in":
            mask = series.isin(list(value))
        elif op == "notin":
            mask = ~series.isin(list(value))
        else:  # contains
            if not isinstance(value, str):
                raise CleaningError(
                    f"filter_rows: op='contains' requires a string value; got {type(value).__name__}."
                )
            mask = series.astype(str).str.contains(value, na=False, regex=False)
    except (TypeError, ValueError) as exc:
        raise CleaningError(
            f"filter_rows: failed to evaluate {column!r} {op} {value!r}: {exc}"
        ) from exc

    new_df = df.loc[mask.fillna(False)].copy()
    builder = _ChangeLogBuilder(
        op="filter_rows",
        params=params,
        rows_before=len(df),
        rows_after=len(new_df),
        affected_columns=[column],
    )
    return new_df, builder.to_dict()


def reset_index_op(
    df: pd.DataFrame,
    mode: Mode,
    drop: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Reset the DataFrame index.

    Args:
        df: Input DataFrame.
        mode: Uniform op mode flag.
        drop: If ``True``, discard the old index; otherwise materialize it as
            a column.

    Returns:
        ``(new_df, change_log)``.
    """
    params = {"drop": drop}
    if df.empty:
        return df.copy(), _empty_df_log("reset_index_op", params)

    new_df = df.reset_index(drop=drop).copy()
    builder = _ChangeLogBuilder(
        op="reset_index_op",
        params=params,
        rows_before=len(df),
        rows_after=len(new_df),
    )
    return new_df, builder.to_dict()


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------


OPS: dict[str, Callable[..., tuple[pd.DataFrame, dict]]] = {
    "fill_missing": fill_missing,
    "drop_missing": drop_missing,
    "drop_duplicates": drop_duplicates,
    "cap_outliers_iqr": cap_outliers_iqr,
    "remove_outliers_iqr": remove_outliers_iqr,
    "strip_whitespace": strip_whitespace,
    "standardize_case": standardize_case,
    "rename_columns": rename_columns,
    "drop_columns": drop_columns,
    "parse_dates": parse_dates,
    "cast_dtype": cast_dtype,
    "downcast_numeric": downcast_numeric,
    "clip_range": clip_range,
    "normalize_minmax": normalize_minmax,
    "standardize_zscore": standardize_zscore,
    "filter_rows": filter_rows,
    "reset_index_op": reset_index_op,
}


def apply_op(
    df: pd.DataFrame,
    op_name: str,
    mode: Mode = "preview",
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict]:
    """Dispatch a registered cleaning op by name.

    Args:
        df: Input DataFrame.
        op_name: Key in :data:`OPS`.
        mode: ``"preview"`` or ``"apply"``.
        **kwargs: Forwarded to the underlying op.

    Returns:
        ``(new_df, change_log)`` from the dispatched op.

    Raises:
        CleaningError: If ``op_name`` is not registered.
    """
    if op_name not in OPS:
        raise CleaningError(
            f"apply_op: unknown op {op_name!r}; available: {sorted(OPS)}."
        )
    return OPS[op_name](df, mode, **kwargs)


# ---------------------------------------------------------------------------
# Session-state helpers (Streamlit-agnostic)
# ---------------------------------------------------------------------------


def take_snapshot(state: dict, df: pd.DataFrame, label: str) -> None:
    """Push a deep-copied snapshot of ``df`` onto the state's undo stack.

    The stack is trimmed from the bottom when it exceeds :data:`MAX_UNDO_DEPTH`.
    Snapshots are deduplicated against the top of the stack using a cheap
    structural fingerprint, so repeatedly snapshotting an unchanged frame is
    free.

    Args:
        state: Mutable mapping (typically ``st.session_state``) supplied by
            the caller. The function manipulates the ``"undo_stack"`` key.
        df: DataFrame to snapshot. Deep-copied before storage to ensure
            future mutations cannot leak in.
        label: Human-readable label, e.g. the op name that produced ``df``.
    """
    stack = state.setdefault("undo_stack", [])
    if stack:
        _, top_df, _ = stack[-1]
        if _df_fingerprint(top_df) == _df_fingerprint(df):
            logger.debug("take_snapshot: fingerprint matches top of stack; skipping.")
            return

    entry = (label, copy.deepcopy(df), datetime.utcnow())
    stack.append(entry)
    if len(stack) > MAX_UNDO_DEPTH:
        # Drop the oldest snapshots so the stack stays bounded.
        del stack[: len(stack) - MAX_UNDO_DEPTH]


def undo(state: dict) -> pd.DataFrame | None:
    """Pop the most recent snapshot.

    Args:
        state: Mutable mapping holding the ``"undo_stack"`` key.

    Returns:
        The popped DataFrame, or ``None`` if the stack is empty.
    """
    stack = state.get("undo_stack")
    if not stack:
        return None
    _, df, _ = stack.pop()
    return df


def get_undo_depth(state: dict) -> int:
    """Return the number of available undo snapshots.

    Args:
        state: Mutable mapping holding the ``"undo_stack"`` key.

    Returns:
        ``len(state['undo_stack'])`` or ``0`` if the key is absent.
    """
    return len(state.get("undo_stack", []))
