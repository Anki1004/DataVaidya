"""Tests for core.cleaning."""
from __future__ import annotations
import pandas as pd
import pytest
from core.cleaning import (
    fill_missing, drop_duplicates, cap_outliers_iqr,
    strip_whitespace, downcast_numeric, apply_op, OPS,
    take_snapshot, undo, CleaningError,
)


def test_fill_missing_median(dirty_df):
    out, log = fill_missing(dirty_df, mode="preview", columns=["age"], strategy="median")
    assert out["age"].isna().sum() == 0
    assert log["op"] == "fill_missing"


def test_drop_duplicates(dirty_df):
    before = len(dirty_df)
    out, _ = drop_duplicates(dirty_df, mode="preview", keep="first")
    assert len(out) < before


def test_cap_outliers_clips_extremes(dirty_df):
    out, _ = cap_outliers_iqr(dirty_df, mode="preview", columns=["income"])
    assert out["income"].max() < 9_999_999


def test_preview_does_not_mutate_state():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    state: dict = {"undo_stack": []}
    out, _ = fill_missing(df, mode="preview", strategy="median")
    # state untouched
    assert state["undo_stack"] == []


def test_apply_op_dispatch(clean_df):
    out, log = apply_op(clean_df, "downcast_numeric", mode="preview")
    assert log["op"] == "downcast_numeric"


def test_apply_op_unknown_raises(clean_df):
    with pytest.raises(CleaningError):
        apply_op(clean_df, "nonexistent_op", mode="preview")


def test_snapshot_and_undo():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [4, 5, 6]})
    state: dict = {"undo_stack": []}
    take_snapshot(state, df1, "first")
    take_snapshot(state, df2, "second")
    restored = undo(state)
    assert restored.equals(df2)


def test_undo_on_empty_stack_returns_none():
    state: dict = {"undo_stack": []}
    assert undo(state) is None


def test_ops_registry_completeness():
    expected = {"fill_missing", "drop_missing", "drop_duplicates",
                "cap_outliers_iqr", "remove_outliers_iqr", "strip_whitespace",
                "standardize_case", "rename_columns", "drop_columns",
                "parse_dates", "cast_dtype", "downcast_numeric",
                "clip_range", "normalize_minmax", "standardize_zscore",
                "filter_rows", "reset_index_op"}
    assert expected <= set(OPS.keys())
