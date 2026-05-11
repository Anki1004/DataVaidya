"""Tests for core.profiling."""
from __future__ import annotations
import pandas as pd
from core.profiling import (
    compute_health_score, missing_summary, duplicate_summary,
    outlier_summary, correlation_matrix, top_correlations,
    cardinality_summary, constant_columns, mixed_dtype_columns,
)


class TestHealthScore:
    def test_empty_df_returns_critical(self, empty_df):
        r = compute_health_score(empty_df)
        assert r["score"] == 0
        assert r["zone"] == "Critical"

    def test_clean_df_scores_high(self, clean_df):
        r = compute_health_score(clean_df)
        assert r["score"] >= 70

    def test_dirty_df_has_deductions(self, dirty_df):
        r = compute_health_score(dirty_df)
        assert r["score"] < 100
        assert len(r["deductions"]) > 0

    def test_single_row_does_not_raise(self, single_row_df):
        r = compute_health_score(single_row_df)
        assert 0 <= r["score"] <= 100

    def test_never_raises_on_weird_data(self):
        df = pd.DataFrame({"a": [None] * 50, "b": [1] * 50})
        r = compute_health_score(df)
        assert 0 <= r["score"] <= 100


class TestSummaries:
    def test_missing_summary(self, dirty_df):
        s = missing_summary(dirty_df)
        assert "column" in s.columns
        assert (s["missing_count"] >= 0).all()

    def test_duplicate_summary(self, dirty_df):
        d = duplicate_summary(dirty_df)
        assert d["total"] >= 1

    def test_correlation_skips_when_too_few_numeric(self, single_row_df):
        c = correlation_matrix(single_row_df)
        # Returns empty df instead of raising
        assert isinstance(c, pd.DataFrame)

    def test_constant_columns_detection(self, dirty_df):
        cols = constant_columns(dirty_df)
        assert "constant_col" in cols

    def test_mixed_dtype_detection(self, dirty_df):
        cols = mixed_dtype_columns(dirty_df)
        assert "mixed" in cols
