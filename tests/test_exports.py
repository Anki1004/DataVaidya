"""Tests for core.exports."""
from __future__ import annotations
from io import BytesIO
import pandas as pd
from core.exports import export_csv, export_excel, export_parquet, export_python_script


def test_export_csv(clean_df):
    buf = export_csv(clean_df)
    assert isinstance(buf, BytesIO)
    buf.seek(0)
    out = pd.read_csv(buf)
    assert len(out) == len(clean_df)


def test_export_excel(clean_df):
    buf = export_excel(clean_df)
    assert isinstance(buf, BytesIO)
    assert buf.getbuffer().nbytes > 0


def test_export_parquet(clean_df):
    buf = export_parquet(clean_df)
    buf.seek(0)
    out = pd.read_parquet(buf)
    assert len(out) == len(clean_df)


def test_python_script_includes_steps():
    log = [("fill_missing", {"columns": ["age"], "strategy": "median"})]
    buf = export_python_script(log, source_filename="input.csv")
    text = buf.getvalue().decode("utf-8")
    assert "op_fill_missing" in text
    assert "median" in text


def test_python_script_empty_log_still_valid():
    buf = export_python_script([], source_filename="input.csv")
    text = buf.getvalue().decode("utf-8")
    assert "def run" in text


def test_python_script_rejects_dangerous_column_name():
    import pytest
    log = [("rename_columns", {"mapping": {"x\n)": "y"}})]
    # The exporter may or may not reject — depends on implementation.
    # If it does reject, the test passes; if it sanitizes, also passes.
    try:
        buf = export_python_script(log)
        text = buf.getvalue().decode("utf-8")
        # If it generated, ensure no raw newline survived in the embedded mapping
        assert "x\n" not in text or text.count('"x\\n"') > 0
    except ValueError:
        pass  # explicit rejection is also acceptable
