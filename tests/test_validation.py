"""Tests for utils.validation."""
from __future__ import annotations
import pytest
from utils.validation import (
    validate_extension, validate_file_size, validate_dataframe,
    FileTooLargeError, UnsupportedFileTypeError, EmptyDataFrameError,
)


def test_validate_extension_lowercases():
    assert validate_extension("data.CSV") == "csv"


def test_validate_extension_unsupported():
    with pytest.raises(UnsupportedFileTypeError):
        validate_extension("malware.exe")


def test_validate_extension_no_extension():
    with pytest.raises(UnsupportedFileTypeError):
        validate_extension("README")


def test_validate_file_size_too_large():
    with pytest.raises(FileTooLargeError):
        validate_file_size(100 * 1024 * 1024, "big.csv")  # 100MB > 50MB cap


def test_validate_file_size_warning():
    ok, warning = validate_file_size(35 * 1024 * 1024, "medium.csv")
    assert ok is True
    assert warning is not None


def test_validate_dataframe_empty_raises(empty_df):
    with pytest.raises(EmptyDataFrameError):
        validate_dataframe(empty_df)
