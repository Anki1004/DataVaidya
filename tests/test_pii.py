"""Tests for core.pii."""
from __future__ import annotations
import pandas as pd
from core.pii import detect_pii, mask_pii, scrub_for_llm, PII_PATTERNS


def test_pan_pattern_positive():
    assert PII_PATTERNS["pan"].search("My PAN is ABCPL1234C confirmed")


def test_pan_pattern_negative_wrong_4th_char():
    # 4th char must be one of PCHFATBLJG — 'X' is invalid
    assert not PII_PATTERNS["pan"].search("Code ABCXL1234C")


def test_aadhaar_pattern():
    assert PII_PATTERNS["aadhaar"].search("2345 6789 0123")
    assert not PII_PATTERNS["aadhaar"].search("1234 5678 9012")  # starts with 1


def test_indian_mobile_pattern():
    assert PII_PATTERNS["mobile"].search("Call me on +91 98765 43210")
    assert PII_PATTERNS["mobile"].search("9876543210")
    assert not PII_PATTERNS["mobile"].search("1234567890")


def test_email_pattern():
    assert PII_PATTERNS["email"].search("contact: user@example.com")


def test_detect_pii_finds_email_column():
    df = pd.DataFrame({"customer_email": [f"user{i}@example.com" for i in range(20)],
                       "age": list(range(20))})
    result = detect_pii(df)
    assert "customer_email" in result
    assert "email" in result["customer_email"]


def test_detect_pii_empty_df_returns_empty():
    assert detect_pii(pd.DataFrame()) == {}


def test_scrub_for_llm_redacts_emails():
    df = pd.DataFrame({"email": ["alice@x.com", "bob@y.com", "carol@z.com"],
                       "score": [10, 20, 30]})
    out = scrub_for_llm(df, n_sample_rows=3)
    assert "alice@x.com" not in out
    assert "<EMAIL>" in out or "***" in out  # some redaction marker present
