"""DataVaidya — single source of truth for constants and configuration.

This module centralizes every magic number, threshold, color token, and
configuration value used across DataVaidya. No other module should redefine
these literals; import from here instead.

Sections
--------
1.  Color palette and Plotly colorway
2.  File ingestion limits
3.  Supported extensions and MIME types
4.  Health score deduction caps and zones
5.  Profiling thresholds (cardinality, imbalance, outliers)
6.  Memory guardrails
7.  LLM configuration
8.  Prompt templates
9.  App metadata
10. Indian PII regex stubs (real patterns live in core/pii.py)
11. health_zone() helper
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping, Tuple

# ---------------------------------------------------------------------------
# 1. Color palette
# ---------------------------------------------------------------------------

BG_DEEP: Final[str] = "#0F172A"
BG_SURFACE: Final[str] = "#1E293B"
PRIMARY_VIOLET: Final[str] = "#8B5CF6"
ACCENT_CYAN: Final[str] = "#06B6D4"
SUCCESS_GREEN: Final[str] = "#22C55E"
WARNING_AMBER: Final[str] = "#F59E0B"
ERROR_RED: Final[str] = "#EF4444"
TEXT_PRIMARY: Final[str] = "#F8FAFC"
TEXT_MUTED: Final[str] = "#94A3B8"

PALETTE: Final[Tuple[str, ...]] = (
    PRIMARY_VIOLET,
    ACCENT_CYAN,
    SUCCESS_GREEN,
    WARNING_AMBER,
    ERROR_RED,
    "#A78BFA",
    "#22D3EE",
    "#34D399",
    "#FBBF24",
    "#F87171",
)

# ---------------------------------------------------------------------------
# 2. File ingestion limits
# ---------------------------------------------------------------------------

MAX_FILE_MB: Final[int] = 50
WARN_FILE_MB: Final[int] = 30
SAMPLE_ROW_LIMIT: Final[int] = 10_000

# ---------------------------------------------------------------------------
# 3. Supported extensions and MIME types
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Final[Tuple[str, ...]] = (
    "csv",
    "tsv",
    "xlsx",
    "xls",
    "parquet",
    "json",
)

MIME_TYPES: Final[Mapping[str, str]] = MappingProxyType({
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "parquet": "application/vnd.apache.parquet",
    "json": "application/json",
})

# ---------------------------------------------------------------------------
# 4. Health score deduction caps and zones
# ---------------------------------------------------------------------------

MISSING_MAX: Final[int] = 30
DUPLICATES_MAX: Final[int] = 15
OUTLIERS_MAX: Final[int] = 10
IMBALANCE_MAX: Final[int] = 10
MIXED_DTYPE_MAX: Final[int] = 10
CARDINALITY_MAX: Final[int] = 5
CONSTANT_MAX: Final[int] = 10
DATE_INCONSISTENCY_MAX: Final[int] = 10

DEDUCTION_CAPS: Final[Mapping[str, int]] = MappingProxyType({
    "missing": MISSING_MAX,
    "duplicates": DUPLICATES_MAX,
    "outliers": OUTLIERS_MAX,
    "imbalance": IMBALANCE_MAX,
    "mixed_dtype": MIXED_DTYPE_MAX,
    "cardinality": CARDINALITY_MAX,
    "constant": CONSTANT_MAX,
    "date_inconsistency": DATE_INCONSISTENCY_MAX,
})

ZONE_CRITICAL: Final[Tuple[int, int]] = (0, 40)
ZONE_NEEDS_WORK: Final[Tuple[int, int]] = (41, 60)
ZONE_GOOD: Final[Tuple[int, int]] = (61, 80)
ZONE_EXCELLENT: Final[Tuple[int, int]] = (81, 100)

ZONE_COLORS: Final[Mapping[str, str]] = MappingProxyType({
    "Critical": ERROR_RED,
    "Needs Work": WARNING_AMBER,
    "Good": ACCENT_CYAN,
    "Excellent": SUCCESS_GREEN,
})

# ---------------------------------------------------------------------------
# 5. Profiling thresholds
# ---------------------------------------------------------------------------

HIGH_CARDINALITY_RATIO: Final[float] = 0.5
HIGH_CARDINALITY_MIN_UNIQUE: Final[int] = 50
IMBALANCE_GINI_THRESHOLD: Final[float] = 0.7
IQR_MULTIPLIER: Final[float] = 1.5
ZSCORE_THRESHOLD: Final[float] = 3.0

# ---------------------------------------------------------------------------
# 6. Memory guardrails (Streamlit Community Cloud — 1 GB working ceiling)
# ---------------------------------------------------------------------------

MEMORY_BUDGET_MB: Final[int] = 1024
MEMORY_WARN_PCT: Final[int] = 70
MEMORY_CRITICAL_PCT: Final[int] = 85
MEMORY_GUARD_MB: Final[int] = 900
VIZ_SAMPLE_ROWS: Final[int] = 10_000

# ---------------------------------------------------------------------------
# 7. LLM configuration
# ---------------------------------------------------------------------------

GROQ_MODEL: Final[str] = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODELS: Final[Tuple[str, ...]] = (
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
)
GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL: Final[str] = "gemini-2.5-flash-lite"
LLM_TIMEOUT_S: Final[int] = 15
LLM_MAX_RETRIES: Final[int] = 3
LLM_MAX_TOKENS: Final[int] = 2000
LLM_SESSION_CAP: Final[int] = 5
LLM_CONTEXT_TOKEN_BUDGET: Final[int] = 4000
LLM_TEMPERATURE: Final[float] = 0.2

LLM_FALLBACK_MESSAGE: Final[str] = (
    "AI insights are temporarily unavailable. Your data quality report below "
    "still includes all rule-based diagnostics."
)

# ---------------------------------------------------------------------------
# 8. Prompt template (versioned — bump suffix to bust LLM prefix cache)
# ---------------------------------------------------------------------------

PROMPT_V1: Final[str] = (
    "You are a senior data analyst at a top Indian consulting firm. "
    "Analyze this dataset summary and produce:\n"
    "1. Three business insights (with the column names that prove each)\n"
    "2. Two data quality warnings (specific + actionable)\n"
    "3. Two suggested next analyses (specific SQL/Python steps)\n"
    "Format as clean Markdown with headers and bullets. "
    "Use Indian business context where relevant. No fluff. No disclaimers."
)

# ---------------------------------------------------------------------------
# 9. App metadata
# ---------------------------------------------------------------------------

APP_NAME: Final[str] = "DataVaidya"
APP_VERSION: Final[str] = "0.1.0"
APP_EMOJI: Final[str] = "🧪"
TAGLINE: Final[str] = "Upload. Diagnose. Clean. Ship. In 60 seconds."

# ---------------------------------------------------------------------------
# 10. Indian PII regex stubs (authoritative patterns live in core/pii.py)
# ---------------------------------------------------------------------------

INDIAN_PII_PATTERNS: Final[Mapping[str, str]] = MappingProxyType({
    "pan": r"",
    "aadhaar": r"",
    "gstin": r"",
    "mobile_in": r"",
    "pincode": r"",
    "ifsc": r"",
    "email": r"",
    "credit_card": r"",
})

# ---------------------------------------------------------------------------
# 11. health_zone() helper
# ---------------------------------------------------------------------------


def health_zone(score: int) -> str:
    """Map an integer health score to its named zone.

    Zones follow the DataVaidya scoring rubric:

    * ``0-40``   -> ``"Critical"``
    * ``41-60``  -> ``"Needs Work"``
    * ``61-80``  -> ``"Good"``
    * ``81-100`` -> ``"Excellent"``

    Parameters
    ----------
    score:
        Integer health score in the inclusive range ``[0, 100]``.

    Returns
    -------
    str
        One of ``"Critical"``, ``"Needs Work"``, ``"Good"``, ``"Excellent"``.

    Raises
    ------
    ValueError
        If ``score`` is outside ``[0, 100]``.
    """
    if not 0 <= score <= 100:
        raise ValueError(f"health score must be in [0, 100]; got {score}")
    if ZONE_CRITICAL[0] <= score <= ZONE_CRITICAL[1]:
        return "Critical"
    if ZONE_NEEDS_WORK[0] <= score <= ZONE_NEEDS_WORK[1]:
        return "Needs Work"
    if ZONE_GOOD[0] <= score <= ZONE_GOOD[1]:
        return "Good"
    return "Excellent"


__all__ = [
    "BG_DEEP", "BG_SURFACE", "PRIMARY_VIOLET", "ACCENT_CYAN",
    "SUCCESS_GREEN", "WARNING_AMBER", "ERROR_RED", "TEXT_PRIMARY",
    "TEXT_MUTED", "PALETTE",
    "MAX_FILE_MB", "WARN_FILE_MB", "SAMPLE_ROW_LIMIT",
    "SUPPORTED_EXTENSIONS", "MIME_TYPES",
    "MISSING_MAX", "DUPLICATES_MAX", "OUTLIERS_MAX", "IMBALANCE_MAX",
    "MIXED_DTYPE_MAX", "CARDINALITY_MAX", "CONSTANT_MAX",
    "DATE_INCONSISTENCY_MAX", "DEDUCTION_CAPS",
    "ZONE_CRITICAL", "ZONE_NEEDS_WORK", "ZONE_GOOD", "ZONE_EXCELLENT",
    "ZONE_COLORS",
    "HIGH_CARDINALITY_RATIO", "HIGH_CARDINALITY_MIN_UNIQUE",
    "IMBALANCE_GINI_THRESHOLD", "IQR_MULTIPLIER", "ZSCORE_THRESHOLD",
    "MEMORY_BUDGET_MB", "MEMORY_WARN_PCT", "MEMORY_CRITICAL_PCT",
    "MEMORY_GUARD_MB", "VIZ_SAMPLE_ROWS",
    "GROQ_MODEL", "GROQ_FALLBACK_MODELS", "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS", "LLM_SESSION_CAP", "LLM_CONTEXT_TOKEN_BUDGET",
    "LLM_TEMPERATURE", "LLM_FALLBACK_MESSAGE",
    "PROMPT_V1",
    "APP_NAME", "APP_VERSION", "APP_EMOJI", "TAGLINE",
    "INDIAN_PII_PATTERNS",
    "health_zone",
]
