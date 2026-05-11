"""Ingestion layer for DataVaidya.

This module is responsible for turning raw uploaded bytes (or demo-dataset
paths) into a validated, in-memory :class:`pandas.DataFrame` together with a
structured :class:`IngestionReport` describing how the bytes were interpreted.

Design notes:
    * No top-level Streamlit imports — the module is fully testable headless.
      The Streamlit pages should wrap calls to :func:`load_uploaded` /
      :func:`load_demo` with ``@st.cache_data`` at the page layer.
    * All public readers return ``(DataFrame, IngestionReport)`` so the UI can
      surface encoding, delimiter, sheet, and sampling decisions to the user.
    * Memory-bounded reads are wrapped in :func:`utils.memory.memory_guard`
      so that out-of-memory situations become an actionable, user-facing
      :class:`MemoryError` rather than a process crash.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import chardet
import pandas as pd

from utils.constants import (
    MAX_FILE_MB,
    SAMPLE_ROW_LIMIT,
    SUPPORTED_EXTENSIONS,
    WARN_FILE_MB,
)
from utils.memory import memory_guard
from utils.validation import (
    EncodingDetectionError,
    validate_dataframe,
    validate_extension,
    validate_file_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass + constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionReport:
    """Structured record of how an upload was decoded.

    Attributes:
        encoding: Character encoding used to decode bytes (``None`` for
            binary formats such as Parquet / Excel).
        delimiter: Field delimiter detected for delimited formats
            (``None`` for non-delimited formats).
        sheet: Sheet name that was loaded (Excel only; ``None`` otherwise).
        sampled: ``True`` if the returned frame is a row-limited sample
            rather than the full dataset.
        rows_returned: Number of rows actually present in the returned frame.
        rows_total_estimate: Best-effort estimate of total rows in the source
            (``None`` when not cheaply computable).
        warnings: Human-readable warning strings worth surfacing in the UI.
    """

    encoding: str | None
    delimiter: str | None
    sheet: str | None
    sampled: bool
    rows_returned: int
    rows_total_estimate: int | None
    warnings: tuple[str, ...]


DEMO_DATASETS: Final[dict[str, str]] = {
    "Titanic": "data/samples/titanic.csv",
    "Iris": "data/samples/iris.csv",
    "Indian Census 2011": "data/samples/census_india_2011.csv",
    "Mumbai Real Estate": "data/samples/mumbai_real_estate.csv",
    "Indian Retail Transactions": "data/samples/retail_transactions.csv",
}

# Regex matching plausible ISO-ish dates for opportunistic date coercion.
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$"
)

# Sniffer sample size (16 KB is enough to capture delimiter + quoting).
_SNIFF_BYTES: Final[int] = 16 * 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _should_sample(file_bytes: bytes, explicit: bool | None) -> bool:
    """Decide whether to read a row-limited sample.

    Args:
        file_bytes: Raw uploaded bytes.
        explicit: User-supplied override; ``None`` means auto-decide.

    Returns:
        ``True`` if the read should be sampled to ``SAMPLE_ROW_LIMIT``.
    """
    if explicit is not None:
        return explicit
    size_mb = len(file_bytes) / (1024 * 1024)
    return size_mb > WARN_FILE_MB


def _detect_encoding(buf: bytes) -> str:
    """Detect the most likely text encoding of ``buf``.

    The strategy is: try strict UTF-8 first (the most common case), then
    UTF-8-with-BOM, then Latin-1 (which never raises but may produce
    mojibake), and finally fall back to :mod:`chardet` for everything else.

    Args:
        buf: Raw bytes to probe (a small prefix is sufficient).

    Returns:
        The name of a codec accepted by :func:`bytes.decode`.

    Raises:
        EncodingDetectionError: If every strategy — including ``chardet`` —
            fails to produce a usable encoding.
    """
    probe = buf[: 64 * 1024] if len(buf) > 64 * 1024 else buf

    # Cheap fast path: explicit BOM means utf-8-sig.
    if probe.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for enc in ("utf-8", "utf-8-sig"):
        try:
            probe.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue

    try:
        detected = chardet.detect(probe) or {}
        encoding = detected.get("encoding")
        confidence = float(detected.get("confidence") or 0.0)
        if encoding and confidence >= 0.5:
            return encoding.lower()
    except (TypeError, ValueError) as exc:
        logger.debug("chardet detection raised: %s", exc)

    # Latin-1 is a true last resort: it always decodes but may mangle text.
    try:
        probe.decode("latin-1")
        return "latin-1"
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 cannot fail
        raise EncodingDetectionError(
            "Could not detect a usable text encoding for the uploaded file."
        ) from exc


def _detect_delimiter(text: str) -> str:
    """Best-effort delimiter detection via :class:`csv.Sniffer`.

    Args:
        text: A decoded text sample (ideally the first ~16 KB of the file).

    Returns:
        The detected single-character delimiter, defaulting to ``","`` on
        any sniffer failure (single-column files trip the sniffer).
    """
    if not text.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error as exc:
        logger.debug("csv.Sniffer failed, defaulting to ',': %s", exc)
        return ","


def _strip_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and remove leading BOM from column headers.

    Mutates ``df`` in place for efficiency on wide frames and returns it for
    fluent chaining.

    Args:
        df: Frame whose columns should be normalised.

    Returns:
        The same frame with cleaned column labels.
    """
    new_cols: list[Any] = []
    for col in df.columns:
        if isinstance(col, str):
            cleaned = col.lstrip("﻿").strip()
            new_cols.append(cleaned)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


def _coerce_obvious_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that look like ISO dates to datetime64.

    Only columns whose **first non-null** value matches an ISO-like pattern
    are attempted. ``errors='ignore'`` ensures any column that resists
    parsing is left untouched.

    Args:
        df: Frame to scan and possibly mutate.

    Returns:
        The same frame, with eligible object columns coerced to datetimes.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if not isinstance(first, str) or not _ISO_DATE_RE.match(first):
            continue
        try:
            converted = pd.to_datetime(df[col], errors="ignore")
            df[col] = converted
        except (ValueError, TypeError) as exc:
            logger.debug("Date coercion skipped for %r: %s", col, exc)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply post-read normalisation common to every reader.

    Args:
        df: Frame fresh out of a pandas reader.

    Returns:
        Validated, header-cleaned, opportunistically date-coerced frame.
    """
    validate_dataframe(df)
    df = _strip_column_whitespace(df)
    df = _coerce_obvious_dates(df)
    return df


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def load_uploaded(
    file_bytes: bytes,
    filename: str,
    sample: bool | None = None,
) -> tuple[pd.DataFrame, IngestionReport]:
    """Validate and dispatch an uploaded file to the correct reader.

    The function performs (in order): extension validation, size validation,
    sampling-decision, reader dispatch, and memory guarding.

    Args:
        file_bytes: Raw uploaded payload.
        filename: Original filename (used for extension detection only —
            the bytes are never written to disk).
        sample: Explicit override for sampling. ``None`` triggers
            auto-decision based on file size vs. :data:`WARN_FILE_MB`.

    Returns:
        Tuple of the loaded frame and a populated
        :class:`IngestionReport`.

    Raises:
        UnsupportedExtensionError: If ``filename`` has an unknown extension.
        FileTooLargeError: If the upload exceeds :data:`MAX_FILE_MB`.
        EmptyDataFrameError: If the upload decodes to zero rows.
        MemoryError: If decoding would exceed the configured RAM headroom.
    """
    validate_extension(filename, SUPPORTED_EXTENSIONS)
    validate_file_size(file_bytes, MAX_FILE_MB)

    ext = Path(filename).suffix.lower().lstrip(".")
    do_sample = _should_sample(file_bytes, sample)

    logger.info(
        "Ingesting upload filename=%s ext=%s size_mb=%.2f sample=%s",
        filename,
        ext,
        len(file_bytes) / (1024 * 1024),
        do_sample,
    )

    try:
        with memory_guard():
            if ext == "csv":
                return read_csv(file_bytes, sample=do_sample)
            if ext == "tsv":
                return read_tsv(file_bytes, sample=do_sample)
            if ext in {"xlsx", "xls", "xlsm"}:
                return read_excel(file_bytes, sheet=0, sample=do_sample)
            if ext == "parquet":
                return read_parquet(file_bytes, sample=do_sample)
            if ext in {"json", "jsonl", "ndjson"}:
                return read_json(file_bytes, sample=do_sample)
            # validate_extension should have rejected this already.
            raise ValueError(f"No reader available for extension: {ext!r}")
    except MemoryError as exc:
        raise MemoryError(
            "file exceeded RAM headroom; retry with sample=True"
        ) from exc


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------


def read_csv(
    file_bytes: bytes, sample: bool = False
) -> tuple[pd.DataFrame, IngestionReport]:
    """Read a CSV payload with progressive encoding fallback.

    Encoding precedence: ``utf-8`` → ``utf-8-sig`` → ``latin-1`` → ``chardet``.
    Delimiter detection uses :class:`csv.Sniffer` over the first 16 KB and
    silently falls back to ``","`` when sniffing fails (which it does on
    single-column files).

    Args:
        file_bytes: Raw CSV bytes.
        sample: When ``True``, read at most :data:`SAMPLE_ROW_LIMIT` rows.

    Returns:
        Tuple of the loaded frame and its :class:`IngestionReport`.
    """
    warnings: list[str] = []

    encoding = _detect_encoding(file_bytes)
    try:
        text_head = file_bytes[:_SNIFF_BYTES].decode(encoding, errors="replace")
    except LookupError:
        # Unknown codec name from chardet → fall back hard to latin-1.
        warnings.append(f"Unknown encoding {encoding!r}; falling back to latin-1.")
        encoding = "latin-1"
        text_head = file_bytes[:_SNIFF_BYTES].decode(encoding, errors="replace")

    delimiter = _detect_delimiter(text_head)

    read_kwargs: dict[str, Any] = {
        "encoding": encoding,
        "sep": delimiter,
        "low_memory": False,
    }
    if sample:
        read_kwargs["nrows"] = SAMPLE_ROW_LIMIT

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)
    except UnicodeDecodeError:
        warnings.append(
            f"Decoding with {encoding!r} failed mid-stream; retrying as latin-1."
        )
        encoding = "latin-1"
        read_kwargs["encoding"] = encoding
        df = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)

    df = _finalize(df)

    report = IngestionReport(
        encoding=encoding,
        delimiter=delimiter,
        sheet=None,
        sampled=sample,
        rows_returned=len(df),
        rows_total_estimate=None,
        warnings=tuple(warnings),
    )
    return df, report


def read_tsv(
    file_bytes: bytes, sample: bool = False
) -> tuple[pd.DataFrame, IngestionReport]:
    """Read a tab-separated payload (delimiter is hard-coded to ``\\t``).

    Encoding detection and sampling behaviour mirror :func:`read_csv`.

    Args:
        file_bytes: Raw TSV bytes.
        sample: When ``True``, read at most :data:`SAMPLE_ROW_LIMIT` rows.

    Returns:
        Tuple of the loaded frame and its :class:`IngestionReport`.
    """
    warnings: list[str] = []
    encoding = _detect_encoding(file_bytes)

    read_kwargs: dict[str, Any] = {
        "encoding": encoding,
        "sep": "\t",
        "low_memory": False,
    }
    if sample:
        read_kwargs["nrows"] = SAMPLE_ROW_LIMIT

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)
    except UnicodeDecodeError:
        warnings.append(
            f"Decoding with {encoding!r} failed mid-stream; retrying as latin-1."
        )
        encoding = "latin-1"
        read_kwargs["encoding"] = encoding
        df = pd.read_csv(io.BytesIO(file_bytes), **read_kwargs)

    df = _finalize(df)

    report = IngestionReport(
        encoding=encoding,
        delimiter="\t",
        sheet=None,
        sampled=sample,
        rows_returned=len(df),
        rows_total_estimate=None,
        warnings=tuple(warnings),
    )
    return df, report


def read_excel(
    file_bytes: bytes,
    sheet: str | int = 0,
    sample: bool = False,
) -> tuple[pd.DataFrame, IngestionReport]:
    """Read an Excel payload via the openpyxl engine.

    When the workbook has multiple sheets and the caller did not specify
    one, the first sheet is loaded and the remaining sheet names are
    surfaced in :attr:`IngestionReport.warnings` so the UI can offer a
    sheet selector on the next round-trip.

    Args:
        file_bytes: Raw ``.xlsx`` / ``.xlsm`` bytes.
        sheet: Sheet identifier (name or 0-indexed position).
        sample: When ``True``, slice the first :data:`SAMPLE_ROW_LIMIT`
            rows after reading.

    Returns:
        Tuple of the loaded frame and its :class:`IngestionReport`.
    """
    warnings: list[str] = []
    buf = io.BytesIO(file_bytes)

    try:
        all_sheets = pd.ExcelFile(buf, engine="openpyxl").sheet_names
    except Exception as exc:  # noqa: BLE001 - openpyxl raises many types
        logger.debug("Could not enumerate sheets: %s", exc)
        all_sheets = []

    if isinstance(sheet, int):
        if all_sheets and 0 <= sheet < len(all_sheets):
            sheet_name: str | int = all_sheets[sheet]
        else:
            sheet_name = sheet
    else:
        sheet_name = sheet

    if isinstance(sheet, int) and sheet == 0 and len(all_sheets) > 1:
        others = [s for s in all_sheets if s != sheet_name]
        warnings.append(
            f"Workbook has multiple sheets; loaded {sheet_name!r}. "
            f"Other sheets: {others}"
        )

    buf.seek(0)
    df = pd.read_excel(buf, sheet_name=sheet_name, engine="openpyxl")

    if sample and len(df) > SAMPLE_ROW_LIMIT:
        df = df.head(SAMPLE_ROW_LIMIT).copy()

    df = _finalize(df)

    report = IngestionReport(
        encoding=None,
        delimiter=None,
        sheet=str(sheet_name),
        sampled=sample,
        rows_returned=len(df),
        rows_total_estimate=None,
        warnings=tuple(warnings),
    )
    return df, report


def read_parquet(
    file_bytes: bytes, sample: bool = False
) -> tuple[pd.DataFrame, IngestionReport]:
    """Read a Parquet payload via the pyarrow engine.

    If pyarrow rejects the file because it contains nested types it cannot
    map to pandas, we retry without specifying ``engine`` so pandas can
    pick whichever backend it has available.

    Args:
        file_bytes: Raw Parquet bytes.
        sample: When ``True``, slice the first :data:`SAMPLE_ROW_LIMIT`
            rows after reading.

    Returns:
        Tuple of the loaded frame and its :class:`IngestionReport`.
    """
    warnings: list[str] = []

    try:
        df = pd.read_parquet(io.BytesIO(file_bytes), engine="pyarrow")
    except Exception as exc:  # noqa: BLE001 - pyarrow.ArrowInvalid + friends
        if "Arrow" in type(exc).__name__ or "arrow" in str(exc).lower():
            warnings.append(
                "Parquet contains types pyarrow could not map; retrying with "
                "pandas' default engine."
            )
            logger.warning("Parquet pyarrow read failed: %s; retrying.", exc)
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            raise

    if sample and len(df) > SAMPLE_ROW_LIMIT:
        df = df.head(SAMPLE_ROW_LIMIT).copy()

    df = _finalize(df)

    report = IngestionReport(
        encoding=None,
        delimiter=None,
        sheet=None,
        sampled=sample,
        rows_returned=len(df),
        rows_total_estimate=None,
        warnings=tuple(warnings),
    )
    return df, report


def read_json(
    file_bytes: bytes, sample: bool = False
) -> tuple[pd.DataFrame, IngestionReport]:
    """Read a JSON or JSON-Lines payload, auto-detecting the structure.

    Detection rule based on the first non-whitespace character:
        * ``[`` → JSON array of records.
        * ``{`` → either a single object (becomes a one-row frame) or a
          JSON-Lines stream — both are attempted.

    Args:
        file_bytes: Raw JSON bytes.
        sample: When ``True``, slice the first :data:`SAMPLE_ROW_LIMIT`
            rows after reading.

    Returns:
        Tuple of the loaded frame and its :class:`IngestionReport`.

    Raises:
        ValueError: If the payload cannot be parsed as records, a single
            object, or JSON-Lines.
    """
    warnings: list[str] = []
    encoding = _detect_encoding(file_bytes)

    try:
        text = file_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        encoding = "utf-8"
        text = file_bytes.decode(encoding, errors="replace")
        warnings.append("JSON re-decoded with utf-8 + replacement characters.")

    stripped = text.lstrip()
    if not stripped:
        # Defer to validate_dataframe / EmptyDataFrameError downstream.
        df = pd.DataFrame()
        df = _finalize(df)
        report = IngestionReport(
            encoding=encoding,
            delimiter=None,
            sheet=None,
            sampled=sample,
            rows_returned=0,
            rows_total_estimate=0,
            warnings=tuple(warnings),
        )
        return df, report

    first = stripped[0]
    df: pd.DataFrame | None = None

    if first == "[":
        df = pd.read_json(io.StringIO(text), orient="records")
    elif first == "{":
        # Try single-object first, then JSON-Lines.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                # If the dict's values are all list-like of equal length,
                # treat it as a column-oriented dataset; otherwise wrap it.
                if obj and all(isinstance(v, list) for v in obj.values()):
                    df = pd.DataFrame(obj)
                else:
                    df = pd.DataFrame([obj])
            elif isinstance(obj, list):
                df = pd.DataFrame(obj)
        except json.JSONDecodeError:
            df = None

        if df is None:
            try:
                df = pd.read_json(io.StringIO(text), lines=True)
                warnings.append("Parsed payload as JSON-Lines (one object per line).")
            except ValueError as exc:
                raise ValueError(
                    "JSON payload is neither a records array, single object, "
                    "nor JSON-Lines."
                ) from exc
    else:
        # Unknown prefix — last-ditch attempts.
        try:
            df = pd.read_json(io.StringIO(text), orient="records")
        except ValueError:
            df = pd.read_json(io.StringIO(text), lines=True)
            warnings.append("Parsed payload as JSON-Lines (fallback).")

    if sample and df is not None and len(df) > SAMPLE_ROW_LIMIT:
        df = df.head(SAMPLE_ROW_LIMIT).copy()

    assert df is not None  # for type-checkers
    df = _finalize(df)

    report = IngestionReport(
        encoding=encoding,
        delimiter=None,
        sheet=None,
        sampled=sample,
        rows_returned=len(df),
        rows_total_estimate=None,
        warnings=tuple(warnings),
    )
    return df, report


# ---------------------------------------------------------------------------
# Excel helpers + demo loader + schema preview
# ---------------------------------------------------------------------------


def list_excel_sheets(file_bytes: bytes) -> list[str]:
    """Enumerate the sheet names of an Excel workbook.

    Args:
        file_bytes: Raw ``.xlsx`` / ``.xlsm`` bytes.

    Returns:
        Ordered list of sheet names as reported by openpyxl.
    """
    return pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl").sheet_names


def load_demo(name: str) -> pd.DataFrame:
    """Load a bundled demo dataset by friendly name.

    Demo files are generated lazily by ``make_demo.py``; if a file is
    missing we log a warning and return an empty frame so the UI can still
    render gracefully instead of crashing.

    Args:
        name: Key into :data:`DEMO_DATASETS` (e.g. ``"Titanic"``).

    Returns:
        Loaded frame, or an empty :class:`pandas.DataFrame` when the demo
        file does not exist on disk.

    Raises:
        KeyError: If ``name`` is not a registered demo dataset.
    """
    if name not in DEMO_DATASETS:
        raise KeyError(
            f"Unknown demo dataset {name!r}. Known: {sorted(DEMO_DATASETS)}"
        )

    rel_path = DEMO_DATASETS[name]
    path = Path(rel_path)
    if not path.exists():
        logger.warning(
            "Demo dataset %r not found at %s; returning empty frame. "
            "Run make_demo.py to generate sample data.",
            name,
            path,
        )
        return pd.DataFrame()

    try:
        with path.open("rb") as fh:
            file_bytes = fh.read()
    except OSError as exc:
        logger.warning("Could not read demo dataset %s: %s", path, exc)
        return pd.DataFrame()

    suffix = path.suffix.lower().lstrip(".")
    try:
        if suffix == "csv":
            df, _ = read_csv(file_bytes)
        elif suffix == "tsv":
            df, _ = read_tsv(file_bytes)
        elif suffix in {"xlsx", "xls", "xlsm"}:
            df, _ = read_excel(file_bytes)
        elif suffix == "parquet":
            df, _ = read_parquet(file_bytes)
        elif suffix in {"json", "jsonl", "ndjson"}:
            df, _ = read_json(file_bytes)
        else:
            logger.warning("Demo dataset %s has unsupported extension.", path)
            return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 - demo loads must not crash UI
        logger.warning("Failed to load demo %s (%s): %s", name, path, exc)
        return pd.DataFrame()

    # Stable cache-busting fingerprint for Streamlit @st.cache_data callers.
    df.attrs["demo_fingerprint"] = hashlib.sha1(file_bytes).hexdigest()[:12]
    df.attrs["demo_name"] = name
    return df


def preview_schema(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """Build a compact schema-overview frame suitable for the UI.

    The output has one row per column of ``df`` with these fields:

    * ``column`` — original column name.
    * ``dtype`` — pandas dtype as a string.
    * ``non_null`` — count of non-null values.
    * ``memory_kb`` — memory used by the column, in kilobytes (2 dp).
    * ``sample_values`` — comma-separated string of up to ``n_rows`` distinct
      non-null sample values.

    Args:
        df: Input frame to summarise.
        n_rows: Maximum number of sample values per column.

    Returns:
        A small :class:`pandas.DataFrame` describing the schema of ``df``.
    """
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(
            columns=["column", "dtype", "non_null", "memory_kb", "sample_values"]
        )

    rows: list[dict[str, Any]] = []
    memory_per_col = df.memory_usage(deep=True, index=False)

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        mem_bytes = int(memory_per_col.get(col, 0))
        sample_series = series.dropna().drop_duplicates().head(n_rows)
        sample_values = ", ".join(
            str(v) if not isinstance(v, str) else v for v in sample_series.tolist()
        )
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "memory_kb": round(mem_bytes / 1024, 2),
                "sample_values": sample_values,
            }
        )

    return pd.DataFrame(
        rows, columns=["column", "dtype", "non_null", "memory_kb", "sample_values"]
    )
