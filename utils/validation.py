"""File and DataFrame validation utilities for DataVaidya.

This module provides validation primitives used across the ingestion and
profiling pipelines: file-size gating, extension normalization, DataFrame
shape checks, and a heuristic safety check that estimates whether a file
can be loaded into memory given pandas' typical expansion factor.

All public exceptions inherit from :class:`ValueError` so callers can catch
them with a single ``except ValueError`` if they prefer broad handling, or
match the specific subclass for targeted UX messaging.
"""

from __future__ import annotations

import logging
import os
from typing import Final

import pandas as pd

from utils.constants import (
    MAX_FILE_MB,
    SUPPORTED_EXTENSIONS,
    WARN_FILE_MB,
)

logger = logging.getLogger(__name__)

# Pandas typically expands CSV/TSV/JSON payloads 3-5x in RAM relative to the
# on-disk byte size (string interning, object dtype overhead, index, etc.).
# We use the upper bound to stay conservative when deciding whether to load.
_PANDAS_EXPANSION_FACTOR: Final[float] = 5.0

# Compound-extension suffixes we strip so a file named ``data.csv.gz`` is
# treated as a CSV. The order matters: longer suffixes first.
_COMPRESSION_SUFFIXES: Final[tuple[str, ...]] = (
    ".gz",
    ".bz2",
    ".zip",
    ".xz",
    ".zst",
)


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds :data:`MAX_FILE_MB`."""


class UnsupportedFileTypeError(ValueError):
    """Raised when a filename's extension is not in :data:`SUPPORTED_EXTENSIONS`."""


class EmptyDataFrameError(ValueError):
    """Raised when a DataFrame has zero rows or zero columns."""


class SchemaError(ValueError):
    """Raised when a DataFrame fails minimum row/column requirements."""


class EncodingDetectionError(ValueError):
    """Raised when text encoding sniffing fails for a textual file."""


def _bytes_to_mb(size_bytes: int) -> float:
    """Convert a byte count to megabytes (binary MB, i.e., MiB).

    Args:
        size_bytes: Size in bytes. Must be non-negative.

    Returns:
        Size in mebibytes (1 MB == 1024 * 1024 bytes).
    """
    return size_bytes / (1024.0 * 1024.0)


def validate_file_size(size_bytes: int, filename: str = "") -> tuple[bool, str | None]:
    """Validate that an uploaded file is within size limits.

    Files at or below :data:`WARN_FILE_MB` pass silently. Files between
    :data:`WARN_FILE_MB` and :data:`MAX_FILE_MB` pass with a warning message.
    Files above :data:`MAX_FILE_MB` raise :class:`FileTooLargeError`.

    Args:
        size_bytes: File size in bytes.
        filename: Optional filename, included in messages and exceptions for
            user-facing context. May be empty.

    Returns:
        A tuple ``(passes, warning_message)`` where ``passes`` is ``True`` if
        the file is within the hard limit and ``warning_message`` is a human-
        readable string if the soft threshold was crossed, otherwise ``None``.

    Raises:
        FileTooLargeError: If the file exceeds :data:`MAX_FILE_MB`.
        ValueError: If ``size_bytes`` is negative.
    """
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")

    size_mb = _bytes_to_mb(size_bytes)
    display_name = filename.strip() or "<unnamed>"

    if size_mb > MAX_FILE_MB:
        logger.warning(
            "File %s rejected: %.2f MB exceeds MAX_FILE_MB=%d",
            display_name,
            size_mb,
            MAX_FILE_MB,
        )
        raise FileTooLargeError(
            f"File '{display_name}' is {size_mb:.2f} MB, which exceeds the "
            f"maximum allowed size of {MAX_FILE_MB} MB."
        )

    if size_mb > WARN_FILE_MB:
        message = (
            f"File '{display_name}' is {size_mb:.2f} MB. "
            f"Files above {WARN_FILE_MB} MB may be slow to profile; "
            f"automatic sampling will be applied where appropriate."
        )
        logger.info(
            "File %s above warn threshold: %.2f MB > %d MB",
            display_name,
            size_mb,
            WARN_FILE_MB,
        )
        return True, message

    return True, None


def validate_extension(filename: str) -> str:
    """Return the normalized (lowercase, no-dot) extension of ``filename``.

    Compression suffixes like ``.gz``/``.bz2``/``.zip``/``.xz``/``.zst`` are
    stripped before extracting the extension, so ``data.csv.gz`` resolves to
    ``"csv"``. Comparison against :data:`SUPPORTED_EXTENSIONS` is case-
    insensitive.

    Args:
        filename: File name (with or without directory components). Leading
            and trailing whitespace is stripped.

    Returns:
        The normalized extension without the leading dot (e.g., ``"csv"``).

    Raises:
        UnsupportedFileTypeError: If ``filename`` is empty/whitespace-only,
            has no extension, or has an extension not in
            :data:`SUPPORTED_EXTENSIONS`.
    """
    cleaned = filename.strip()
    if not cleaned:
        raise UnsupportedFileTypeError(
            "Filename is empty or whitespace-only; cannot determine type."
        )

    # Use only the basename so a path like /tmp/.hidden/foo.csv works.
    base = os.path.basename(cleaned).lower()

    # Strip a single compression suffix if present (e.g., ".csv.gz" -> ".csv").
    for suffix in _COMPRESSION_SUFFIXES:
        if base.endswith(suffix) and base != suffix:
            base = base[: -len(suffix)]
            break

    # ``os.path.splitext`` returns ("foo", ".csv") or ("foo", "") for dotless
    # files. It also handles dotfiles correctly: (".env", "") not ("", ".env").
    root, ext = os.path.splitext(base)
    if not ext or not root:
        raise UnsupportedFileTypeError(
            f"File '{filename}' has no extension; supported types are: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    normalized = ext.lstrip(".")
    if normalized not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File '{filename}' has unsupported extension '.{normalized}'. "
            f"Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    logger.debug("Validated extension for %s -> %s", filename, normalized)
    return normalized


def validate_dataframe(df: pd.DataFrame) -> None:
    """Assert that a DataFrame has at least one row and one column.

    Args:
        df: The DataFrame to validate.

    Raises:
        SchemaError: If ``df`` is not a :class:`pandas.DataFrame`.
        EmptyDataFrameError: If ``df`` has zero rows or zero columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(
            f"Expected pandas.DataFrame, got {type(df).__name__}."
        )

    n_rows, n_cols = df.shape
    if n_rows == 0:
        raise EmptyDataFrameError("DataFrame has 0 rows.")
    if n_cols == 0:
        raise EmptyDataFrameError("DataFrame has 0 columns.")


def validate_schema(
    df: pd.DataFrame,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
) -> None:
    """Validate that a DataFrame meets minimum size requirements.

    This is a stricter version of :func:`validate_dataframe` used by the
    profiling pipeline where some analyses require more than one row/column
    to be meaningful (e.g., correlation needs >= 2 columns).

    Args:
        df: The DataFrame to validate.
        min_rows: Minimum required number of rows. Must be >= 0.
        min_cols: Minimum required number of columns. Must be >= 0.

    Raises:
        ValueError: If ``min_rows`` or ``min_cols`` is negative.
        SchemaError: If ``df`` is not a DataFrame or fails the size check.
        EmptyDataFrameError: If the DataFrame is completely empty.
    """
    if min_rows < 0:
        raise ValueError(f"min_rows must be non-negative, got {min_rows}")
    if min_cols < 0:
        raise ValueError(f"min_cols must be non-negative, got {min_cols}")

    validate_dataframe(df)

    n_rows, n_cols = df.shape
    if n_rows < min_rows:
        raise SchemaError(
            f"DataFrame has {n_rows} rows but at least {min_rows} required."
        )
    if n_cols < min_cols:
        raise SchemaError(
            f"DataFrame has {n_cols} columns but at least {min_cols} required."
        )


def is_safe_to_load(file_size_mb: float, available_ram_mb: float) -> tuple[bool, str]:
    """Decide whether it is safe to load a file given current RAM headroom.

    Estimates in-memory size as ``file_size_mb * 5`` (pandas' typical worst-
    case expansion for text formats) and compares against available RAM.

    Args:
        file_size_mb: File size on disk, in megabytes. Must be non-negative.
        available_ram_mb: Available process RAM headroom, in megabytes. Must
            be non-negative.

    Returns:
        A tuple ``(safe, reason)``. ``safe`` is ``True`` if the estimated
        in-memory footprint fits comfortably; ``reason`` is a human-readable
        explanation suitable for surfacing in the UI.

    Raises:
        ValueError: If either argument is negative.
    """
    if file_size_mb < 0:
        raise ValueError(f"file_size_mb must be non-negative, got {file_size_mb}")
    if available_ram_mb < 0:
        raise ValueError(
            f"available_ram_mb must be non-negative, got {available_ram_mb}"
        )

    if file_size_mb == 0:
        return False, "File is empty (0 bytes); nothing to load."

    estimated_mb = file_size_mb * _PANDAS_EXPANSION_FACTOR

    if available_ram_mb <= 0:
        return False, (
            f"No available RAM headroom; loading {file_size_mb:.2f} MB "
            f"would require ~{estimated_mb:.0f} MB."
        )

    if estimated_mb > available_ram_mb:
        return False, (
            f"Estimated in-memory size {estimated_mb:.0f} MB "
            f"(file {file_size_mb:.2f} MB x {_PANDAS_EXPANSION_FACTOR:g}) "
            f"exceeds available headroom {available_ram_mb:.0f} MB."
        )

    return True, (
        f"OK: estimated ~{estimated_mb:.0f} MB in memory, "
        f"{available_ram_mb:.0f} MB available."
    )
