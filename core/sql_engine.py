"""SQL execution engine backed by an in-memory DuckDB connection.

This module exposes a small, defensive API for running ad-hoc SQL against
user-supplied :class:`pandas.DataFrame` objects from a Streamlit front end.
The module is importable headless: Streamlit is only imported inside
:func:`get_connection`, which is decorated with ``@st.cache_resource`` so a
single connection is reused for the lifetime of the session.

Safety model
------------
Real safety comes from constructing the DuckDB connection with
``enable_external_access='false'`` and ``lock_configuration='true'``, which
disables ATTACH / COPY / httpfs / extension loading at the engine level.
The :func:`validate_sql` keyword blocklist is defense-in-depth only.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class UnsafeSQLError(ValueError):
    """Raised when SQL contains blocked keywords or multi-statement chaining."""


class SQLTimeoutError(RuntimeError):
    """Raised when a query exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------

#: Keywords rejected by :func:`validate_sql`. Defense-in-depth only — the
#: authoritative guard is the connection-level config in :func:`get_connection`.
BLOCKED_KEYWORDS: frozenset[str] = frozenset(
    {
        "ATTACH",
        "DETACH",
        "COPY",
        "INSTALL",
        "LOAD",
        "PRAGMA",
        "EXPORT",
        "IMPORT",
        "CALL",
        ".SYSTEM",
        ".SHELL",
        "UPDATE_EXTENSIONS",
        "CREATE_MACRO",
    }
)

DEFAULT_MAX_ROWS: int = 100_000
DEFAULT_TIMEOUT_S: float = 30.0
HISTORY_LIMIT: int = 200

# Regex used to validate identifiers passed to register_dataframe.
_IDENTIFIER_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Regex matching a trailing LIMIT clause (case-insensitive). We treat the
# query as already-limited if it contains a top-level LIMIT keyword.
_LIMIT_RE: re.Pattern[str] = re.compile(r"\bLIMIT\b", re.IGNORECASE)

# Patterns used by _strip_comments / _strip_string_literals.
_BLOCK_COMMENT_RE: re.Pattern[str] = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE: re.Pattern[str] = re.compile(r"--[^\n]*")
# Match single- or double-quoted string literals, including doubled-quote
# escapes (e.g. 'it''s'). Non-greedy with an explicit escape allowance.
_SINGLE_STRING_RE: re.Pattern[str] = re.compile(r"'(?:''|[^'])*'")
_DOUBLE_STRING_RE: re.Pattern[str] = re.compile(r'"(?:""|[^"])*"')

# Word tokenizer — yields runs of [A-Za-z_][A-Za-z0-9_]* style tokens.
_WORD_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the per-session in-memory DuckDB connection.

    Streamlit is imported lazily so the rest of this module remains
    importable in non-Streamlit contexts (tests, CLIs, notebooks).

    The connection is configured to disable any external access:

    * ``enable_external_access='false'`` — disables ATTACH, httpfs, COPY
      to/from the filesystem, and extension auto-install.
    * ``lock_configuration='true'`` — prevents the session from re-enabling
      external access via ``SET`` later on.

    Returns:
        A cached, session-scoped :class:`duckdb.DuckDBPyConnection`.
    """
    import streamlit as st  # local import — keeps module headless-importable.

    @st.cache_resource(show_spinner=False)
    def _cached_connection() -> duckdb.DuckDBPyConnection:
        config = {
            "enable_external_access": "false",
            "lock_configuration": "true",
        }
        conn = duckdb.connect(database=":memory:", config=config)
        logger.debug("Created new DuckDB session connection: %r", conn)
        return conn

    return _cached_connection()


# ---------------------------------------------------------------------------
# DataFrame registration
# ---------------------------------------------------------------------------


def _identifier_safe(name: str) -> bool:
    """Return ``True`` if *name* is a safe SQL identifier."""
    return bool(_IDENTIFIER_RE.match(name))


def register_dataframe(name: str, df: pd.DataFrame) -> None:
    """Register or replace *df* as a DuckDB view named *name*.

    Args:
        name: The view name. Must match ``[A-Za-z_][A-Za-z0-9_]*``.
        df:   The pandas DataFrame to register.

    Raises:
        ValueError: If *name* is not a safe identifier.
    """
    if not isinstance(name, str) or not _identifier_safe(name):
        raise ValueError(
            f"invalid identifier {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    conn = get_connection()
    # register() replaces any existing view of the same name in-place.
    conn.register(name, df)
    logger.debug("Registered DataFrame as view %s (rows=%d)", name, len(df))


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------


def _strip_comments(sql: str) -> str:
    """Strip ``/* ... */`` block comments and ``-- ...`` line comments."""
    no_block = _BLOCK_COMMENT_RE.sub(" ", sql)
    return _LINE_COMMENT_RE.sub(" ", no_block)


def _strip_string_literals(sql: str) -> str:
    """Replace string literal bodies with empty quotes.

    This ensures blocklist tokens hidden inside ``'...'`` or ``"..."`` are not
    flagged, e.g. ``SELECT 'DROP'`` becomes ``SELECT ''``.
    """
    no_single = _SINGLE_STRING_RE.sub("''", sql)
    return _DOUBLE_STRING_RE.sub('""', no_single)


def _tokenize(sql: str) -> list[str]:
    """Tokenize *sql* into uppercase word tokens."""
    return [m.group(0).upper() for m in _WORD_RE.finditer(sql)]


def validate_sql(sql: str) -> None:
    """Validate *sql* against the blocklist and multi-statement rule.

    The algorithm is deliberately conservative:

    1. Strip block and line comments.
    2. Strip the *bodies* of string literals so that keywords hidden in
       quoted text do not trigger the blocklist.
    3. Reject the query if a semicolon remains after stripping a single
       optional trailing ``;``.
    4. Tokenize the cleaned SQL and reject if any token is in
       :data:`BLOCKED_KEYWORDS`. Dotted commands like ``.SYSTEM`` are
       checked as substrings since they would not tokenize as words.
    5. Reject empty / whitespace-only input.

    Raises:
        ValueError: If the query is empty.
        UnsafeSQLError: If the query contains multiple statements or a
            blocked keyword.
    """
    if sql is None or not sql.strip():
        raise ValueError("empty query")

    cleaned = _strip_string_literals(_strip_comments(sql))

    # Drop a single trailing semicolon, then check for any remaining ones.
    trimmed = cleaned.rstrip().rstrip(";")
    if ";" in trimmed:
        raise UnsafeSQLError("multi-statement SQL is not allowed")

    upper = cleaned.upper()
    # Dotted CLI commands (.SYSTEM, .SHELL) won't survive word tokenization.
    for needle in (".SYSTEM", ".SHELL"):
        if needle in upper:
            raise UnsafeSQLError(f"blocked keyword: {needle}")

    tokens = set(_tokenize(cleaned))
    # Subtract the dotted entries which we already covered above.
    word_blocked = {kw for kw in BLOCKED_KEYWORDS if not kw.startswith(".")}
    hit = tokens & word_blocked
    if hit:
        # Sort for deterministic error messages.
        raise UnsafeSQLError(
            "blocked keyword(s): " + ", ".join(sorted(hit))
        )


# ---------------------------------------------------------------------------
# Timeout-enforced execution
# ---------------------------------------------------------------------------


class _ExecResult:
    """Mutable holder used by :func:`_run_with_timeout` to ferry data back."""

    __slots__ = ("df", "error")

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.error: BaseException | None = None


def _run_with_timeout(
    conn: duckdb.DuckDBPyConnection, sql: str, timeout_s: float
) -> pd.DataFrame:
    """Execute *sql* on *conn* with a wall-clock timeout.

    The query runs on a daemon worker thread. If it does not finish within
    *timeout_s* seconds, the main thread calls :py:meth:`conn.interrupt` and
    raises :class:`SQLTimeoutError`. The worker is given a short grace period
    to unwind before we surface the timeout.

    Args:
        conn: The DuckDB connection.
        sql: The validated query.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        The result as a :class:`pandas.DataFrame`.

    Raises:
        SQLTimeoutError: If execution exceeds *timeout_s*.
        duckdb.Error: Any DuckDB-level error is re-raised unchanged.
    """
    result = _ExecResult()

    def _worker() -> None:
        try:
            # Materialize within the worker so the timeout covers fetch as
            # well as planning/execution.
            result.df = conn.execute(sql).fetch_df()
        except BaseException as exc:  # noqa: BLE001 — captured for the caller.
            result.error = exc

    thread = threading.Thread(
        target=_worker, name="duckdb-exec", daemon=True
    )
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        # Ask DuckDB to abort, then briefly wait for the worker to unwind.
        try:
            conn.interrupt()
        except duckdb.Error:
            logger.exception("conn.interrupt() raised while cancelling query")
        thread.join(timeout=1.0)
        raise SQLTimeoutError(
            f"query exceeded timeout of {timeout_s:.1f}s"
        )

    if result.error is not None:
        # Re-raise the worker's exception in the caller's context.
        raise result.error

    assert result.df is not None  # for type-checkers; guaranteed if no error.
    return result.df


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


def _apply_row_limit(sql: str, max_rows: int) -> str:
    """Return *sql* with a ``LIMIT`` clause appended if one is absent.

    The check is intentionally simple — it merely looks for the literal
    word ``LIMIT`` in the (already comment-stripped) text. If present we
    leave the query alone and rely on pandas-side slicing for truncation.
    """
    cleaned = _strip_string_literals(_strip_comments(sql))
    if _LIMIT_RE.search(cleaned):
        return sql
    trimmed = sql.rstrip().rstrip(";")
    return f"{trimmed} LIMIT {max_rows + 1}"


def run_query(
    sql: str,
    df: pd.DataFrame | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> pd.DataFrame:
    """Validate, execute, and return the result of *sql*.

    If *df* is supplied it is registered as the view ``df`` before
    execution. The query is wrapped in a worker thread so it can be
    interrupted on timeout. The result is truncated to *max_rows*:

    * If the query had no existing ``LIMIT`` clause, we append
      ``LIMIT max_rows + 1`` (so we can detect truncation) and slice.
    * Otherwise we let the query run as written and slice the result.

    Args:
        sql: The SQL text. Will be validated by :func:`validate_sql`.
        df:  Optional DataFrame to register as ``df``.
        max_rows: Maximum number of rows to return.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        A :class:`pandas.DataFrame` containing at most *max_rows* rows.

    Raises:
        ValueError: If *sql* is empty.
        UnsafeSQLError: If *sql* fails validation.
        SQLTimeoutError: If execution exceeds *timeout_s*.
        duckdb.Error: If DuckDB raises during planning or execution.
    """
    validate_sql(sql)

    if df is not None:
        register_dataframe("df", df)

    conn = get_connection()
    effective_sql = _apply_row_limit(sql, max_rows)

    started = time.perf_counter()
    result_df = _run_with_timeout(conn, effective_sql, timeout_s)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Truncate in pandas as a backstop — covers both the LIMIT-already-present
    # path and any edge case where DuckDB returned more rows than expected.
    if len(result_df) > max_rows:
        result_df = result_df.iloc[:max_rows].copy()

    logger.info(
        "SQL executed in %d ms, returned %d rows", elapsed_ms, len(result_df)
    )
    return result_df


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def record_history(state: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append *entry* to ``state['sql_history']`` and trim to the cap.

    The history list is created on demand. After append we keep at most
    :data:`HISTORY_LIMIT` entries (the most recent ones).
    """
    history = state.setdefault("sql_history", [])
    history.append(entry)
    if len(history) > HISTORY_LIMIT:
        # Drop the oldest entries — list is chronological (append-only).
        del history[: len(history) - HISTORY_LIMIT]


def get_history(state: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    """Return up to *limit* history entries, most recent first.

    Args:
        state: A mapping (typically ``st.session_state``) containing a
            ``'sql_history'`` list.
        limit: Maximum number of entries to return.

    Returns:
        A list of history dicts with keys ``sql``, ``ran_at``, ``rowcount``,
        ``duration_ms``, ``error``.
    """
    history = state.get("sql_history", []) or []
    # Reverse copy so we don't mutate the caller's list.
    reversed_history = list(reversed(history))
    if limit is not None and limit >= 0:
        return reversed_history[:limit]
    return reversed_history


def clear_history(state: dict[str, Any]) -> None:
    """Empty ``state['sql_history']``."""
    state["sql_history"] = []


# ---------------------------------------------------------------------------
# Convenience: run + record in one shot
# ---------------------------------------------------------------------------


def run_and_record(
    state: dict[str, Any],
    sql: str,
    df: pd.DataFrame | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> pd.DataFrame:
    """Run *sql* via :func:`run_query` and append a history entry.

    On success the entry records ``rowcount`` and ``duration_ms`` with
    ``error=None``. On failure the entry records ``error=str(exc)`` (with
    a special ``'timeout'`` sentinel for :class:`SQLTimeoutError`) and the
    exception is re-raised so the caller can render it.
    """
    started = time.perf_counter()
    ran_at = datetime.now(timezone.utc).isoformat()
    try:
        result = run_query(sql, df=df, max_rows=max_rows, timeout_s=timeout_s)
    except SQLTimeoutError:
        duration_ms = int((time.perf_counter() - started) * 1000)
        record_history(
            state,
            {
                "sql": sql,
                "ran_at": ran_at,
                "rowcount": 0,
                "duration_ms": duration_ms,
                "error": "timeout",
            },
        )
        raise
    except duckdb.Error as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        record_history(
            state,
            {
                "sql": sql,
                "ran_at": ran_at,
                "rowcount": 0,
                "duration_ms": duration_ms,
                "error": str(exc),
            },
        )
        raise
    except (UnsafeSQLError, ValueError) as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        record_history(
            state,
            {
                "sql": sql,
                "ran_at": ran_at,
                "rowcount": 0,
                "duration_ms": duration_ms,
                "error": str(exc),
            },
        )
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    record_history(
        state,
        {
            "sql": sql,
            "ran_at": ran_at,
            "rowcount": int(len(result)),
            "duration_ms": duration_ms,
            "error": None,
        },
    )
    return result
