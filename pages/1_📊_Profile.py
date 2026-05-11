"""Profile page: dataset ingestion and quality diagnosis."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.constants import (
    APP_NAME,
    APP_EMOJI,
    MAX_FILE_MB,
    WARN_FILE_MB,
    SUPPORTED_EXTENSIONS,
)

st.set_page_config(
    page_title=f"Profile · {APP_NAME}",
    page_icon="📊",
    layout="wide",
)

from app import init_session_state  # noqa: E402  (must come after set_page_config)

init_session_state()

from utils.memory import sample_for_viz, df_memory_summary  # noqa: E402
from utils.validation import (  # noqa: E402
    validate_extension,
    validate_file_size,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from ui.theme import inject_css, PLOTLY_CONFIG  # noqa: E402
from ui.components import (  # noqa: E402
    gradient_header,
    metric_card,
    health_gauge,
    health_breakdown_grid,
    info_pill,
    empty_state,
    before_after_metrics,
)
from ui.charts import (  # noqa: E402
    missing_bar,
    correlation_heatmap,
    numeric_distribution,
    categorical_top10,
    box_plot_outliers,
    memory_footprint_bar,
)
from core.ingestion import load_uploaded, load_demo, DEMO_DATASETS, preview_schema  # noqa: E402
from core.profiling import (  # noqa: E402
    compute_health_score,
    missing_summary,
    duplicate_summary,
    outlier_summary,
    correlation_matrix,
    top_correlations,
    cardinality_summary,
    memory_footprint,
    distribution_summary,
)
from core.pii import detect_pii, get_pii_summary  # noqa: E402

inject_css()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(df: pd.DataFrame) -> tuple[Any, ...]:
    """Build a cheap stable fingerprint for caching keyed on a DataFrame.

    Combines shape, column names, and a CSV-serialised head sample so that
    Streamlit's hasher can key cached results without hashing the entire frame.
    """
    try:
        head_csv = df.head(50).to_csv(index=False)
    except Exception:
        head_csv = ""
    return (df.shape, tuple(df.columns), head_csv)


@st.cache_data(show_spinner="Reading file...", max_entries=3, ttl=600)
def _cached_load(file_bytes: bytes, filename: str, sample: bool):
    """Cache wrapper around :func:`load_uploaded`."""
    return load_uploaded(file_bytes, filename, sample=sample)


@st.cache_data(show_spinner="Loading demo...", max_entries=3, ttl=600)
def _cached_demo(name: str):
    """Cache wrapper around :func:`load_demo`."""
    return load_demo(name)


@st.cache_data(show_spinner="Scoring dataset health...", max_entries=4, ttl=600)
def _cached_health(fp: tuple[Any, ...], df: pd.DataFrame) -> dict:
    """Return cached health-score payload keyed on a DataFrame fingerprint."""
    return compute_health_score(df)


@st.cache_data(show_spinner="Scanning missing values...", max_entries=4, ttl=600)
def _cached_missing(fp: tuple[Any, ...], df: pd.DataFrame) -> pd.DataFrame:
    """Cached missing-value summary."""
    return missing_summary(df)


@st.cache_data(show_spinner="Looking for duplicates...", max_entries=4, ttl=600)
def _cached_duplicates(fp: tuple[Any, ...], df: pd.DataFrame) -> dict:
    """Cached duplicate-row summary."""
    return duplicate_summary(df)


@st.cache_data(show_spinner="Detecting outliers...", max_entries=4, ttl=600)
def _cached_outliers(fp: tuple[Any, ...], df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Cached outlier summary for the given detection method."""
    return outlier_summary(df, method=method)


@st.cache_data(show_spinner="Computing correlations...", max_entries=4, ttl=600)
def _cached_corr(fp: tuple[Any, ...], df: pd.DataFrame) -> pd.DataFrame:
    """Cached correlation matrix."""
    return correlation_matrix(df)


@st.cache_data(show_spinner="Ranking correlations...", max_entries=4, ttl=600)
def _cached_top_corr(fp: tuple[Any, ...], df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Cached top-N correlation pairs."""
    return top_correlations(df, n)


@st.cache_data(show_spinner="Counting unique values...", max_entries=4, ttl=600)
def _cached_cardinality(fp: tuple[Any, ...], df: pd.DataFrame) -> pd.DataFrame:
    """Cached cardinality summary."""
    return cardinality_summary(df)


@st.cache_data(show_spinner="Measuring memory footprint...", max_entries=4, ttl=600)
def _cached_memory(fp: tuple[Any, ...], df: pd.DataFrame) -> pd.DataFrame:
    """Cached per-column memory footprint."""
    return memory_footprint(df)


@st.cache_data(show_spinner="Scanning for PII...", max_entries=4, ttl=600)
def _cached_pii(fp: tuple[Any, ...], df: pd.DataFrame) -> dict:
    """Cached PII detection summary."""
    return get_pii_summary(df)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

gradient_header("📊 Profile", level=1)
st.caption("Upload a dataset or pick a demo to see its quality diagnosis.")


# ---------------------------------------------------------------------------
# Upload row
# ---------------------------------------------------------------------------

col_a, col_b = st.columns([2, 1])
with col_a:
    uploaded = st.file_uploader(
        "Drag and drop a file",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
        help=f"Up to {MAX_FILE_MB}MB. CSV, TSV, Excel, Parquet, JSON.",
    )
with col_b:
    demo_options = [""] + list(DEMO_DATASETS.keys())
    preselect = st.session_state.get("demo_preselect")
    default_idx = demo_options.index(preselect) if preselect in demo_options else 0
    demo_name = st.selectbox(
        "…or pick a demo",
        options=demo_options,
        index=default_idx,
    )


# ---------------------------------------------------------------------------
# Load logic
# ---------------------------------------------------------------------------

df: pd.DataFrame | None = None
report: dict | None = None

if uploaded is not None:
    try:
        validate_extension(uploaded.name)
        size_mb = uploaded.size / (1024 * 1024)
        ok, warning = validate_file_size(uploaded.size, uploaded.name)
        if warning:
            st.warning(warning)
        sample = size_mb > WARN_FILE_MB
        if sample:
            st.info(
                f"File > {WARN_FILE_MB}MB — loading first 10,000 rows for fast profiling."
            )
        df, report = _cached_load(uploaded.getvalue(), uploaded.name, sample)
        st.session_state["df_raw"] = df
        st.session_state["df"] = df.copy()
        st.session_state["df_meta"] = {
            "filename": uploaded.name,
            "size_bytes": uploaded.size,
            "rows": len(df),
            "cols": len(df.columns),
            "uploaded_at": pd.Timestamp.now().isoformat(),
            "source": "upload",
        }
    except (FileTooLargeError, UnsupportedFileTypeError) as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # pragma: no cover - surface load errors to user
        st.error(f"Could not read file: {e}")
        st.stop()
elif demo_name:
    df = _cached_demo(demo_name)
    if df is None or df.empty:
        st.warning(
            f"Demo dataset '{demo_name}' not found. "
            "Run `python make_demo.py` to generate samples."
        )
        st.stop()
    st.session_state["df_raw"] = df
    st.session_state["df"] = df.copy()
    st.session_state["df_meta"] = {
        "filename": demo_name,
        "rows": len(df),
        "cols": len(df.columns),
        "source": "demo",
        "uploaded_at": pd.Timestamp.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

df = st.session_state.get("df")
if df is None:
    empty_state("No data yet", "Upload a file above or pick a demo to begin.")
    st.stop()

fp = _fingerprint(df)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_schema,
    tab_health,
    tab_missing,
    tab_dupes,
    tab_outliers,
    tab_corr,
    tab_dist,
    tab_card,
    tab_pii,
) = st.tabs(
    [
        "Schema",
        "Health",
        "Missing",
        "Duplicates",
        "Outliers",
        "Correlation",
        "Distributions",
        "Cardinality",
        "PII",
    ]
)


# ---- Schema --------------------------------------------------------------
with tab_schema:
    mem_info = df_memory_summary(df)
    mem_mb = mem_info.get("total_mb", df.memory_usage(deep=True).sum() / (1024 * 1024))
    dtypes_unique = df.dtypes.astype(str).nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{len(df):,}")
    with c2:
        metric_card("Columns", f"{len(df.columns):,}")
    with c3:
        metric_card("Memory", f"{mem_mb:,.2f} MB")
    with c4:
        metric_card("Dtypes", f"{dtypes_unique}")

    st.markdown("#### Schema")
    st.dataframe(
        preview_schema(df),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Sample first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)


# ---- Health --------------------------------------------------------------
with tab_health:
    health = _cached_health(fp, df)
    st.session_state["profile_report"] = health
    st.session_state["health_score"] = health.get("score")
    st.session_state["health_breakdown"] = health.get("deductions", [])

    score = health.get("score", 0)
    zone = health.get("zone") or (
        "Healthy" if score >= 80 else "Needs attention" if score >= 60 else "Critical"
    )
    deductions = health.get("deductions", []) or []
    biggest = None
    if deductions:
        try:
            biggest = max(
                deductions,
                key=lambda d: d.get("points", d.get("weight", 0)) if isinstance(d, dict) else 0,
            )
        except Exception:
            biggest = deductions[0]

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            health_gauge(score),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with right:
        st.markdown(f"### Status: **{zone}**")
        st.markdown(
            f"Your dataset scored **{score}/100** across completeness, uniqueness, "
            "consistency, distribution shape, and PII safety."
        )
        if biggest and isinstance(biggest, dict):
            label = biggest.get("label") or biggest.get("name") or biggest.get("category", "issue")
            points = biggest.get("points", biggest.get("weight", 0))
            st.markdown(
                f"**Biggest drag:** {label} — costing roughly {points:.0f} points."
            )
        else:
            st.success("No major issues detected. Nice dataset.")

    st.markdown("#### Breakdown")
    health_breakdown_grid(deductions)

    with st.expander("What this score means"):
        st.markdown(
            """
            The **Health Score** is a weighted blend of five dimensions:

            - **Completeness (30%)** — share of non-missing cells.
            - **Uniqueness (20%)** — penalty for duplicate rows.
            - **Consistency (20%)** — dtype coherence and value-range sanity.
            - **Distribution (15%)** — outlier load on numeric columns.
            - **PII safety (15%)** — penalty when sensitive fields are exposed.

            Scores above **80** are healthy, **60–79** need attention, and below **60**
            are critical. Fix the biggest drag first — the **Clean** page can resolve
            most issues automatically.
            """
        )


# ---- Missing -------------------------------------------------------------
with tab_missing:
    miss_df = _cached_missing(fp, df)
    total_cells = int(df.shape[0]) * int(df.shape[1]) if df.size else 0
    total_missing = int(df.isna().sum().sum())
    overall_pct = (total_missing / total_cells * 100) if total_cells else 0.0

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Missing cells", f"{total_missing:,}")
    with c2:
        metric_card("Overall missing %", f"{overall_pct:.2f}%")

    st.plotly_chart(
        missing_bar(df),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown("#### Per-column summary")
    st.dataframe(miss_df, use_container_width=True, hide_index=True)


# ---- Duplicates ----------------------------------------------------------
with tab_dupes:
    dup_info = _cached_duplicates(fp, df)
    dup_count = int(dup_info.get("count", 0))
    dup_pct = float(dup_info.get("pct", 0.0))

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Duplicate rows", f"{dup_count:,}")
    with c2:
        metric_card("Share of dataset", f"{dup_pct:.2f}%")

    sample = dup_info.get("sample")
    if dup_count and isinstance(sample, pd.DataFrame) and not sample.empty:
        st.markdown("#### Example duplicate rows")
        st.dataframe(sample, use_container_width=True)
    elif dup_count:
        st.info("Duplicates were detected but no sample is available.")
    else:
        st.success("No duplicate rows found.")


# ---- Outliers ------------------------------------------------------------
@st.fragment
def _outliers_fragment(df: pd.DataFrame, fp: tuple[Any, ...]) -> None:
    """Outlier explorer fragment so the method/column picker doesn't rerun the page."""
    method = st.radio(
        "Detection method",
        options=["IQR", "Z-score"],
        index=0,
        horizontal=True,
        key="profile_outlier_method",
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for outlier detection.")
        return

    picked = st.multiselect(
        "Columns to chart",
        options=numeric_cols,
        default=numeric_cols[: min(4, len(numeric_cols))],
        key="profile_outlier_cols",
    )

    if picked:
        cols_per_row = 2
        for i in range(0, len(picked), cols_per_row):
            row = st.columns(cols_per_row)
            for slot, col_name in zip(row, picked[i : i + cols_per_row]):
                with slot:
                    st.plotly_chart(
                        box_plot_outliers(df, col_name),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

    st.markdown("#### Outlier summary")
    summary = _cached_outliers(fp, df, method)
    st.dataframe(summary, use_container_width=True, hide_index=True)


with tab_outliers:
    _outliers_fragment(df, fp)


# ---- Correlation ---------------------------------------------------------
with tab_corr:
    st.plotly_chart(
        correlation_heatmap(df),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown("#### Top correlations")
    top_corr = _cached_top_corr(fp, df, 5)
    if isinstance(top_corr, pd.DataFrame) and not top_corr.empty:
        st.dataframe(top_corr, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough numeric columns to rank correlations.")


# ---- Distributions -------------------------------------------------------
@st.fragment
def _distributions_fragment(df: pd.DataFrame) -> None:
    """Distribution explorer fragment — switches chart by dtype."""
    cols = list(df.columns)
    if not cols:
        st.info("No columns to explore.")
        return

    chosen = st.selectbox(
        "Column",
        options=cols,
        key="profile_dist_col",
    )
    if not chosen:
        return

    series = df[chosen]
    if pd.api.types.is_numeric_dtype(series):
        st.plotly_chart(
            numeric_distribution(df, chosen),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    else:
        st.plotly_chart(
            categorical_top10(df, chosen),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


with tab_dist:
    _distributions_fragment(df)


# ---- Cardinality ---------------------------------------------------------
with tab_card:
    card_df = _cached_cardinality(fp, df)
    st.dataframe(card_df, use_container_width=True, hide_index=True)


# ---- PII -----------------------------------------------------------------
with tab_pii:
    pii_summary = _cached_pii(fp, df)
    detections = pii_summary.get("detections", {}) if isinstance(pii_summary, dict) else {}
    category_counts = (
        pii_summary.get("category_counts", {}) if isinstance(pii_summary, dict) else {}
    )

    if detections:
        cat_str = ", ".join(f"{k}: {v}" for k, v in category_counts.items()) or "—"
        st.warning(
            f"Detected potential PII in {len(detections)} column(s). "
            f"Categories — {cat_str}."
        )

        rows = []
        for col_name, cats in detections.items():
            if isinstance(cats, (list, tuple, set)):
                rows.append({"column": col_name, "categories": ", ".join(map(str, cats))})
            elif isinstance(cats, dict):
                rows.append(
                    {
                        "column": col_name,
                        "categories": ", ".join(map(str, cats.keys())),
                    }
                )
            else:
                rows.append({"column": col_name, "categories": str(cats)})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No likely PII detected by the heuristic scanner.")

    st.caption(
        "DPDP Act 2023 compliance — masking available on the Clean page."
    )
