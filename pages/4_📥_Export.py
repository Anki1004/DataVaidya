import streamlit as st
import pandas as pd
import io
from datetime import datetime

from utils.constants import APP_NAME, APP_VERSION
from ui.theme import inject_css
from ui.components import gradient_header, metric_card, info_pill, empty_state
from core.exports import (
    export_csv,
    export_excel,
    export_parquet,
    export_profile_html,
    export_pdf_summary,
    export_python_script,
)

st.set_page_config(page_title=f"Export · {APP_NAME}", page_icon="📥", layout="wide")
from app import init_session_state
init_session_state()
inject_css()

gradient_header("📥 Export", level=1)
st.caption("Download your cleaned dataset and a reproducible Python script.")

df = st.session_state.get("df")
if df is None:
    empty_state(
        "No data loaded",
        "Upload a dataset on the Profile page first.",
        cta="Go to Profile",
        cta_page="pages/1_📊_Profile.py",
    )
    st.stop()

# --- Summary cards ---
meta = st.session_state.get("df_meta", {})
log_count = len(st.session_state.get("cleaning_log", []))
mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

scol1, scol2, scol3, scol4 = st.columns(4)
scol1.metric("Rows", f"{len(df):,}")
scol2.metric("Columns", f"{len(df.columns)}")
scol3.metric("Memory", f"{mem_mb:.2f} MB")
scol4.metric("Cleaning steps", log_count)

st.divider()


# --- Filename helper ---
def _safe_stem(filename: str | None) -> str:
    if not filename:
        return "datavaidya_export"
    stem = filename.rsplit(".", 1)[0]
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)[:50] or "datavaidya_export"


stem = _safe_stem(meta.get("filename"))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Cleaned data downloads ---
st.subheader("Cleaned data")
row1 = st.columns(3, gap="medium")

with row1[0]:
    st.markdown(
        '<div class="dv-card"><h4>📄 CSV</h4><p>Universal, opens in Excel/Sheets. UTF-8 with BOM.</p></div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download .csv",
        data=export_csv(df),
        file_name=f"{stem}_cleaned_{ts}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_csv",
    )

with row1[1]:
    st.markdown(
        '<div class="dv-card"><h4>📊 Excel</h4><p>.xlsx with auto-fit column widths.</p></div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download .xlsx",
        data=export_excel(df),
        file_name=f"{stem}_cleaned_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_xlsx",
    )

with row1[2]:
    st.markdown(
        '<div class="dv-card"><h4>🗜️ Parquet</h4><p>Snappy-compressed columnar — smaller, faster.</p></div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download .parquet",
        data=export_parquet(df),
        file_name=f"{stem}_cleaned_{ts}.parquet",
        mime="application/octet-stream",
        use_container_width=True,
        key="dl_parquet",
    )

# --- Reports & code ---
st.subheader("Reports & code")
row2 = st.columns(3, gap="medium")

with row2[0]:
    st.markdown(
        '<div class="dv-card"><h4>🧾 Profile report (HTML)</h4><p>Full ydata-profiling deep dive.</p></div>',
        unsafe_allow_html=True,
    )
    if st.button("Generate HTML report", key="gen_html", use_container_width=True):
        with st.spinner("Building profile report (this can take 10-60s for big files)..."):
            html_bytes = export_profile_html(df, title=f"{stem} — Profile")
        st.session_state["_html_report"] = html_bytes
    if st.session_state.get("_html_report"):
        st.download_button(
            "Download .html",
            data=st.session_state["_html_report"],
            file_name=f"{stem}_profile_{ts}.html",
            mime="text/html",
            use_container_width=True,
            key="dl_html",
        )

with row2[1]:
    st.markdown(
        '<div class="dv-card"><h4>📑 Executive PDF</h4><p>AI summary + key metrics, rendered as A4 PDF.</p></div>',
        unsafe_allow_html=True,
    )
    ai_summary = st.session_state.get("ai_summary", "")
    if ai_summary:
        pdf_bytes = export_pdf_summary(ai_summary, title=f"{stem} — Executive Summary")
        st.download_button(
            "Download .pdf",
            data=pdf_bytes,
            file_name=f"{stem}_summary_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf",
        )
    else:
        st.caption("Generate an AI summary first on the AI Insights page.")
        st.button("Download .pdf", disabled=True, use_container_width=True, key="dl_pdf_dis")

with row2[2]:
    st.markdown(
        '<div class="dv-card"><h4>🐍 Python script</h4><p>Reproducible cleaning pipeline — run locally on the full file.</p></div>',
        unsafe_allow_html=True,
    )
    log = st.session_state.get("cleaning_log", [])
    py_bytes = export_python_script(
        log,
        source_filename=meta.get("filename", "input.csv"),
        app_version=APP_VERSION,
    )
    st.download_button(
        "Download .py",
        data=py_bytes,
        file_name=f"{stem}_clean_{ts}.py",
        mime="text/x-python",
        use_container_width=True,
        key="dl_py",
    )

# --- Cleaning log expander ---
st.divider()
if st.session_state.get("cleaning_log"):
    with st.expander(f"📜 Cleaning operations applied ({log_count})"):
        for i, (op, params) in enumerate(st.session_state["cleaning_log"], 1):
            st.code(
                f"{i}. {op}({', '.join(f'{k}={v!r}' for k, v in params.items())})",
                language="python",
            )
else:
    st.info("No cleaning operations applied yet — the Python script will be a no-op load/save.")
