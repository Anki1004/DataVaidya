"""DataVaidya — main Streamlit entrypoint.

A thin routing entrypoint. The four main features live under ``pages/`` and are
auto-discovered by Streamlit's native multipage system.

This module is responsible for:
    * Setting page config (must be the first Streamlit call).
    * Injecting global CSS.
    * Initialising the session-state schema (idempotent across reruns).
    * Rendering the sidebar (logo, page links, memory monitor, reset button).
    * Rendering the hero, three quickstart cards, and a recent-activity card.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from core.ingestion import DEMO_DATASETS
from ui.components import empty_state, hero
from ui.onboarding import maybe_show_onboarding
from ui.theme import inject_css
from utils.constants import (
    APP_EMOJI,
    APP_NAME,
    APP_VERSION,
    LLM_SESSION_CAP,
    TAGLINE,
)
from utils.memory import force_gc, render_memory_widget

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title=f"{APP_NAME} — Data Quality Engine",
    page_icon=APP_EMOJI,
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session-state schema
# ---------------------------------------------------------------------------
DEFAULT_STATE: dict[str, Any] = {
    "onboarded_v1": False,
    "df_raw": None,            # pd.DataFrame | None — original upload
    "df": None,                # pd.DataFrame | None — current working df
    "df_meta": {},             # {filename, size_bytes, rows, cols, uploaded_at, source}
    "schema": None,
    "profile_report": None,    # HealthReport TypedDict
    "health_score": None,
    "health_breakdown": None,  # dict — deductions
    "undo_stack": [],          # list[tuple[str, pd.DataFrame, str]]
    "redo_stack": [],
    "cleaning_log": [],        # list[tuple[str, dict]] — (op_name, kwargs)
    "snapshots": {},           # dict[str, pd.DataFrame]
    "ai_summary": "",
    "ai_call_count": 0,
    "ai_feedback": {"up": 0, "down": 0},
    "ai_last_error": None,
    "sql_history": [],
    "history": [],             # audit trail [{page, action, ts}]
    "demo_preselect": None,    # str | None — set when user clicks "Try demo dataset"
}


def init_session_state() -> None:
    """Apply :data:`DEFAULT_STATE` via ``setdefault`` for every key.

    Idempotent — safe to call at the top of every page on every rerun.
    Does not overwrite existing values.
    """
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
@st.fragment(run_every="10s")
def _memory_fragment() -> None:
    """Re-render the memory widget every 10s without rerunning the full app."""
    render_memory_widget()


def _render_sidebar() -> None:
    """Render the global sidebar: logo, page links, memory monitor, reset."""
    with st.sidebar:
        st.markdown(f"{APP_EMOJI} **{APP_NAME}**")
        st.caption(f"v{APP_VERSION}")

        st.markdown("#### Navigation")

        # Page links — wrap each in try/except so a missing/renamed page does
        # not crash the sidebar. ``st.page_link`` exists on Streamlit >= 1.30.
        page_files = [
            ("pages/1_📊_Profile.py", "Profile", "📊"),
            ("pages/2_🧹_Clean.py", "Clean", "🧹"),
            ("pages/3_🤖_AI_Insights.py", "AI Insights", "🤖"),
            ("pages/4_📥_Export.py", "Export", "📥"),
        ]
        page_link = getattr(st, "page_link", None)
        if page_link is not None:
            for path, label, icon in page_files:
                try:
                    page_link(path, label=label, icon=icon)
                except (FileNotFoundError, ValueError, st.errors.StreamlitAPIException):
                    # Page not yet created or path mismatch — skip silently.
                    continue
        else:
            for _path, label, icon in page_files:
                st.markdown(f"- {icon} {label}")

        st.divider()

        # Memory monitor (auto-refreshes every 10s)
        _memory_fragment()

        st.divider()

        if st.button("Reset session", use_container_width=True, key="sidebar_reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            force_gc()
            st.rerun()

        st.caption("Built in 🇮🇳")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
def _render_hero() -> None:
    """Render the hero block: gradient-mesh background, typewriter title, tagline, version pill."""
    hero(APP_NAME, tagline=TAGLINE, version=APP_VERSION)


def _render_quickstart() -> None:
    """Render the three quickstart cards: upload / demo / docs."""
    cols = st.columns(3, gap="medium")

    with cols[0]:
        st.markdown(
            '<div class="dv-card">'
            "<h3>📤 Upload your data</h3>"
            "<p>CSV, Excel, Parquet, JSON, TSV — up to 50MB.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Start uploading",
            type="primary",
            use_container_width=True,
            key="cta_upload",
        ):
            try:
                st.switch_page("pages/1_📊_Profile.py")
            except st.errors.StreamlitAPIException:
                st.warning("Profile page is not available yet.")

    with cols[1]:
        st.markdown(
            '<div class="dv-card">'
            "<h3>🧪 Try a demo dataset</h3>"
            "<p>Indian Census, Mumbai Real Estate, Retail Transactions, and more.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        demo_choice = st.selectbox(
            "Choose a dataset",
            options=[""] + list(DEMO_DATASETS.keys()),
            key="cta_demo",
        )
        if demo_choice:
            st.session_state["demo_preselect"] = demo_choice
            try:
                st.switch_page("pages/1_📊_Profile.py")
            except st.errors.StreamlitAPIException:
                st.warning("Profile page is not available yet.")

    with cols[2]:
        st.markdown(
            '<div class="dv-card">'
            "<h3>📚 Read the docs</h3>"
            "<p>Feature tour, FAQs, and deployment guide.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.link_button(
            "Open README",
            "https://github.com/your-handle/datavaidya",
            use_container_width=True,
        )


def _format_ts(ts: Any) -> str:
    """Best-effort formatting of an audit-trail timestamp."""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return str(ts)
    return str(ts) if ts is not None else ""


def _render_recent_activity() -> None:
    """Show the last few audit-trail entries, or an empty-state caption."""
    history = st.session_state.get("history") or []
    if not history:
        st.caption("Your activity will appear here once you start using DataVaidya.")
        return

    st.subheader("Recent activity")
    recent = list(history)[-5:][::-1]
    rows = [
        {
            "When": _format_ts(entry.get("ts")),
            "Page": entry.get("page", ""),
            "Action": entry.get("action", ""),
        }
        for entry in recent
        if isinstance(entry, dict)
    ]
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        empty_state(
            title="No recent activity",
            description="Upload a dataset to get started.",
            icon="📭",
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Top-level entrypoint executed by Streamlit on every script run."""
    inject_css()
    init_session_state()
    _render_sidebar()
    maybe_show_onboarding()

    _render_hero()
    st.markdown("---")
    _render_quickstart()
    st.markdown("---")
    _render_recent_activity()


main()
