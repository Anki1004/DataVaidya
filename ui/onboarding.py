"""First-run onboarding for DataVaidya.

Displays a three-step walkthrough the first time a user lands on the app.
Uses Streamlit's native ``@st.dialog`` when available (Streamlit >= 1.35) and
gracefully falls back to a sidebar expander on older versions. The dismissal
flag is versioned (``onboarded_v1``) so a future redesign can re-prompt every
user by bumping the constant to ``onboarded_v2``.
"""

from __future__ import annotations

import streamlit as st

from utils.constants import APP_EMOJI, APP_NAME, TAGLINE


# Bump this (e.g. to "onboarded_v2") after a major redesign to re-trigger the
# walkthrough for every existing user.
ONBOARDED_KEY: str = "onboarded_v1"


# ---------------------------------------------------------------------------
# Walkthrough content
# ---------------------------------------------------------------------------
_STEPS: tuple[tuple[str, str, str], ...] = (
    (
        "1",
        "Upload your data",
        "Drag in any CSV, Excel, or Parquet file (up to 50 MB). "
        "DataVaidya auto-detects the schema and gets to work.",
    ),
    (
        "2",
        "Get a health diagnosis",
        "A single health score plus a full breakdown of every quality issue — "
        "missing values, outliers, type mismatches, duplicates, and more.",
    ),
    (
        "3",
        "Clean and export",
        "Toggle the fixes you want, preview the result, then download the "
        "cleaned data along with a reproducible Python script.",
    ),
)


def _render_step_cards() -> None:
    """Render the three walkthrough cards using the dv-card HTML pattern."""
    for number, title, body in _STEPS:
        st.markdown(
            f"""
            <div class="dv-card" style="margin-bottom: 12px;">
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div class="dv-card__step" style="
                        flex: 0 0 auto;
                        width: 32px; height: 32px;
                        border-radius: 999px;
                        display: flex; align-items: center; justify-content: center;
                        background: rgba(124, 58, 237, 0.15);
                        color: #7c3aed;
                        font-weight: 700;
                    ">{number}</div>
                    <div style="flex: 1 1 auto;">
                        <div class="dv-card__title" style="
                            font-weight: 600;
                            font-size: 1rem;
                            margin-bottom: 4px;
                        ">{title}</div>
                        <div class="dv-card__body" style="
                            font-size: 0.9rem;
                            line-height: 1.45;
                            opacity: 0.85;
                        ">{body}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _dismiss() -> None:
    """Mark onboarding complete and rerun so the dialog/expander goes away."""
    st.session_state[ONBOARDED_KEY] = True
    # ``st.rerun`` exists on modern Streamlit; older versions had
    # ``experimental_rerun``. Be defensive.
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


# ---------------------------------------------------------------------------
# Modern path: native dialog (Streamlit >= 1.35)
# ---------------------------------------------------------------------------
if hasattr(st, "dialog"):

    @st.dialog("Welcome to DataVaidya")  # type: ignore[misc]
    def _onboarding_dialog() -> None:
        """Render the welcome dialog using Streamlit's native modal."""
        st.markdown(
            f"### {APP_EMOJI} {APP_NAME}\n"
            f"<div style='opacity: 0.75; margin-bottom: 16px;'>{TAGLINE}</div>",
            unsafe_allow_html=True,
        )
        _render_step_cards()
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Got it", type="primary", use_container_width=True, key="dv_onboard_got_it"):
            _dismiss()

else:  # pragma: no cover - only hit on older Streamlit

    def _onboarding_dialog() -> None:
        """Sidebar-expander fallback for Streamlit versions without ``st.dialog``."""
        with st.sidebar.expander(f"Welcome to {APP_NAME}", expanded=True):
            st.markdown(
                f"### {APP_EMOJI} {APP_NAME}\n"
                f"<div style='opacity: 0.75; margin-bottom: 12px;'>{TAGLINE}</div>",
                unsafe_allow_html=True,
            )
            _render_step_cards()
            if st.button("Got it", type="primary", use_container_width=True, key="dv_onboard_got_it"):
                _dismiss()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def maybe_show_onboarding() -> None:
    """Open the welcome walkthrough if the user has not dismissed it yet.

    Reads ``st.session_state[ONBOARDED_KEY]`` — when falsy (the default for a
    fresh session) the dialog (or sidebar fallback) is rendered. Clicking
    "Got it" flips the flag to ``True`` and reruns the app, so the dialog
    will not reappear for the remainder of the session.
    """
    if st.session_state.get(ONBOARDED_KEY):
        return
    _onboarding_dialog()


__all__ = ["ONBOARDED_KEY", "maybe_show_onboarding"]
