import streamlit as st
import pandas as pd

from utils.constants import APP_NAME, LLM_SESSION_CAP, LLM_FALLBACK_MESSAGE
from ui.theme import inject_css
from ui.components import gradient_header, info_pill, empty_state, metric_card
from core.llm import stream_summary, check_call_budget
from core.profiling import compute_health_score
from core.pii import detect_pii

st.set_page_config(page_title=f"AI Insights · {APP_NAME}", page_icon="🤖", layout="wide")

from app import init_session_state
init_session_state()
inject_css()

gradient_header("🤖 AI Insights", level=1)
st.caption(
    f"Groq-powered executive summary tailored for Indian business context. "
    f"{LLM_SESSION_CAP} calls per session."
)

df = st.session_state.get("df")
profile_report = st.session_state.get("profile_report")

if df is None:
    empty_state(
        "No data loaded",
        "Upload a dataset on the Profile page first.",
        cta="Go to Profile",
        cta_page="pages/1_📊_Profile.py",
    )
    st.stop()

if profile_report is None:
    # Compute on the fly if user skipped profile inspection
    profile_report = compute_health_score(df)
    st.session_state["profile_report"] = profile_report

# -----------------------------------------------------------------------------
# Header metrics row
# -----------------------------------------------------------------------------
hcol1, hcol2, hcol3 = st.columns([1, 1, 2])
with hcol1:
    calls_used = st.session_state.get("ai_call_count", 0)
    st.metric("Calls used", f"{calls_used} / {LLM_SESSION_CAP}")
with hcol2:
    feedback = st.session_state.get("ai_feedback", {"up": 0, "down": 0})
    st.metric("Helpful 👍 / 👎", f"{feedback['up']} / {feedback['down']}")
with hcol3:
    st.caption(
        "⚠️ PII (emails, phone, Aadhaar, PAN) is scrubbed before sending to LLM providers."
    )


# -----------------------------------------------------------------------------
# Generate button + streaming summary panel (as a fragment)
# -----------------------------------------------------------------------------
def _render_summary_panel():
    """Fragment that streams the LLM response without re-running the whole page."""
    if st.button("Generate executive summary", type="primary", key="ai_generate"):
        allowed, refusal = check_call_budget()
        if not allowed:
            st.warning(refusal)
            return
        try:
            placeholder = st.empty()
            collected = []
            with st.spinner("Asking the model..."):
                for chunk in stream_summary(
                    df,
                    dict(profile_report),
                    dataset_name=st.session_state.get("df_meta", {}).get(
                        "filename", "dataset"
                    ),
                ):
                    collected.append(chunk)
                    placeholder.markdown("".join(collected))
            st.session_state["ai_summary"] = "".join(collected)
        except Exception as e:
            st.error(f"AI generation failed: {e}")
            st.session_state["ai_last_error"] = str(e)

    # Show last summary if present
    if st.session_state.get("ai_summary"):
        st.markdown("---")
        st.markdown(st.session_state["ai_summary"])


# Apply the fragment decorator at call-time so it works as a function reference.
_render_summary_panel = st.fragment(_render_summary_panel)
_render_summary_panel()


# -----------------------------------------------------------------------------
# Action row (only when a summary exists)
# -----------------------------------------------------------------------------
if st.session_state.get("ai_summary"):
    st.markdown("---")
    a, b, c, d = st.columns([1, 1, 1, 3])
    with a:
        if st.button("📋 Copy", key="ai_copy", use_container_width=True):
            # Streamlit doesn't have native clipboard write — show the text
            # in a code block as fallback.
            st.toast("Select the markdown below and copy", icon="📋")
    with b:
        if st.button("👍 Helpful", key="ai_thumbup", use_container_width=True):
            st.session_state["ai_feedback"]["up"] += 1
            st.toast("Thanks for the feedback!", icon="🙏")
    with c:
        if st.button("👎 Not helpful", key="ai_thumbdown", use_container_width=True):
            st.session_state["ai_feedback"]["down"] += 1
            st.toast("Thanks — we'll improve.", icon="🙏")
    with d:
        if st.button("🔁 Regenerate", key="ai_regenerate"):
            st.session_state["ai_summary"] = ""
            st.rerun()


# -----------------------------------------------------------------------------
# Debug / info expanders
# -----------------------------------------------------------------------------
with st.expander("ℹ️ How this works"):
    st.markdown(
        """
- **Primary model:** Groq `llama-3.3-70b-versatile` (free tier, sub-second response).
- **Fallback chain:** Groq `gpt-oss-120b` → Groq `llama-3.1-8b-instant` → Gemini `gemini-2.5-flash` → static message.
- **Context budget:** ~4000 tokens. Schema, stats, top correlations, quality flags, and a 3-row PII-scrubbed sample.
- **Privacy:** Indian PII (PAN, Aadhaar, GSTIN, IFSC, mobile, email) is regex-scrubbed before any outbound call.
- **Limits:** 5 calls per session — refresh to reset.
        """
    )

if st.session_state.get("ai_last_error"):
    with st.expander("⚠️ Last error (debug)"):
        st.code(st.session_state["ai_last_error"])
