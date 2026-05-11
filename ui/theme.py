"""DataVaidya design system: CSS injection and Plotly theming.

Exposes a single ``inject_css`` entry point for Streamlit pages plus a fully
configured Plotly ``Template`` so every chart in the app shares the same
visual language as the surrounding UI.
"""
from __future__ import annotations

import plotly.graph_objects as go

from utils.constants import (
    BG_DEEP,
    BG_SURFACE,
    PRIMARY_VIOLET,
    ACCENT_CYAN,
    SUCCESS_GREEN,
    WARNING_AMBER,
    ERROR_RED,
    TEXT_PRIMARY,
    TEXT_MUTED,
    PALETTE,
)

__all__ = [
    "inject_css",
    "PLOTLY_TEMPLATE",
    "PLOTLY_COLORWAY",
    "PLOTLY_HEATMAP_SCALE",
    "PLOTLY_CONFIG",
]


# ---------------------------------------------------------------------------
# CSS — the canonical DataVaidya stylesheet.
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #0F172A;
    --surface: #1E293B;
    --surface-2: #273449;
    --border: #334155;
    --border-soft: rgba(148, 163, 184, 0.12);
    --primary: #8B5CF6;
    --primary-600: #7C3AED;
    --primary-300: #A78BFA;
    --cyan: #06B6D4;
    --success: #22C55E;
    --warning: #F59E0B;
    --error: #EF4444;
    --text: #F8FAFC;
    --muted: #94A3B8;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-glow: 0 0 0 1px rgba(139, 92, 246, 0.35), 0 8px 24px -8px rgba(139, 92, 246, 0.45);
    --shadow-soft: 0 1px 2px rgba(0,0,0,0.25), 0 4px 16px rgba(0,0,0,0.18);
    --t-fast: 120ms cubic-bezier(0.4, 0, 0.2, 1);
    --t-med: 200ms cubic-bezier(0.4, 0, 0.2, 1);
}
html, body, [class*="css"], .stApp {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    color: var(--text);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background:
        radial-gradient(1200px 600px at 80% -10%, rgba(139, 92, 246, 0.08), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(6, 182, 212, 0.06), transparent 60%),
        var(--bg);
}
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--primary), var(--primary-600)); border-radius: 999px; }
*::-webkit-scrollbar-track { background: transparent; }
::selection { background: rgba(139, 92, 246, 0.35); color: var(--text); }
.main .block-container, section.main > div.block-container {
    padding-top: 2rem; padding-bottom: 4rem;
    padding-left: 2.5rem; padding-right: 2.5rem;
    max-width: 1400px;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] * { color: var(--text); }
h1, h2, h3, h4 { font-family: 'Inter'; font-weight: 800; letter-spacing: -0.02em; color: var(--text); }
h1 { font-size: 2.25rem; margin-bottom: 0.75rem; }
h2 { font-size: 1.6rem; margin-top: 1.5rem; }
h3 { font-size: 1.2rem; }
.gradient-text {
    background: linear-gradient(90deg, var(--primary) 0%, var(--cyan) 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    font-weight: 800;
}
.stButton > button, .stDownloadButton > button {
    font-weight: 600; font-size: 0.875rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    background: var(--surface-2); color: var(--text);
    padding: 0.55rem 1.1rem;
    transition: transform var(--t-fast), box-shadow var(--t-med), background var(--t-med);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, var(--primary), var(--primary-600));
    border-color: var(--primary);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
    color: #fff;
}
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.25rem;
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: 1.85rem;
    color: var(--text);
}
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono'; font-size: 0.8rem; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-lg);
    transition: border-color var(--t-med), box-shadow var(--t-med);
}
[data-testid="stFileUploader"] section:hover, [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--primary);
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.10);
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent; border: none; color: var(--muted);
    font-weight: 500; padding: 0.65rem 1rem;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text); border-bottom-color: var(--primary);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    overflow: hidden;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea, [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text) !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.20) !important;
    outline: none !important;
}
[data-testid="stToast"] {
    background: var(--surface);
    border-left: 3px solid var(--primary);
    border-radius: var(--radius-md);
}
[data-testid="stPlotlyChart"] {
    border-radius: var(--radius-lg);
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    padding: 0.5rem;
}
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
}
.stSpinner > div > div { border-top-color: var(--primary) !important; }

/* ===== Card: 1px violet border, glow + lift on hover ===== */
.dv-card {
    position: relative;
    background: var(--surface);
    border: 1px solid rgba(139, 92, 246, 0.22);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset, 0 4px 16px rgba(0,0,0,0.18);
    transition: transform var(--t-med), border-color var(--t-med),
                box-shadow var(--t-med), background var(--t-med);
    will-change: transform;
}
.dv-card::before {
    content: "";
    position: absolute; inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(135deg, rgba(139,92,246,0.0), rgba(139,92,246,0.0));
    -webkit-mask: linear-gradient(#000, #000) content-box, linear-gradient(#000, #000);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
    transition: background var(--t-med);
}
.dv-card:hover {
    transform: translateY(-3px);
    border-color: rgba(139, 92, 246, 0.75);
    box-shadow: 0 0 0 1px rgba(139,92,246,0.35),
                0 12px 32px -10px rgba(139,92,246,0.45),
                0 4px 16px rgba(0,0,0,0.22);
    background: linear-gradient(180deg, rgba(139,92,246,0.04), rgba(139,92,246,0.0)) , var(--surface);
}
.dv-card:hover::before {
    background: linear-gradient(135deg, rgba(139,92,246,0.55), rgba(6,182,212,0.45));
}

/* ===== Buttons: press feedback + lift ===== */
.stButton > button, .stDownloadButton > button {
    transition: transform 80ms cubic-bezier(0.4,0,0.2,1),
                box-shadow var(--t-med),
                background var(--t-med),
                border-color var(--t-med),
                color var(--t-med);
    will-change: transform;
}
.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="baseButton-primary"]:active,
[data-testid="baseButton-secondary"]:active {
    transform: translateY(0) scale(0.97);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.25);
    filter: brightness(0.95);
}

/* ===== Hero: animated radial-gradient mesh background ===== */
.dv-hero {
    position: relative;
    isolation: isolate;
    padding: 2.75rem 2rem 2.25rem;
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-bottom: 1.25rem;
    background:
        radial-gradient(60% 80% at 18% 24%, rgba(139,92,246,0.22), transparent 60%),
        radial-gradient(55% 75% at 82% 14%, rgba(6,182,212,0.18), transparent 60%),
        radial-gradient(70% 80% at 50% 110%, rgba(34,211,238,0.10), transparent 60%),
        linear-gradient(180deg, rgba(30,41,59,0.55), rgba(15,23,42,0.0));
    border: 1px solid rgba(139,92,246,0.18);
    box-shadow: 0 8px 32px -12px rgba(139,92,246,0.25), 0 1px 0 rgba(255,255,255,0.03) inset;
}
.dv-hero::before, .dv-hero::after {
    content: "";
    position: absolute;
    inset: -20%;
    z-index: -1;
    background:
        radial-gradient(circle at 30% 30%, rgba(139,92,246,0.14) 0%, transparent 38%),
        radial-gradient(circle at 70% 60%, rgba(6,182,212,0.12)  0%, transparent 38%),
        radial-gradient(circle at 50% 80%, rgba(167,139,250,0.10) 0%, transparent 38%);
    filter: blur(20px);
    animation: dv-mesh-drift 22s ease-in-out infinite alternate;
}
.dv-hero::after {
    animation-duration: 28s;
    animation-direction: alternate-reverse;
    opacity: 0.65;
}
@keyframes dv-mesh-drift {
    0%   { transform: translate3d(0,   0,   0) scale(1); }
    50%  { transform: translate3d(2%,  -3%, 0) scale(1.04); }
    100% { transform: translate3d(-3%, 2%,  0) scale(0.98); }
}

/* ===== Typewriter animation for the main headline ===== */
.dv-typewriter {
    display: inline-block;
    overflow: hidden;
    white-space: nowrap;
    vertical-align: bottom;
    width: 0;
    border-right: 3px solid var(--primary);
    animation:
        dv-type    1.6s steps(var(--dv-chars, 12), end) 0.15s forwards,
        dv-caret   0.85s step-end 1.8s infinite;
    padding-right: 0.15ch;
}
@keyframes dv-type  { to { width: calc(var(--dv-chars, 12) * 1ch + 0.15ch); } }
@keyframes dv-caret { 50% { border-color: transparent; } }

/* ===== Upload zone: soft pulsing glow ===== */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {
    animation: dv-pulse 2.8s ease-in-out infinite;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploader"] section:hover {
    animation: none;
}
@keyframes dv-pulse {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.45),
                    0 0 0 0 rgba(6, 182, 212, 0.25);
    }
    50% {
        box-shadow: 0 0 0 8px rgba(139, 92, 246, 0.0),
                    0 0 16px 4px rgba(139, 92, 246, 0.18);
    }
}

/* ===== Reduced motion: respect user preference ===== */
@media (prefers-reduced-motion: reduce) {
    .dv-typewriter {
        width: calc(var(--dv-chars, 12) * 1ch + 0.15ch) !important;
        animation: none !important;
        border-right: none !important;
    }
    .dv-hero::before, .dv-hero::after { animation: none !important; }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section { animation: none !important; }
    .dv-card, .stButton > button { transition: none !important; }
}

.dv-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.65rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.dv-pill.info    { background: rgba(6,182,212,.12);  color: #67E8F9; border: 1px solid rgba(6,182,212,.3); }
.dv-pill.success { background: rgba(34,197,94,.12);  color: #86EFAC; border: 1px solid rgba(34,197,94,.3); }
.dv-pill.warning { background: rgba(245,158,11,.12); color: #FCD34D; border: 1px solid rgba(245,158,11,.3); }
.dv-pill.error   { background: rgba(239,68,68,.12);  color: #FCA5A5; border: 1px solid rgba(239,68,68,.3); }
"""


# ---------------------------------------------------------------------------
# Plotly theme tokens.
# ---------------------------------------------------------------------------
PLOTLY_COLORWAY: tuple[str, ...] = tuple(PALETTE)

PLOTLY_HEATMAP_SCALE: list[list] = [
    [0.0, "#1E1B4B"],
    [0.2, "#4C1D95"],
    [0.4, "#7C3AED"],
    [0.6, "#8B5CF6"],
    [0.8, "#22D3EE"],
    [1.0, "#67E8F9"],
]


def _build_template() -> go.layout.Template:
    """Construct the DataVaidya Plotly Template (layout + data defaults)."""
    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=13, color=TEXT_PRIMARY),
        colorway=list(PLOTLY_COLORWAY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=56, b=44),
        title=dict(
            font=dict(family="Inter", size=18, color=TEXT_PRIMARY),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.18)",
            zeroline=False,
            tickfont=dict(color=TEXT_MUTED, size=12),
            title=dict(font=dict(color=TEXT_MUTED, size=12)),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.18)",
            zeroline=False,
            tickfont=dict(color=TEXT_MUTED, size=12),
            title=dict(font=dict(color=TEXT_MUTED, size=12)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
        hoverlabel=dict(
            bgcolor=BG_SURFACE,
            bordercolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=TEXT_PRIMARY, size=12),
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=TEXT_MUTED,
            activecolor=PRIMARY_VIOLET,
        ),
        separators=".,",
    )

    # Per-trace defaults.
    template.data.heatmap = [go.Heatmap(colorscale=PLOTLY_HEATMAP_SCALE)]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    template.data.box = [go.Box(marker=dict(color=PRIMARY_VIOLET))]
    template.data.histogram = [go.Histogram(marker=dict(line=dict(width=0)))]
    template.data.scatter = [go.Scatter(marker=dict(line=dict(width=0)))]

    return template


PLOTLY_TEMPLATE: go.layout.Template = _build_template()


PLOTLY_CONFIG: dict = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "resetScale2d",
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "datavaidya_chart",
        "scale": 2,
    },
}


def inject_css() -> None:
    """Inject DataVaidya design system styles. Call once at top of every page.

    Must not be cached: Streamlit needs to re-emit the ``<style>`` block on
    every script run so it survives full reruns.
    """
    import streamlit as st

    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
