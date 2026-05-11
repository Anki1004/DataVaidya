"""Reusable Streamlit components for DataVaidya.

All components follow the design system defined in :mod:`ui.theme`. Helpers
that render HTML use ``unsafe_allow_html=True``; chart-returning helpers hand
back a :class:`plotly.graph_objects.Figure` so callers control sizing.
"""
from __future__ import annotations

from html import escape
from typing import Any, Literal

import plotly.graph_objects as go
import streamlit as st

from utils.constants import (
    PRIMARY_VIOLET,
    ACCENT_CYAN,
    SUCCESS_GREEN,
    WARNING_AMBER,
    ERROR_RED,
    TEXT_MUTED,
    BG_SURFACE,
    ZONE_COLORS,
    health_zone,
)
from ui.theme import PLOTLY_TEMPLATE, PLOTLY_CONFIG

__all__ = [
    "gradient_header",
    "metric_card",
    "health_gauge",
    "health_breakdown_grid",
    "info_pill",
    "code_block",
    "before_after_metrics",
    "empty_state",
    "version_badge",
]


# ---------------------------------------------------------------------------
# Internal color map for metric_card accents.
# ---------------------------------------------------------------------------
_COLOR_MAP: dict[str, str] = {
    "violet": PRIMARY_VIOLET,
    "cyan": ACCENT_CYAN,
    "green": SUCCESS_GREEN,
    "amber": WARNING_AMBER,
    "red": ERROR_RED,
}


# ---------------------------------------------------------------------------
# Headings & pills
# ---------------------------------------------------------------------------
def gradient_header(text: str, *, level: int = 1, animated: bool = False) -> None:
    """Render heading with violet→cyan gradient text.

    Parameters
    ----------
    text:
        Heading text. HTML-escaped before rendering.
    level:
        Heading level 1-3. Anything outside that range falls back to ``h2``.
    animated:
        If ``True``, apply the pure-CSS typewriter animation (``.dv-typewriter``)
        with a blinking caret. The ``--dv-chars`` CSS variable is set inline so
        the animation duration matches the text length.
    """
    tag = f"h{level}" if level in (1, 2, 3) else "h2"
    safe = escape(str(text))
    if animated:
        chars = max(1, len(str(text)))
        st.markdown(
            f'<{tag} class="gradient-text dv-typewriter" '
            f'style="--dv-chars:{chars};">{safe}</{tag}>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<{tag} class="gradient-text">{safe}</{tag}>',
            unsafe_allow_html=True,
        )


def hero(title: str, tagline: str = "", version: str = "") -> None:
    """Render the DataVaidya hero block with animated gradient-mesh background.

    Pure-CSS — uses ``.dv-hero`` mesh from :mod:`ui.theme` and the typewriter
    animation on the title. Falls back gracefully if ``tagline`` or ``version``
    are empty.

    Parameters
    ----------
    title:
        Main headline. Rendered with violet→cyan gradient + typewriter effect.
    tagline:
        Optional sub-line shown beneath the title in muted text.
    version:
        Optional version string rendered as a pill at the bottom-right.
    """
    safe_title = escape(str(title))
    chars = max(1, len(str(title)))
    parts = [
        '<div class="dv-hero">',
        f'  <h1 class="gradient-text dv-typewriter" '
        f'style="--dv-chars:{chars};margin:0 0 0.5rem 0;">{safe_title}</h1>',
    ]
    if tagline:
        parts.append(
            f'  <p style="color:var(--muted);font-size:1.1rem;margin:0 0 0.75rem 0;">'
            f'{escape(str(tagline))}</p>'
        )
    if version:
        parts.append(
            f'  <span class="dv-pill info" style="font-family:JetBrains Mono,monospace;">'
            f'v{escape(str(version))}</span>'
        )
    parts.append('</div>')
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def info_pill(
    text: str,
    variant: Literal["info", "warning", "success", "error"] = "info",
) -> None:
    """Inline pill using the ``.dv-pill`` design-system class."""
    variant = variant if variant in ("info", "warning", "success", "error") else "info"
    safe = escape(str(text))
    st.markdown(
        f'<span class="dv-pill {variant}">{safe}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI / metric cards
# ---------------------------------------------------------------------------
def metric_card(
    label: str,
    value: str | int | float | None,
    delta: str | None = None,
    color: Literal["violet", "cyan", "green", "amber", "red"] = "violet",
) -> None:
    """Custom KPI card.

    Renders an HTML card styled by the design system. ``value`` of ``None``
    falls back to an em-dash with muted text so empty states stay legible.
    """
    accent = _COLOR_MAP.get(color, PRIMARY_VIOLET)
    is_empty = value is None
    display_value = "—" if is_empty else _format_value(value)
    value_color = TEXT_MUTED if is_empty else "var(--text)"

    label_html = escape(str(label))
    value_html = escape(str(display_value))

    delta_html = ""
    if delta and not is_empty:
        delta_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.8rem;color:{accent};margin-top:0.35rem;">'
            f"{escape(str(delta))}</div>"
        )

    html = (
        f'<div class="dv-card" style="border-left:3px solid {accent};">'
        f'<div style="color:var(--muted);font-size:0.75rem;'
        f'text-transform:uppercase;letter-spacing:0.06em;'
        f'font-weight:500;margin-bottom:0.5rem;">{label_html}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;'
        f'font-weight:600;font-size:1.75rem;color:{value_color};'
        f'line-height:1.1;">{value_html}</div>'
        f"{delta_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _format_value(value: str | int | float) -> str:
    """Format numbers compactly; leave strings untouched."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


# ---------------------------------------------------------------------------
# Health gauge + breakdown
# ---------------------------------------------------------------------------
def health_gauge(score: int | float | None) -> go.Figure:
    """Plotly Indicator gauge (0-100) with four health zones.

    Returns the figure — callers render via
    ``st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)``.
    """
    has_data = score is not None
    try:
        numeric = float(score) if has_data else 0.0
    except (TypeError, ValueError):
        numeric = 0.0
        has_data = False

    if numeric != numeric:  # NaN guard
        numeric = 0.0
        has_data = False

    numeric = max(0.0, min(100.0, numeric))

    zone_critical = ZONE_COLORS.get("Critical", ERROR_RED)
    zone_needs = ZONE_COLORS.get("Needs Work", WARNING_AMBER)
    zone_good = ZONE_COLORS.get("Good", ACCENT_CYAN)
    zone_excellent = ZONE_COLORS.get("Excellent", SUCCESS_GREEN)

    if has_data:
        zone_label = health_zone(numeric)
        bar_color = ZONE_COLORS.get(zone_label, PRIMARY_VIOLET)
        title_text = (
            f"<span style='font-size:0.85rem;color:{TEXT_MUTED};'>HEALTH SCORE</span>"
            f"<br><span style='font-size:0.75rem;color:{bar_color};"
            f"font-family:Inter;font-weight:600;'>{escape(zone_label).upper()}</span>"
        )
    else:
        bar_color = TEXT_MUTED
        title_text = (
            f"<span style='font-size:0.85rem;color:{TEXT_MUTED};'>HEALTH SCORE</span>"
            f"<br><span style='font-size:0.75rem;color:{TEXT_MUTED};'>No data</span>"
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=numeric,
            number={
                "font": {"family": "JetBrains Mono", "size": 44, "color": "#F8FAFC"},
                "suffix": "",
                "valueformat": ".0f",
            },
            title={"text": title_text, "font": {"family": "Inter", "size": 14}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": TEXT_MUTED,
                    "tickfont": {"color": TEXT_MUTED, "size": 11},
                    "tickvals": [0, 25, 50, 75, 100],
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": f"{zone_critical}26"},
                    {"range": [50, 70], "color": f"{zone_needs}26"},
                    {"range": [70, 90], "color": f"{zone_good}26"},
                    {"range": [90, 100], "color": f"{zone_excellent}26"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 2},
                    "thickness": 0.85,
                    "value": numeric,
                },
            },
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=280,
        margin=dict(l=24, r=24, t=64, b=24),
    )
    return fig


def health_breakdown_grid(deductions: dict[str, float] | None) -> None:
    """4-card grid for penalty categories.

    Always reserves four slots; if ``deductions`` is empty (or ``None``) a
    single full-width "no data" card is rendered instead.
    """
    if not deductions:
        st.markdown(
            '<div class="dv-card" style="text-align:center;">'
            f'<div style="color:{TEXT_MUTED};font-size:0.9rem;">'
            "No penalty data available yet — run an audit to populate the breakdown."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Sort descending by penalty magnitude, keep top 4 to fit grid.
    items = sorted(deductions.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:4]

    cols = st.columns(4)
    for idx in range(4):
        with cols[idx]:
            if idx < len(items):
                label, value = items[idx]
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                color = _penalty_color(numeric)
                metric_card(
                    label=str(label),
                    value=f"-{numeric:.1f}",
                    delta="penalty",
                    color=color,
                )
            else:
                # Reserved empty slot — keeps layout stable.
                st.markdown(
                    '<div class="dv-card" style="opacity:0.45;border-style:dashed;">'
                    f'<div style="color:{TEXT_MUTED};font-size:0.75rem;'
                    'text-transform:uppercase;letter-spacing:0.06em;">—</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:1.5rem;color:{TEXT_MUTED};">0.0</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )


def _penalty_color(value: float) -> Literal["violet", "cyan", "green", "amber", "red"]:
    """Map a penalty magnitude to a card accent color."""
    if value >= 15:
        return "red"
    if value >= 8:
        return "amber"
    if value >= 3:
        return "violet"
    return "cyan"


# ---------------------------------------------------------------------------
# Code & comparison helpers
# ---------------------------------------------------------------------------
def code_block(code: str, language: str = "python") -> None:
    """Consistent code-block wrapper around ``st.code``."""
    st.code(code, language=language)


def before_after_metrics(before: dict[str, Any], after: dict[str, Any]) -> None:
    """4-column before/after comparison.

    Renders one row per canonical key (``rows``, ``cells``, ``memory_mb``,
    ``health_score``). Missing values on either side show ``—``; deltas are
    auto-computed when both sides are numeric.
    """
    before = before or {}
    after = after or {}

    rows: list[tuple[str, str, bool]] = [
        ("rows", "Rows", False),
        ("cells", "Cells", False),
        ("memory_mb", "Memory (MB)", False),
        ("health_score", "Health Score", True),
    ]

    cols = st.columns(4)
    for col, (key, label, higher_better) in zip(cols, rows):
        with col:
            b_val = before.get(key)
            a_val = after.get(key)
            display_before = _format_value(b_val) if b_val is not None else "—"
            display_after = _format_value(a_val) if a_val is not None else "—"

            delta_text, delta_color = _compute_delta(b_val, a_val, higher_better)

            label_html = escape(label)
            html = (
                '<div class="dv-card" style="padding:1rem 1.1rem;">'
                f'<div style="color:var(--muted);font-size:0.72rem;'
                'text-transform:uppercase;letter-spacing:0.06em;'
                f'font-weight:500;margin-bottom:0.5rem;">{label_html}</div>'
                '<div style="display:flex;align-items:baseline;gap:0.5rem;'
                'font-family:\'JetBrains Mono\',monospace;">'
                f'<span style="color:{TEXT_MUTED};font-size:0.95rem;">'
                f'{escape(display_before)}</span>'
                f'<span style="color:{TEXT_MUTED};">→</span>'
                '<span style="color:var(--text);font-size:1.25rem;font-weight:600;">'
                f'{escape(display_after)}</span>'
                '</div>'
                f'<div style="margin-top:0.4rem;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.78rem;color:{delta_color};">{escape(delta_text)}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)


def _compute_delta(
    before: Any, after: Any, higher_better: bool
) -> tuple[str, str]:
    """Compute display delta + color for the before/after comparison."""
    try:
        b = float(before)
        a = float(after)
    except (TypeError, ValueError):
        return "—", TEXT_MUTED

    if b != b or a != a:  # NaN guard
        return "—", TEXT_MUTED

    diff = a - b
    if diff == 0:
        return "no change", TEXT_MUTED

    pct = (diff / b * 100.0) if b not in (0, 0.0) else None
    sign = "+" if diff > 0 else "−"
    magnitude = abs(diff)

    if pct is not None:
        body = f"{sign}{_format_value(magnitude)} ({sign}{abs(pct):.1f}%)"
    else:
        body = f"{sign}{_format_value(magnitude)}"

    improving = (diff > 0) if higher_better else (diff < 0)
    color = SUCCESS_GREEN if improving else ERROR_RED
    return body, color


# ---------------------------------------------------------------------------
# Empty state + sidebar badge
# ---------------------------------------------------------------------------
def empty_state(
    title: str,
    message: str,
    *,
    cta: str | None = None,
    cta_page: str | None = None,
) -> None:
    """Centered card prompting next action.

    If ``cta_page`` is provided, a Streamlit ``st.page_link`` is rendered
    underneath the card. Otherwise the CTA label (if any) becomes a plain
    button-styled hint inside the card.
    """
    title_html = escape(str(title))
    message_html = escape(str(message))

    inline_cta = ""
    if cta and not cta_page:
        inline_cta = (
            '<div style="margin-top:1rem;display:inline-block;padding:0.55rem 1.1rem;'
            'border-radius:12px;background:linear-gradient(135deg,#8B5CF6,#7C3AED);'
            'color:#fff;font-weight:600;font-size:0.875rem;">'
            f"{escape(cta)}</div>"
        )

    html = (
        '<div class="dv-card" style="text-align:center;padding:2.5rem 1.5rem;'
        'max-width:560px;margin:2rem auto;">'
        '<div style="font-size:1.25rem;font-weight:700;'
        f'color:var(--text);margin-bottom:0.5rem;">{title_html}</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.95rem;line-height:1.55;">'
        f'{message_html}</div>'
        f"{inline_cta}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    if cta and cta_page:
        try:
            st.page_link(cta_page, label=cta, icon="→")
        except Exception:
            # st.page_link requires a valid registered page; fall back gracefully.
            st.markdown(
                f'<div style="text-align:center;color:{TEXT_MUTED};font-size:0.85rem;">'
                f"Navigate to: <code>{escape(cta_page)}</code></div>",
                unsafe_allow_html=True,
            )


def version_badge() -> None:
    """Render APP_VERSION + tagline in the sidebar footer."""
    try:
        from utils.constants import APP_VERSION, APP_TAGLINE  # type: ignore
    except ImportError:  # pragma: no cover - constants may evolve
        APP_VERSION = "0.1.0"
        APP_TAGLINE = "Diagnose. Heal. Visualize."

    with st.sidebar:
        st.markdown(
            '<div style="margin-top:2rem;padding-top:1rem;'
            'border-top:1px solid rgba(148,163,184,0.12);text-align:center;">'
            '<div class="gradient-text" style="font-size:0.95rem;'
            'font-weight:800;letter-spacing:-0.01em;">DataVaidya</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.72rem;'
            f'font-family:\'JetBrains Mono\',monospace;margin-top:0.25rem;">'
            f"v{escape(str(APP_VERSION))}</div>"
            f'<div style="color:{TEXT_MUTED};font-size:0.7rem;'
            f'margin-top:0.35rem;font-style:italic;">{escape(str(APP_TAGLINE))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
