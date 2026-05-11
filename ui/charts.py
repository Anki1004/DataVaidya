"""Plotly chart builders for DataVaidya.

All functions are pure: given a pandas object, return a ``plotly.graph_objects.Figure``.
None raise on empty / edge inputs — instead they return a placeholder figure with a
centered "No data available" annotation. The shared ``PLOTLY_TEMPLATE`` from
``ui.theme`` is applied as the final layout step so every chart inherits the app's
visual identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import PLOTLY_COLORWAY, PLOTLY_HEATMAP_SCALE, PLOTLY_TEMPLATE
from utils.constants import ACCENT_CYAN, ERROR_RED, PRIMARY_VIOLET, TEXT_MUTED


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _empty_figure(message: str = "No data available") -> go.Figure:
    """Return a templated placeholder figure with a centered annotation.

    Used whenever a chart cannot be drawn (empty frame, insufficient columns,
    all-NaN series, etc.). The figure is fully valid Plotly output — callers
    can render it without special-casing.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color=TEXT_MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
    return fig


def _apply_template(fig: go.Figure, **layout: Any) -> go.Figure:
    """Apply the shared template plus any extra layout kwargs as the last step."""
    fig.update_layout(template=PLOTLY_TEMPLATE, **layout)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def missing_bar(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of missing-value percentage per column.

    Columns are sorted descending by missing share; only the worst ``top_n``
    are shown. An empty or column-less frame returns a placeholder.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return _empty_figure()

    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    missing_pct = missing_pct.head(top_n)

    if missing_pct.empty:
        return _empty_figure()

    # Plot ascending so the worst column ends up at the top of the bar chart.
    missing_pct = missing_pct.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=missing_pct.values,
            y=missing_pct.index.astype(str),
            orientation="h",
            marker=dict(color=PRIMARY_VIOLET),
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>",
        )
    )
    return _apply_template(
        fig,
        title="Missing values by column",
        xaxis_title="Missing (%)",
        yaxis_title=None,
        height=max(300, 22 * len(missing_pct) + 120),
        margin=dict(l=10, r=20, t=60, b=40),
    )


def correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> go.Figure:
    """Heatmap of pairwise numeric correlations.

    Returns a placeholder when fewer than two numeric columns are present
    after coercion (correlation is undefined in that case).
    """
    if df is None or df.empty:
        return _empty_figure()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return _empty_figure("Need at least 2 numeric columns")

    try:
        corr = numeric_df.corr(method=method)
    except Exception:
        return _empty_figure("Correlation could not be computed")

    if corr.empty or corr.isna().all().all():
        return _empty_figure()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.astype(str),
            y=corr.index.astype(str),
            colorscale=PLOTLY_HEATMAP_SCALE,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate="<b>%{y}</b> &times; <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        )
    )
    return _apply_template(
        fig,
        title=f"Correlation ({method})",
        xaxis=dict(tickangle=-45),
        height=max(360, 26 * len(corr) + 160),
        margin=dict(l=10, r=20, t=60, b=80),
    )


def numeric_distribution(series: pd.Series) -> go.Figure:
    """Histogram of a numeric series with an overlaid KDE line.

    Bars are rendered in ``PRIMARY_VIOLET``; the KDE curve in ``ACCENT_CYAN``.
    The KDE is skipped when there are fewer than five non-null observations.
    """
    if series is None or len(series) == 0:
        return _empty_figure()

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return _empty_figure()

    name = str(getattr(series, "name", "value") or "value")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=numeric.values,
            name=name,
            marker=dict(color=PRIMARY_VIOLET, line=dict(width=0)),
            opacity=0.85,
            histnorm="probability density",
            hovertemplate="Range: %{x}<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # KDE overlay — only attempt when we have enough data and some variance.
    if len(numeric) >= 5 and float(numeric.std(ddof=0) or 0.0) > 0:
        try:
            from scipy.stats import gaussian_kde  # type: ignore

            kde = gaussian_kde(numeric.values)
            x_min, x_max = float(numeric.min()), float(numeric.max())
            xs = np.linspace(x_min, x_max, 200)
            ys = kde(xs)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="KDE",
                    line=dict(color=ACCENT_CYAN, width=2.5),
                    hovertemplate="x = %{x:.3f}<br>density = %{y:.4f}<extra></extra>",
                )
            )
        except Exception:
            # SciPy missing or singular covariance — silently fall back to histogram only.
            pass

    return _apply_template(
        fig,
        title=f"Distribution of {name}",
        xaxis_title=name,
        yaxis_title="Density",
        bargap=0.02,
        showlegend=True,
        height=360,
        margin=dict(l=10, r=20, t=60, b=40),
    )


def categorical_top10(series: pd.Series) -> go.Figure:
    """Horizontal bar of the top 10 categories by frequency.

    Values are cast to ``str`` before counting so mixed-type columns are safe.
    Anything outside the top 10 is collapsed into a single ``"Other"`` bucket.
    """
    if series is None or len(series) == 0:
        return _empty_figure()

    cleaned = series.dropna()
    if cleaned.empty:
        return _empty_figure()

    counts = cleaned.astype(str).value_counts()
    if counts.empty:
        return _empty_figure()

    top = counts.head(10)
    remainder = int(counts.iloc[10:].sum()) if len(counts) > 10 else 0
    if remainder > 0:
        top = pd.concat([top, pd.Series({"Other": remainder})])

    # Sort ascending for horizontal bar so the largest category sits on top.
    top = top.sort_values(ascending=True)
    name = str(getattr(series, "name", "category") or "category")

    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index.astype(str),
            orientation="h",
            marker=dict(color=PRIMARY_VIOLET),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        )
    )
    return _apply_template(
        fig,
        title=f"Top categories — {name}",
        xaxis_title="Count",
        yaxis_title=None,
        height=max(300, 28 * len(top) + 120),
        margin=dict(l=10, r=20, t=60, b=40),
    )


def box_plot_outliers(series: pd.Series) -> go.Figure:
    """Single-series box plot with outliers drawn in ``ERROR_RED``."""
    if series is None or len(series) == 0:
        return _empty_figure()

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return _empty_figure()

    name = str(getattr(series, "name", "value") or "value")

    fig = go.Figure(
        go.Box(
            x=numeric.values,
            name=name,
            orientation="h",
            boxpoints="outliers",
            marker=dict(color=ERROR_RED, size=6, opacity=0.85),
            line=dict(color=PRIMARY_VIOLET),
            fillcolor="rgba(124, 58, 237, 0.25)",
            hovertemplate="%{x}<extra></extra>",
        )
    )
    return _apply_template(
        fig,
        title=f"Box plot — {name}",
        xaxis_title=name,
        showlegend=False,
        height=260,
        margin=dict(l=10, r=20, t=60, b=40),
    )


def class_imbalance_bar(series: pd.Series) -> go.Figure:
    """Bar chart of class frequencies with a balanced-share reference line.

    Shows a horizontal dashed line at ``1 / k`` where ``k`` is the number of
    distinct classes — anything well above or below that line is imbalanced.
    """
    if series is None or len(series) == 0:
        return _empty_figure()

    cleaned = series.dropna()
    if cleaned.empty:
        return _empty_figure()

    counts = cleaned.astype(str).value_counts()
    if counts.empty:
        return _empty_figure()

    total = int(counts.sum())
    shares = counts / total
    k = len(shares)
    balanced = 1.0 / k if k > 0 else 0.0

    name = str(getattr(series, "name", "class") or "class")

    fig = go.Figure(
        go.Bar(
            x=shares.index.astype(str),
            y=shares.values,
            marker=dict(color=PRIMARY_VIOLET),
            customdata=counts.values,
            hovertemplate="<b>%{x}</b><br>Share: %{y:.2%}<br>Count: %{customdata:,}<extra></extra>",
            name="Share",
        )
    )

    if k > 1:
        fig.add_hline(
            y=balanced,
            line=dict(color=ACCENT_CYAN, width=2, dash="dash"),
            annotation_text=f"Balanced (1/{k} = {balanced:.2%})",
            annotation_position="top right",
            annotation_font=dict(color=ACCENT_CYAN, size=11),
        )

    return _apply_template(
        fig,
        title=f"Class balance — {name}",
        xaxis_title=name,
        yaxis_title="Share",
        yaxis=dict(tickformat=".0%"),
        showlegend=False,
        height=340,
        margin=dict(l=10, r=20, t=60, b=40),
    )


def memory_footprint_bar(memory_dict: dict[str, float]) -> go.Figure:
    """Bar chart of column → memory (MB), sorted descending, truncated to 30."""
    if not memory_dict:
        return _empty_figure()

    # Coerce to floats, drop anything that can't be made numeric.
    cleaned: dict[str, float] = {}
    for col, mb in memory_dict.items():
        try:
            cleaned[str(col)] = float(mb)
        except (TypeError, ValueError):
            continue

    if not cleaned:
        return _empty_figure()

    ordered = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)[:30]
    # Reverse so the largest sits at the top of a horizontal-style read.
    ordered = list(reversed(ordered))
    cols = [c for c, _ in ordered]
    vals = [v for _, v in ordered]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=cols,
            orientation="h",
            marker=dict(color=PLOTLY_COLORWAY[0] if PLOTLY_COLORWAY else PRIMARY_VIOLET),
            hovertemplate="<b>%{y}</b><br>%{x:.3f} MB<extra></extra>",
        )
    )
    return _apply_template(
        fig,
        title="Memory footprint by column",
        xaxis_title="Memory (MB)",
        yaxis_title=None,
        height=max(320, 22 * len(cols) + 120),
        margin=dict(l=10, r=20, t=60, b=40),
    )


__all__ = [
    "missing_bar",
    "correlation_heatmap",
    "numeric_distribution",
    "categorical_top10",
    "box_plot_outliers",
    "class_imbalance_bar",
    "memory_footprint_bar",
]
