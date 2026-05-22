"""
Statistical process control (p-chart) logic for the PreventativeScan tool.

P-chart methodology:
    p_bar  = weighted average issue rate for a category across all weeks
    SE     = sqrt(p_bar * (1 - p_bar) / n_week)   per week
    UCL    = p_bar + 3 * SE,  clipped to [0, 1]
    LCL    = p_bar - 3 * SE,  clipped to [0, 1]

Signals:
    - above_ucl:  week rate > UCL
    - below_lcl:  week rate < LCL  (shown on chart; not treated as negative alert)
    - run_above:  N consecutive weeks above centerline (not just UCL)

Plotly chart builder included so the UI can call one function per category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    CONTROL_CHART_MIN_WEEKS,
    CONTROL_CHART_MIN_N,
    OPERATIONAL_CATEGORIES,
)


# ── Core p-chart computation ──────────────────────────────────────────────────

def compute_control_limits(weekly_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Given the output of metrics.weekly_category_rates(), compute p-chart
    control limits for each category.

    Returns the same DataFrame with added columns:
        p_bar, se, ucl, lcl,
        above_ucl, below_lcl,
        has_limits (False if insufficient data)
    """
    out_rows = []

    for category, grp in weekly_rates.groupby("issue_category"):
        grp = grp.sort_values("week_start").copy()

        # Only use weeks with enough responses for the point to be meaningful
        valid_weeks = grp[grp["weekly_total"] >= CONTROL_CHART_MIN_N]
        n_valid = len(valid_weeks)

        if n_valid < CONTROL_CHART_MIN_WEEKS:
            grp["p_bar"] = np.nan
            grp["se"] = np.nan
            grp["ucl"] = np.nan
            grp["lcl"] = np.nan
            grp["above_ucl"] = False
            grp["below_lcl"] = False
            grp["has_limits"] = False
            out_rows.append(grp)
            continue

        # Weighted p_bar: sum of all category occurrences / sum of all weekly totals
        p_bar = (
            valid_weeks["count"].sum() / valid_weeks["weekly_total"].sum()
        )

        grp["p_bar"] = p_bar
        grp["has_limits"] = True

        # Per-week SE and limits
        def _limits(row):
            n = row["weekly_total"]
            if n < CONTROL_CHART_MIN_N:
                return pd.Series({"se": np.nan, "ucl": np.nan, "lcl": np.nan})
            se = np.sqrt(p_bar * (1 - p_bar) / n)
            ucl = min(1.0, p_bar + 3 * se)
            lcl = max(0.0, p_bar - 3 * se)
            return pd.Series({"se": se, "ucl": ucl, "lcl": lcl})

        limits = grp.apply(_limits, axis=1)
        grp["se"] = limits["se"]
        grp["ucl"] = limits["ucl"]
        grp["lcl"] = limits["lcl"]

        grp["above_ucl"] = grp["issue_rate"] > grp["ucl"]
        grp["below_lcl"] = (grp["issue_rate"] < grp["lcl"]) & grp["lcl"].notna()

        out_rows.append(grp)

    if not out_rows:
        return weekly_rates.copy()

    return pd.concat(out_rows, ignore_index=True)


def detect_run_above_centerline(
    chart_df: pd.DataFrame, consecutive: int = 2
) -> pd.DataFrame:
    """
    Flag weeks that are part of a run of N or more consecutive weeks above
    the centerline (p_bar) for a given category.

    Adds column: run_above_cl (bool)
    """
    out_rows = []
    for category, grp in chart_df.groupby("issue_category"):
        grp = grp.sort_values("week_start").copy()
        grp["run_above_cl"] = False

        if "p_bar" not in grp.columns or grp["p_bar"].isna().all():
            out_rows.append(grp)
            continue

        above = (grp["issue_rate"] > grp["p_bar"]).tolist()
        flags = [False] * len(above)

        streak = 0
        for i, val in enumerate(above):
            if val:
                streak += 1
            else:
                streak = 0
            if streak >= consecutive:
                for j in range(i - streak + 1, i + 1):
                    flags[j] = True

        grp["run_above_cl"] = flags
        out_rows.append(grp)

    return pd.concat(out_rows, ignore_index=True)


def spike_summary(chart_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of categories and weeks where an above-UCL signal fired.
    Useful for the alert engine.
    """
    spikes = chart_df[chart_df["above_ucl"].fillna(False)].copy()
    if spikes.empty:
        return pd.DataFrame(
            columns=["issue_category", "week_start", "week_label",
                     "issue_rate", "ucl", "p_bar"]
        )
    return (
        spikes[["issue_category", "week_start", "week_label", "issue_rate", "ucl", "p_bar"]]
        .sort_values(["issue_category", "week_start"])
        .reset_index(drop=True)
    )


# ── Plotly chart builder ──────────────────────────────────────────────────────

def build_control_chart(
    category: str,
    chart_df: pd.DataFrame,
    height: int = 420,
) -> go.Figure:
    """
    Build a Plotly p-chart figure for a single issue category.

    chart_df should be the output of compute_control_limits(), optionally
    enriched with detect_run_above_centerline().
    """
    grp = chart_df[chart_df["issue_category"] == category].sort_values("week_start")

    has_limits = grp["has_limits"].any() if "has_limits" in grp.columns else False
    x_labels = grp["week_label"].tolist()
    rates = grp["issue_rate"].tolist()

    fig = go.Figure()

    # ── UCL / LCL bands ──────────────────────────────────────────────────────
    if has_limits:
        ucl_vals = grp["ucl"].tolist()
        lcl_vals = grp["lcl"].tolist()
        p_bar_vals = grp["p_bar"].tolist()

        # Shaded control region (LCL to UCL)
        fig.add_trace(
            go.Scatter(
                x=x_labels + x_labels[::-1],
                y=ucl_vals + lcl_vals[::-1],
                fill="toself",
                fillcolor="rgba(173, 216, 230, 0.25)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Control region",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_labels, y=ucl_vals,
                mode="lines",
                line=dict(color="#e06c75", width=1.5, dash="dash"),
                name="UCL",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels, y=lcl_vals,
                mode="lines",
                line=dict(color="#61afef", width=1.5, dash="dash"),
                name="LCL",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels, y=p_bar_vals,
                mode="lines",
                line=dict(color="#98c379", width=1.5, dash="dot"),
                name="Centerline (p̄)",
            )
        )

    # ── Issue rate line ───────────────────────────────────────────────────────
    above_ucl = grp["above_ucl"].fillna(False).tolist() if "above_ucl" in grp.columns else [False] * len(rates)
    colors = ["#e06c75" if a else "#abb2bf" for a in above_ucl]

    counts = grp["count"].tolist() if "count" in grp.columns else [None] * len(rates)
    totals = grp["weekly_total"].tolist() if "weekly_total" in grp.columns else [None] * len(rates)

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=rates,
            mode="lines+markers",
            line=dict(color="#abb2bf", width=2),
            marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
            name="Issue rate",
            customdata=list(zip(counts, totals)),
            hovertemplate=(
                "%{x}<br>"
                "Rate: %{y:.1%}<br>"
                "n: %{customdata[0]} of %{customdata[1]} responses"
                "<extra></extra>"
            ),
        )
    )

    # ── Out-of-control annotations ────────────────────────────────────────────
    for i, (is_above, x_val, y_val) in enumerate(zip(above_ucl, x_labels, rates)):
        if is_above:
            fig.add_annotation(
                x=x_val, y=y_val,
                text="⚠",
                showarrow=False,
                yshift=14,
                font=dict(size=14, color="#e06c75"),
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    insufficient_note = (
        "" if has_limits
        else "<br><sub>Insufficient data for control limits "
        f"(need ≥{CONTROL_CHART_MIN_WEEKS} weeks with n≥{CONTROL_CHART_MIN_N})</sub>"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{category}</b> — Weekly Issue Rate{insufficient_note}",
            font=dict(size=15),
        ),
        xaxis=dict(title="Week", tickangle=-45),
        yaxis=dict(
            title="Issue rate",
            tickformat=".0%",
            range=[0, max(max(rates, default=0) * 1.3, 0.05)],
        ),
        height=height,
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(t=60, b=80, l=60, r=20),
    )

    return fig


def location_weekly_rates(
    df: pd.DataFrame, category: str, location: str
) -> pd.DataFrame:
    """
    Compute weekly category rates scoped to a single location.
    Returns a DataFrame compatible with compute_control_limits().
    """
    loc_df = df[df["member_location"] == location].copy()
    if loc_df.empty or "week_start" not in loc_df.columns:
        return pd.DataFrame()

    weekly_totals = (
        loc_df[loc_df["week_start"].notna()]
        .groupby("week_start")["response_id"]
        .count()
        .rename("weekly_total")
    )
    cat_counts = (
        loc_df[
            (loc_df["issue_category"] == category) & loc_df["week_start"].notna()
        ]
        .groupby("week_start")["response_id"]
        .count()
        .rename("count")
    )
    combined = pd.DataFrame(weekly_totals).join(cat_counts, how="left").fillna(0)
    combined["count"] = combined["count"].astype(int)
    combined["issue_category"] = category
    combined["issue_rate"] = combined.apply(
        lambda r: r["count"] / r["weekly_total"] if r["weekly_total"] > 0 else 0.0,
        axis=1,
    )
    week_label_map = (
        loc_df[loc_df["week_start"].notna()]
        .drop_duplicates("week_start")
        .set_index("week_start")["week_label"]
        .to_dict()
    )
    combined["week_label"] = combined.index.map(week_label_map)
    return combined.reset_index()


def location_control_chart(
    df: pd.DataFrame,
    category: str,
    location: str,
    height: int = 380,
) -> go.Figure | None:
    """
    Build a location-specific p-chart.
    Returns None if there are fewer than CONTROL_CHART_MIN_WEEKS valid points.
    """
    rates = location_weekly_rates(df, category, location)
    if rates.empty or len(rates) < CONTROL_CHART_MIN_WEEKS:
        return None

    chart_df = compute_control_limits(rates)
    fig = build_control_chart(category, chart_df, height=height)
    fig.update_layout(
        title=dict(text=f"<b>{category}</b> — {location} Weekly Issue Rate")
    )
    return fig
