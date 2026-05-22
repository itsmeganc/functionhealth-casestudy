"""
PreventativeScan — Automated Feedback Analysis Tool
Streamlit UI  |  Phase 5
"""

from __future__ import annotations

import os
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Local modules ─────────────────────────────────────────────────────────────
from loader import load_csv
from analyzer import run_analysis, cache_stats
from metrics import (
    compute_nps, nps_by_location, nps_by_week, nps_by_day,
    top_issues, weekly_summary, weekly_category_rates,
    daily_summary, daily_category_counts,
    issue_velocity,
    location_deviation_scores, location_health_scores,
    location_persistence_flags, churn_risk_concentration,
    promoter_drivers, sentiment_breakdown,
    sentiment_over_time, week_over_week_deltas,
)
from control_charts import (
    compute_control_limits, detect_run_above_centerline,
    spike_summary, build_control_chart, location_control_chart,
    location_weekly_rates,
)
from alerts import generate_alerts, alerts_by_tier, alerts_for_day
from summaries import daily_digest, weekly_digest
from export import (
    enriched_csv, enriched_excel,
    daily_digest_csv, daily_digest_excel,
    weekly_digest_csv, weekly_digest_excel,
    alerts_csv, alerts_excel,
    location_analysis_csv, location_analysis_excel,
)
from theme import (
    inject_css, alert_box, kpi_card, section_header,
    PRIMARY, DARK, MEDIUM_GRAY, LIGHT_GRAY, CREAM, WHITE,
    CHART_COLORS, SEVERITY_COLORS,
    CRITICAL_BORDER, WARNING_BORDER, LOCATION_BORDER,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "member_feedback.csv"
CACHE_PATH = Path(__file__).parent / "cache" / "analysis_cache.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PreventativeScan | Feedback Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_css(), unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_raw() -> tuple[pd.DataFrame, list[str]]:
    return load_csv(str(DATA_PATH))


@st.cache_data(show_spinner=False)
def _load_from_upload(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    from loader import load_csv_from_bytes as _lcb
    return _lcb(file_bytes)


@st.cache_data(show_spinner=False)
def _get_enriched_df_json(df_raw_json: str, _n_analyzed: int) -> str:
    """
    Merge the analysis cache into the raw df and return as JSON.
    _n_analyzed is used as a cache-buster: when new rows are analysed the
    count changes and Streamlit drops the cached result.
    """
    import io as _io
    from analyzer import load_from_cache as _lfc
    df = pd.read_json(_io.StringIO(df_raw_json), orient="split")
    df["response_date"] = pd.to_datetime(df["response_date"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    return _lfc(df, CACHE_PATH).to_json(orient="split", date_format="iso")


@st.cache_data(show_spinner=False)
def _compute_signal_directory(enriched_df_json: str, chart_df_json: str) -> str:
    """
    Find every (issue_category, location) pair with ≥1 above-UCL week.
    Checks both network-level (all locations) and per-location rates vs
    the network UCL.  Returns a JSON string of the summary DataFrame.
    """
    import io as _io

    df = pd.read_json(_io.StringIO(enriched_df_json), orient="split")
    df["response_date"] = pd.to_datetime(df["response_date"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    chart = pd.read_json(_io.StringIO(chart_df_json), orient="split")
    chart["week_start"] = pd.to_datetime(chart["week_start"])

    rows: list[dict] = []

    # ── Network-level signals ─────────────────────────────────────────────────
    net_spikes = spike_summary(chart)
    for _, r in net_spikes.iterrows():
        rows.append({
            "issue_category": r["issue_category"],
            "location": "All locations",
            "week_label": r["week_label"],
        })

    # ── Location-level signals (vs network UCL) ───────────────────────────────
    if "member_location" in df.columns and "week_start" in df.columns and "issue_category" in df.columns:
        net_ucl = chart[chart["has_limits"].fillna(False)][
            ["issue_category", "week_label", "ucl"]
        ].copy()

        grp = (
            df[df["week_start"].notna() & df["issue_category"].notna()]
            .groupby(["member_location", "issue_category", "week_label"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            df[df["week_start"].notna()]
            .groupby(["member_location", "week_label"])
            .size()
            .reset_index(name="weekly_total")
        )
        grp = grp.merge(totals, on=["member_location", "week_label"], how="left")
        grp["issue_rate"] = grp["count"] / grp["weekly_total"].replace(0, float("nan"))
        grp = grp.merge(net_ucl, on=["issue_category", "week_label"], how="left")
        grp["above_ucl"] = grp["issue_rate"] > grp["ucl"].fillna(float("inf"))

        for _, r in grp[grp["above_ucl"]].iterrows():
            rows.append({
                "issue_category": r["issue_category"],
                "location": r["member_location"],
                "week_label": r["week_label"],
            })

    empty = pd.DataFrame(columns=["issue_category", "location", "signal_weeks", "latest_signal"])
    if not rows:
        return empty.to_json(orient="split")

    all_sigs = pd.DataFrame(rows)
    summary = (
        all_sigs.groupby(["issue_category", "location"])
        .agg(signal_weeks=("week_label", "count"), latest_signal=("week_label", "max"))
        .reset_index()
        .sort_values("latest_signal", ascending=False)
        .reset_index(drop=True)
    )
    return summary.to_json(orient="split")


@st.cache_data(show_spinner=False)
def _compute_all_metrics(enriched_df_json: str) -> dict:
    """Run all metric computations on a pre-enriched DataFrame."""
    import io as _io
    df = pd.read_json(_io.StringIO(enriched_df_json), orient="split")
    df["response_date"] = pd.to_datetime(df["response_date"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    if "churn_risk" in df.columns:
        df["churn_risk"] = df["churn_risk"].astype(bool)
    if "severity_score" in df.columns:
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(2).astype(int)

    weekly_rates = weekly_category_rates(df)
    chart_df = compute_control_limits(weekly_rates)
    chart_df = detect_run_above_centerline(chart_df)
    spikes = spike_summary(chart_df)

    dev = location_deviation_scores(df)
    persist = location_persistence_flags(df)
    health = location_health_scores(df)

    all_alerts = generate_alerts(df, chart_df, spikes, dev, persist)

    return {
        "weekly_rates": weekly_rates.to_json(orient="split", date_format="iso"),
        "chart_df": chart_df.to_json(orient="split", date_format="iso"),
        "spikes": spikes.to_json(orient="split", date_format="iso"),
        "dev": dev.to_json(orient="split"),
        "persist": persist.to_json(orient="split"),
        "health": health.to_json(orient="split"),
        "all_alerts": all_alerts,
    }


def _restore_metrics(cache: dict) -> dict:
    """Deserialise the JSON strings back to DataFrames."""
    import io as _io
    def _df(key):
        return pd.read_json(_io.StringIO(cache[key]), orient="split")
    return {
        "weekly_rates": _df("weekly_rates"),
        "chart_df": _df("chart_df"),
        "spikes": _df("spikes"),
        "dev": _df("dev"),
        "persist": _df("persist"),
        "health": _df("health"),
        "all_alerts": cache["all_alerts"],
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Render sidebar controls, return (filtered_df, filter_state)."""
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.25rem;font-weight:700;color:{DARK};'
            f'margin-bottom:4px">🩻 PreventativeScan</div>'
            f'<div style="font-size:0.78rem;color:{MEDIUM_GRAY};'
            f'margin-bottom:1.5rem">Feedback Analysis Tool</div>',
            unsafe_allow_html=True,
        )

        # ── API Key ───────────────────────────────────────────────────────────
        st.markdown("**Analysis**")
        api_key = st.text_input(
            "Anthropic API key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Required to run AI analysis. Set ANTHROPIC_API_KEY in .env to pre-fill.",
        )

        stats = cache_stats(CACHE_PATH)
        total_rows = len(df_raw)
        cached_n = stats["total_cached"]
        uncached_n = total_rows - cached_n

        cache_pct = int(cached_n / total_rows * 100) if total_rows else 0
        st.caption(
            f"{cached_n}/{total_rows} responses analysed ({cache_pct}%)"
            + (f" — {stats['parse_errors']} parse errors" if stats["parse_errors"] else "")
        )

        run_disabled = not api_key or uncached_n == 0
        run_label = (
            "✓ Analysis up to date" if uncached_n == 0
            else f"Run analysis ({uncached_n} new)"
        )
        if st.button(run_label, disabled=run_disabled, use_container_width=True):
            st.session_state.run_analysis = True

        st.divider()

        # ── Data source (collapsible) ─────────────────────────────────────────
        with st.expander("📂 Data Source", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload feedback CSV",
                type=["csv"],
                key="csv_upload",
                help=(
                    "Expected columns: response_id, nps_score, feedback_text, "
                    "response_date, member_location. "
                    "Leave empty to use the default dataset."
                ),
            )
            if uploaded_file is not None:
                st.caption(f"✅ Using: **{uploaded_file.name}**")
            else:
                st.caption("Using default dataset.")

        st.divider()

        # ── Date range ────────────────────────────────────────────────────────
        st.markdown("**Filters**")
        valid_dates = df_raw["response_date"].dropna()
        min_d = valid_dates.min().date()
        max_d = valid_dates.max().date()

        from datetime import timedelta as _td
        default_start = max(min_d, max_d - _td(days=29))  # last 30 days inclusive
        date_range = st.date_input(
            "Date range",
            value=(default_start, max_d),
            min_value=min_d,
            max_value=max_d,
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, max_d

        # ── Location ──────────────────────────────────────────────────────────
        all_locs = sorted(df_raw["member_location"].dropna().unique().tolist())
        selected_locs = st.multiselect(
            "Locations",
            options=all_locs,
            default=all_locs,
            placeholder="All locations",
        )
        if not selected_locs:
            selected_locs = all_locs

        # ── NPS segment ───────────────────────────────────────────────────────
        selected_segments = st.multiselect(
            "NPS segment",
            options=["Promoter", "Passive", "Detractor"],
            default=["Promoter", "Passive", "Detractor"],
            placeholder="All segments",
        )
        if not selected_segments:
            selected_segments = ["Promoter", "Passive", "Detractor"]

        st.divider()
        st.caption(
            f"Data: {min_d.strftime('%b %d, %Y')} – {max_d.strftime('%b %d, %Y')}"
            f"\n\n{total_rows} total responses · 17 locations"
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = (
        (df_raw["response_date"].dt.date >= start_d)
        & (df_raw["response_date"].dt.date <= end_d)
        & (df_raw["member_location"].isin(selected_locs))
        & (df_raw["nps_segment"].isin(selected_segments + ["Unknown"]))
    )
    filtered = df_raw[mask].copy()

    return filtered, {
        "api_key": api_key,
        "start_d": start_d,
        "end_d": end_d,
        "locations": selected_locs,
        "segments": selected_segments,
        "uploaded_file": uploaded_file,
    }


# ── Analysis runner ───────────────────────────────────────────────────────────

def maybe_run_analysis(df_raw: pd.DataFrame, api_key: str) -> None:
    if not st.session_state.get("run_analysis"):
        return
    st.session_state.run_analysis = False

    progress_bar = st.progress(0, text="Starting analysis…")

    def _progress(completed: int, total: int) -> None:
        pct = int(completed / total * 100) if total else 100
        progress_bar.progress(pct / 100, text=f"Analysing… {completed}/{total} responses")

    with st.spinner("Running AI analysis — this may take a few minutes on first run…"):
        run_analysis(df_raw, api_key, CACHE_PATH, progress_callback=_progress)

    progress_bar.empty()
    st.success("Analysis complete. Refreshing…")
    st.cache_data.clear()
    st.rerun()


# ── Helper: NPS color ─────────────────────────────────────────────────────────

def _nps_color(nps: float | None) -> str:
    if nps is None:
        return MEDIUM_GRAY
    if nps >= 30:
        return "#22C55E"
    if nps >= 0:
        return WARNING_BORDER
    return CRITICAL_BORDER


def _delta_str(val: float | None, prefix: str = "") -> str:
    if val is None:
        return ""
    sign = "+" if val > 0 else ""
    return f"{prefix}{sign}{val}"


# ── Tab renderers ─────────────────────────────────────────────────────────────

def render_briefing_tab(
    df: pd.DataFrame,
    all_alerts: list[dict],
    dev_df: pd.DataFrame,
    persist_df: pd.DataFrame,
    df_full: pd.DataFrame,
) -> None:
    """Combined Briefing + Overview tab.
    df       = sidebar-filtered data (KPI strip, daily drill-down, NPS dist, location sections)
    df_full  = full enriched dataset, unfiltered — used only for the two 6-month trend charts
    """
    if df.empty:
        st.info("No responses match the current filters.")
        return

    from config import ALERT_EXCLUDE_CATEGORIES as _EXCL, OPERATIONAL_CATEGORIES as _OP_CATS, SUBCATEGORY_MAP as _SUBCAT_MAP

    # ── 1. KPI STRIP ─────────────────────────────────────────────────────────
    nps = compute_nps(df)
    avg_sev = df["severity_score"].mean() if "severity_score" in df.columns else None

    # ── Prior-period comparison ───────────────────────────────────────────────
    # Compare the selected range against the immediately preceding equivalent-
    # length period, sliced from the full unfiltered dataset.
    _prior_nps_delta: float | None = None
    _prior_vol_delta: int | None = None
    _prior_prom_delta: float | None = None
    _prior_det_delta: float | None = None
    _prior_sev_delta: float | None = None
    _delta_label = "vs prior period"
    _comparison_note = ""

    if (
        not df.empty
        and "response_date" in df.columns
        and df_full is not None
        and not df_full.empty
        and "response_date" in df_full.columns
    ):
        _cur_min = pd.Timestamp(df["response_date"].min()).normalize()
        _cur_max = pd.Timestamp(df["response_date"].max()).normalize()
        _range_days = (_cur_max - _cur_min).days + 1

        _prior_end = _cur_min - pd.Timedelta(days=1)
        _prior_start = _prior_end - pd.Timedelta(days=_range_days - 1)

        # For exactly-7-day ranges, use week label if available; otherwise
        # always use "vs prior period" for the inline delta string.
        if _range_days == 7 and "week_label" in df_full.columns:
            _pw_df = df_full[
                (df_full["response_date"] >= _prior_start)
                & (df_full["response_date"] <= _prior_end)
            ]
            if not _pw_df.empty and _pw_df["week_label"].notna().any():
                _delta_label = f"vs {_pw_df['week_label'].dropna().iloc[0]}"

        # Full comparison dates shown in the caption below the strip
        _period_noun = (
            f"{_range_days}-day period"
            if _range_days != 7
            else "week"
        )
        _comparison_note = (
            f"Deltas compare against **{_prior_start.strftime('%b %d, %Y')} – "
            f"{_prior_end.strftime('%b %d, %Y')}** "
            f"(preceding {_period_noun})."
        )

        _df_prior = df_full[
            (df_full["response_date"] >= _prior_start)
            & (df_full["response_date"] <= _prior_end)
        ].copy()

        if not _df_prior.empty:
            _prior_nps_obj = compute_nps(_df_prior)
            _prior_sev = (
                _df_prior["severity_score"].mean()
                if "severity_score" in _df_prior.columns
                else None
            )
            if nps["nps"] is not None and _prior_nps_obj["nps"] is not None:
                _prior_nps_delta = round(
                    float(nps["nps"]) - float(_prior_nps_obj["nps"]), 1
                )
            if avg_sev is not None and _prior_sev is not None:
                _prior_sev_delta = round(float(avg_sev) - float(_prior_sev), 2)
            _prior_vol_delta = int(nps["total"]) - int(_prior_nps_obj["total"])
            if (
                nps.get("promoter_pct") is not None
                and _prior_nps_obj.get("promoter_pct") is not None
            ):
                _prior_prom_delta = round(
                    float(nps["promoter_pct"]) - float(_prior_nps_obj["promoter_pct"]), 1
                )
            if (
                nps.get("detractor_pct") is not None
                and _prior_nps_obj.get("detractor_pct") is not None
            ):
                _prior_det_delta = round(
                    float(nps["detractor_pct"]) - float(_prior_nps_obj["detractor_pct"]), 1
                )

    c1, c2, c3, c4, c5 = st.columns(5)
    nps_val = nps["nps"]
    # NPS: higher is better → default delta_color (green for positive) is correct
    c1.metric(
        "NPS Score",
        f"{nps_val:.0f}" if nps_val is not None else "—",
        delta=(
            f"{_prior_nps_delta:+.1f} pts {_delta_label}"
            if _prior_nps_delta is not None
            else None
        ),
    )
    # Total responses: neutral
    c2.metric(
        "Total Responses",
        nps["total"],
        delta=(
            f"{_prior_vol_delta:+d} {_delta_label}"
            if _prior_vol_delta is not None
            else None
        ),
        delta_color="off",
    )
    # Promoter rate: higher is better → default delta_color is correct
    c3.metric(
        "Promoters",
        f"{nps['promoter_pct']:.0f}%",
        delta=(
            f"{_prior_prom_delta:+.1f}pp {_delta_label}"
            if _prior_prom_delta is not None
            else None
        ),
        help="Score 9–10",
    )
    # Detractor rate: lower is better → inverse coloring
    c4.metric(
        "Detractors",
        f"{nps['detractor_pct']:.0f}%",
        delta=(
            f"{_prior_det_delta:+.1f}pp {_delta_label}"
            if _prior_det_delta is not None
            else None
        ),
        delta_color="inverse",
        help="Score 0–6 · lower is better · negative delta (↓) = improvement",
    )
    # Avg severity: lower is better → inverse coloring
    c5.metric(
        "Avg Severity",
        f"{avg_sev:.1f}" if avg_sev else "—",
        delta=(
            f"{_prior_sev_delta:+.2f} {_delta_label}"
            if _prior_sev_delta is not None
            else None
        ),
        delta_color="inverse",
    )

    # Helper text: selected range + preceding comparison period
    if not df.empty and "response_date" in df.columns:
        _kpi_start = df["response_date"].min().strftime("%b %d, %Y")
        _kpi_end = df["response_date"].max().strftime("%b %d, %Y")
        _range_note = (
            f"KPI metrics reflect the selected date range: **{_kpi_start} – {_kpi_end}**."
        )
        st.caption(
            f"{_range_note} {_comparison_note}"
            if _comparison_note
            else f"{_range_note} Deltas compare against the immediately preceding "
            "equivalent-length period."
        )

    # ── 2. DAILY DRILL-DOWN ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        section_header("Daily Drill-Down", "Responses and criticals for a specific date"),
        unsafe_allow_html=True,
    )

    available_days = sorted(df["day_label"].dropna().unique().tolist(), reverse=True)
    if not available_days:
        st.info("No dated responses in the current filter.")
    else:
        selected_day = st.selectbox(
            "Select date",
            options=available_days,
            index=0,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%A, %B %d, %Y"),
        )
        digest = daily_digest(df, selected_day)

        if not digest["has_data"]:
            st.info(f"No responses on {selected_day}.")
        else:
            # Day-level KPI strip — NPS shown with sample size caveat
            day_nps = digest["nps_snapshot"]["nps"] if digest["nps_snapshot"] else None
            day_n = digest["total_responses"]
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Responses", day_n)
            d2.metric(
                "New Criticals",
                digest["new_critical_count"],
                delta=None if digest["new_critical_count"] == 0 else "⚠ Action needed",
                delta_color="inverse",
            )
            d3.metric(
                "Avg Severity",
                f"{digest['avg_severity']:.1f}" if digest["avg_severity"] else "—",
            )
            d4.metric("Churn Risk", digest["churn_risk_count"])
            d5.metric(
                "NPS (day)",
                f"{day_nps:.0f}" if day_nps is not None else "—",
                help=f"Based on {day_n} response{'s' if day_n != 1 else ''}. "
                "Daily NPS is unreliable at low volumes — use as a directional signal only.",
            )

            col_l, col_r = st.columns([1, 1])

            with col_l:
                # ── Issue distribution: all categories, stacked by subcategory ──
                st.markdown(
                    section_header(
                        "Issue Distribution Today",
                        "All issue categories · stacked by subcategory",
                    ),
                    unsafe_allow_html=True,
                )
                day_df = df[df["day_label"] == selected_day].copy()
                op_day = day_df[~day_df["issue_category"].isin(_EXCL)]

                # Operational categories to always display (excluding excl set)
                _op_cats_display = [c for c in _OP_CATS if c not in _EXCL]

                # Count responses by category + subcategory
                if not op_day.empty and "issue_subcategory" in op_day.columns:
                    actual = (
                        op_day.groupby(["issue_category", "issue_subcategory"])["response_id"]
                        .count()
                        .reset_index(name="count")
                    )
                else:
                    actual = pd.DataFrame(
                        columns=["issue_category", "issue_subcategory", "count"]
                    )

                # Ensure every operational category appears on the y-axis even with 0
                # by adding a zero-count placeholder row for any missing category
                existing_cats = set(actual["issue_category"].unique())
                placeholders = [
                    {"issue_category": c, "issue_subcategory": "", "count": 0}
                    for c in _op_cats_display if c not in existing_cats
                ]
                if placeholders:
                    actual = pd.concat(
                        [actual, pd.DataFrame(placeholders)], ignore_index=True
                    )

                # Sort: highest total at top of horizontal chart (ascending order)
                cat_totals = actual.groupby("issue_category")["count"].sum()
                cat_order = (
                    pd.Series({c: cat_totals.get(c, 0) for c in _op_cats_display})
                    .sort_values(ascending=True)
                    .index.tolist()
                )

                # Build a stable color map for all possible subcategories so
                # colors are consistent across date changes
                _palette = px.colors.qualitative.Safe + px.colors.qualitative.Pastel
                all_subs = [
                    sub
                    for c in _op_cats_display
                    for sub in _SUBCAT_MAP.get(c, [])
                ]
                _sub_colors = {
                    sub: _palette[i % len(_palette)]
                    for i, sub in enumerate(all_subs)
                }
                _sub_colors[""] = "rgba(0,0,0,0)"  # placeholder = invisible

                fig_day_dist = px.bar(
                    actual,
                    x="count",
                    y="issue_category",
                    color="issue_subcategory",
                    orientation="h",
                    barmode="stack",
                    color_discrete_map=_sub_colors,
                    category_orders={"issue_category": cat_order},
                    labels={
                        "count": "Responses",
                        "issue_category": "",
                        "issue_subcategory": "Subcategory",
                    },
                )
                # Hide the invisible placeholder from the legend
                for trace in fig_day_dist.data:
                    if trace.name == "":
                        trace.showlegend = False
                fig_day_dist.update_layout(
                    height=max(280, len(cat_order) * 38 + 80),
                    margin=dict(t=10, b=10, l=0, r=20),
                    plot_bgcolor=WHITE,
                    paper_bgcolor=WHITE,
                    legend=dict(
                        orientation="v",
                        x=1.01,
                        y=1.0,
                        font=dict(size=10),
                        title=dict(text="Subcategory", font=dict(size=10)),
                    ),
                    xaxis=dict(title="Responses"),
                )
                st.plotly_chart(fig_day_dist, use_container_width=True)

            with col_r:
                st.markdown(
                    section_header(
                        "Critical Responses",
                        f"{digest['new_critical_count']} severity-5 or high-churn responses",
                    ),
                    unsafe_allow_html=True,
                )
                crits = digest.get("critical_responses", [])
                if crits:
                    for c in crits:
                        st.markdown(
                            alert_box(
                                "Critical",
                                f"{c['response_id']} · {c['member_location']} · NPS {c.get('nps_score', '—')}",
                                f"<em>{c.get('feedback_text', '')[:220]}…</em>",
                                meta=f"Category: {c.get('category', '—')} · "
                                f"Severity: {c.get('severity_score')} · {c.get('trigger', '')}",
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        alert_box(
                            "",
                            "No criticals today",
                            "No severity-5 or high-churn responses on this date.",
                            "",
                        ),
                        unsafe_allow_html=True,
                    )

            quotes = digest.get("notable_quotes", [])
            if quotes:
                st.markdown(
                    section_header(
                        "Notable Quotes",
                        "Responses with AI-assigned severity ≥ 4, sorted highest first",
                    ),
                    unsafe_allow_html=True,
                )
                for q in quotes:
                    sev_color = SEVERITY_COLORS.get(q["severity"], MEDIUM_GRAY)
                    st.markdown(
                        f'<div style="border-left:3px solid {sev_color};padding:10px 14px;'
                        f"margin-bottom:8px;background:{CREAM};border-radius:0 6px 6px 0\">"
                        f'<span style="font-size:0.75rem;color:{MEDIUM_GRAY}">'
                        f"{q['response_id']} · {q['location']} · NPS {q.get('nps_score', '—')} · "
                        f'<strong style="color:{sev_color}">Severity {q["severity"]}</strong> · '
                        f"{q.get('category', '')}</span><br>"
                        f'<span style="font-size:0.875rem;color:{DARK}">{q["text"][:280]}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── 3 & 4: Build 6-month unfiltered slice for trend charts ───────────────
    six_mo_cutoff = df_full["response_date"].max() - pd.Timedelta(weeks=26)
    df_6mo = df_full[df_full["response_date"] >= six_mo_cutoff].copy()
    six_mo_label = "Last 6 months · unaffected by date / location filters"

    # ── 3. NPS DISTRIBUTION + RESPONSE VOLUME ────────────────────────────────
    st.markdown("---")
    ch_l, ch_r = st.columns([1, 1])

    with ch_l:
        st.markdown(
            section_header("NPS Score Distribution", "Response count by score"),
            unsafe_allow_html=True,
        )
        scored = df[df["nps_score"].notna()].copy()
        scored["nps_score"] = scored["nps_score"].astype(int)
        score_counts = (
            scored.groupby("nps_score").size().reset_index(name="count").sort_values("nps_score")
        )
        score_counts["color"] = score_counts["nps_score"].apply(
            lambda s: "#22C55E" if s >= 9 else "#6B7280" if s >= 7 else CRITICAL_BORDER
        )
        fig_dist = go.Figure(
            go.Bar(
                x=score_counts["nps_score"],
                y=score_counts["count"],
                marker_color=score_counts["color"].tolist(),
                hovertemplate="Score %{x}: %{y} responses<extra></extra>",
            )
        )
        fig_dist.update_layout(
            height=300,
            xaxis=dict(title="NPS Score", tickmode="linear", dtick=1),
            yaxis=dict(title="Responses"),
            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,
            margin=dict(t=10, b=40, l=40, r=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with ch_r:
        st.markdown(
            section_header("Response Volume Over Time", six_mo_label),
            unsafe_allow_html=True,
        )
        weekly_6mo = weekly_summary(df_6mo)
        if not weekly_6mo.empty:
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    x=weekly_6mo["week_label"],
                    y=weekly_6mo["response_count"],
                    mode="lines+markers",
                    line=dict(color=PRIMARY, width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>%{y} responses<extra></extra>",
                )
            )
            fig_vol.update_layout(
                height=300,
                xaxis=dict(title="", tickangle=-45),
                yaxis=dict(title="Responses"),
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                margin=dict(t=10, b=60, l=40, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    # ── 4. SENTIMENT OVER TIME + ISSUE VELOCITY (side by side) ─────────────
    st.markdown("---")
    _col_sent, _col_vel = st.columns([1, 1])

    with _col_sent:
        st.markdown(
            section_header("Sentiment Distribution Over Time", six_mo_label),
            unsafe_allow_html=True,
        )
        sent_time = sentiment_over_time(df_6mo)
        if not sent_time.empty:
            pivot = sent_time.pivot_table(
                index="week_label", columns="sentiment", values="count", fill_value=0
            ).reset_index()
            sent_colors = {
                "Positive": "#22C55E",
                "Neutral": "#6B7280",
                "Mixed": WARNING_BORDER,
                "Negative": CRITICAL_BORDER,
            }
            fig_sent = go.Figure()
            for sent in ["Negative", "Mixed", "Neutral", "Positive"]:
                if sent in pivot.columns:
                    fig_sent.add_trace(
                        go.Bar(
                            name=sent,
                            x=pivot["week_label"],
                            y=pivot[sent],
                            marker_color=sent_colors.get(sent, MEDIUM_GRAY),
                            hovertemplate=f"{sent}: %{{y}}<extra></extra>",
                        )
                    )
            fig_sent.update_layout(
                barmode="stack",
                height=340,
                xaxis=dict(title="", tickangle=-45),
                yaxis=dict(title="Responses"),
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                margin=dict(t=10, b=60, l=40, r=10),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    with _col_vel:
        st.markdown(
            section_header("Issue Velocity", "Week-over-week rate change by category"),
            unsafe_allow_html=True,
        )
        _vel_rates = weekly_category_rates(df)
        _vel = issue_velocity(_vel_rates if not _vel_rates.empty else weekly_category_rates(df))
        if not _vel.empty:
            _vel_latest = (
                _vel[_vel["week_start"].notna()]
                .sort_values("week_start")
                .groupby("issue_category")
                .last()
                .reset_index()
                [["issue_category", "rate_change", "pct_change"]]
                .dropna(subset=["rate_change"])
                .sort_values("rate_change", ascending=False)
                .head(10)
            )
            _vel_latest["rate_change_pct"] = _vel_latest["rate_change"] * 100
            _vel_latest["color"] = _vel_latest["rate_change"].apply(
                lambda v: CRITICAL_BORDER if v > 0 else "#22C55E"
            )
            fig_vel = go.Figure(go.Bar(
                x=_vel_latest["rate_change_pct"],
                y=_vel_latest["issue_category"],
                orientation="h",
                marker_color=_vel_latest["color"].tolist(),
                hovertemplate="%{y}: %{x:+.1f}pp<extra></extra>",
            ))
            fig_vel.update_layout(
                height=340,
                xaxis=dict(title="Rate change (pp)", ticksuffix="pp"),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                margin=dict(t=10, b=40, l=0, r=20),
            )
            st.plotly_chart(fig_vel, use_container_width=True)

    # ── 5. ISSUE DISTRIBUTION (filtered period) ─────────────────────────────
    st.markdown("---")
    st.markdown(
        section_header(
            "Issue Distribution",
            "Filtered period · all operational categories stacked by subcategory",
        ),
        unsafe_allow_html=True,
    )
    _op_cats_display = [c for c in _OP_CATS if c not in _EXCL]
    _op_df = df[~df["issue_category"].isin(_EXCL)] if "issue_category" in df.columns else pd.DataFrame()

    if not _op_df.empty and "issue_subcategory" in _op_df.columns:
        _dist_actual = (
            _op_df.groupby(["issue_category", "issue_subcategory"])["response_id"]
            .count()
            .reset_index(name="count")
        )
    else:
        _dist_actual = pd.DataFrame(columns=["issue_category", "issue_subcategory", "count"])

    _existing = set(_dist_actual["issue_category"].unique())
    _placeholders = [
        {"issue_category": c, "issue_subcategory": "", "count": 0}
        for c in _op_cats_display if c not in _existing
    ]
    if _placeholders:
        _dist_actual = pd.concat([_dist_actual, pd.DataFrame(_placeholders)], ignore_index=True)

    _cat_totals = _dist_actual.groupby("issue_category")["count"].sum()
    _cat_order = (
        pd.Series({c: _cat_totals.get(c, 0) for c in _op_cats_display})
        .sort_values(ascending=True)
        .index.tolist()
    )

    _palette = px.colors.qualitative.Safe + px.colors.qualitative.Pastel
    _all_subs = [sub for c in _op_cats_display for sub in _SUBCAT_MAP.get(c, [])]
    _sub_colors = {sub: _palette[i % len(_palette)] for i, sub in enumerate(_all_subs)}
    _sub_colors[""] = "rgba(0,0,0,0)"

    fig_brief_dist = px.bar(
        _dist_actual,
        x="count",
        y="issue_category",
        color="issue_subcategory",
        orientation="h",
        barmode="stack",
        color_discrete_map=_sub_colors,
        category_orders={"issue_category": _cat_order},
        labels={"count": "Responses", "issue_category": "", "issue_subcategory": "Subcategory"},
    )
    for trace in fig_brief_dist.data:
        if trace.name == "":
            trace.showlegend = False
    fig_brief_dist.update_layout(
        height=max(280, len(_cat_order) * 38 + 80),
        margin=dict(t=10, b=10, l=0, r=20),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        legend=dict(
            orientation="v", x=1.01, y=1.0,
            font=dict(size=10),
            title=dict(text="Subcategory", font=dict(size=10)),
        ),
        xaxis=dict(title="Responses"),
    )
    st.plotly_chart(fig_brief_dist, use_container_width=True)

    # ── 6. LOCATION ISSUES REQUIRING INVESTIGATION ───────────────────────────
    if not dev_df.empty and "is_flagged" in dev_df.columns:
        flagged_locs = dev_df[dev_df["is_flagged"]].copy()
        if not flagged_locs.empty:
            st.markdown("---")
            st.markdown(
                section_header(
                    "📍 Location Issues Requiring Investigation",
                    "Locations where a specific issue rate is ≥50% above the network average "
                    "(min 3 responses)",
                ),
                unsafe_allow_html=True,
            )
            ncols = min(3, len(flagged_locs))
            loc_cols = st.columns(ncols)
            for i, (_, row) in enumerate(flagged_locs.head(3).iterrows()):
                dev_pct = int(row["deviation_score"] * 100)
                sev_color = CRITICAL_BORDER if dev_pct > 100 else WARNING_BORDER
                with loc_cols[i]:
                    st.markdown(
                        f'<div style="background:{CREAM};border:1px solid {sev_color};'
                        f"border-top:3px solid {sev_color};border-radius:8px;"
                        f'padding:14px 16px;margin-bottom:12px">'
                        f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                        f"{row['member_location']}</div>"
                        f'<div style="color:{sev_color};font-weight:600;font-size:0.8rem;'
                        f'margin-top:3px">{row["issue_category"]}</div>'
                        f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                        f'<span style="font-size:1.4rem;font-weight:700;color:{sev_color}">'
                        f"+{dev_pct}%</span> above network average</div>"
                        f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:6px">'
                        f"{row['loc_count']} responses · "
                        f"Location rate: {row['loc_rate'] * 100:.1f}% · "
                        f"Network avg: {row['network_rate'] * 100:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── 7. PERSISTENT LOCATION ISSUES ────────────────────────────────────────
    if not persist_df.empty:
        st.markdown(
            section_header(
                "🔁 Persistent Location Issues",
                "Problems ranking in the top 3 for their location for multiple consecutive weeks "
                "— these are systemic, not one-off",
            ),
            unsafe_allow_html=True,
        )
        p_cols = st.columns(2)
        for i, (_, row) in enumerate(persist_df.head(4).iterrows()):
            with p_cols[i % 2]:
                st.markdown(
                    alert_box(
                        "Warning",
                        f"{row['member_location']} — {row['issue_category']}",
                        f"Ranked as a top-3 issue at this location for "
                        f"<strong>{row['max_consecutive_weeks']} consecutive weeks</strong>. "
                        "This pattern indicates a systemic operational problem, not random variation.",
                        meta=f"Total weeks in top-3: {row['weeks_in_top3']}",
                    ),
                    unsafe_allow_html=True,
                )

    # ── 8. SUGGESTED INVESTIGATIONS & ACTIONS ────────────────────────────────
    if "suggested_operational_action" in df.columns:
        st.markdown("---")
        st.markdown(
            section_header(
                "🔧 Suggested Investigations & Actions",
                "Recommended next steps based on the highest-severity responses — "
                "generated by AI analysis",
            ),
            unsafe_allow_html=True,
        )
        action_df = (
            df[
                df["suggested_operational_action"].str.strip().ne("")
                & df["severity_score"].notna()
                & ~df["issue_category"].isin({"No Comment", "Positive Feedback"})
            ]
            .sort_values("severity_score", ascending=False)
            .drop_duplicates(subset=["suggested_operational_action"])
            .head(6)
        )
        if not action_df.empty:
            for _, row in action_df.iterrows():
                sev = int(row["severity_score"])
                sev_color = SEVERITY_COLORS.get(sev, MEDIUM_GRAY)
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:14px;'
                    f"padding:12px 0;border-bottom:1px solid {LIGHT_GRAY}\">"
                    f'<div style="min-width:34px;height:34px;border-radius:50%;'
                    f"background:{sev_color};color:white;font-weight:700;font-size:0.8rem;"
                    f'display:flex;align-items:center;justify-content:center;flex-shrink:0">'
                    f"SEV {sev}</div>"
                    f"<div>"
                    f'<div style="font-size:0.88rem;color:{DARK};font-weight:500">'
                    f"{row['suggested_operational_action']}</div>"
                    f'<div style="font-size:0.73rem;color:{MEDIUM_GRAY};margin-top:3px">'
                    f"{row.get('issue_category', '')} · {row.get('member_location', '')} · "
                    f"Owner: {row.get('recommended_owner', '')} · {row.get('response_id', '')}"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )


def render_top_issues_tab(df: pd.DataFrame) -> None:
    issues = top_issues(df, n=5)
    if not issues:
        st.info("No operational issue data available. Run analysis first.")
        return

    # Date range context for the metric label
    date_ctx = ""
    if not df.empty and "response_date" in df.columns:
        d0 = df["response_date"].min().strftime("%b %d")
        d1 = df["response_date"].max().strftime("%b %d, %Y")
        date_ctx = f" · {d0} – {d1}"

    st.markdown(
        section_header(
            "Top 5 Issue Categories",
            f"Ranked by **avg severity score** in the selected period{date_ctx}. "
            "Excludes Positive Feedback and No Comment. Trend arrow = week-over-week rate change.",
        ),
        unsafe_allow_html=True,
    )

    # Compute WoW velocity so we can show trend arrows per category
    weekly_rates = weekly_category_rates(df)
    vel_latest: dict[str, float] = {}
    if not weekly_rates.empty:
        vel = issue_velocity(weekly_rates)
        if not vel.empty and "week_start" in vel.columns:
            vel_latest = (
                vel[vel["week_start"].notna()]
                .sort_values("week_start")
                .groupby("issue_category")
                .last()["rate_change"]
                .to_dict()
            )

    def _trend_arrow(cat: str) -> str:
        rc = vel_latest.get(cat, 0) or 0
        if rc > 0.02:
            return "↑ Rising"
        if rc < -0.02:
            return "↓ Falling"
        return "→ Stable"

    # Summary table
    table_rows = [
        {
            "Category": i["category"],
            "# Responses": i["count"],
            "% of Total": f"{i['pct_of_total']:.1f}%",
            "WoW Trend": _trend_arrow(i["category"]),
            "Avg Severity": f"{i['avg_severity']:.1f}",
            "Detractor %": f"{i['detractor_pct']:.0f}%",
            "Churn Risk": i["churn_risk_count"],
            "Owner": i["recommended_owner"],
        }
        for i in issues
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Per-category drill-down
    for issue in issues:
        cat = issue["category"]
        sev_color = SEVERITY_COLORS.get(round(issue["avg_severity"]), MEDIUM_GRAY)
        with st.expander(
            f"**{cat}** — {issue['count']} responses · Avg severity {issue['avg_severity']:.1f} · {issue['detractor_pct']:.0f}% detractors"
        ):
            exp_l, exp_r = st.columns([1, 1])

            with exp_l:
                # Subcategory donut
                subs = issue["subcategories"]
                if subs:
                    sub_df = (
                        pd.DataFrame([{"subcategory": k, "count": v} for k, v in subs.items()])
                        .sort_values("count", ascending=False)
                    )
                    _donut_palette = px.colors.qualitative.Safe + px.colors.qualitative.Pastel
                    _donut_colors = [
                        _donut_palette[i % len(_donut_palette)]
                        for i in range(len(sub_df))
                    ]
                    fig = go.Figure(go.Pie(
                        labels=sub_df["subcategory"],
                        values=sub_df["count"],
                        hole=0.5,
                        marker=dict(colors=_donut_colors),
                        textinfo="percent",
                        hovertemplate="%{label}<br>%{value} responses (%{percent})<extra></extra>",
                        sort=False,
                    ))
                    fig.update_layout(
                        title=dict(text="Subcategory breakdown", font=dict(size=13)),
                        height=280,
                        margin=dict(t=40, b=10, l=10, r=10),
                        paper_bgcolor=WHITE,
                        legend=dict(
                            orientation="v",
                            x=1.02,
                            y=0.5,
                            font=dict(size=10),
                        ),
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Example quotes
                st.markdown(
                    f'<div style="font-weight:600;font-size:0.85rem;color:{DARK};'
                    f'margin-bottom:8px">Example Quotes</div>',
                    unsafe_allow_html=True,
                )
                for q in issue["example_quotes"]:
                    s = q["severity"]
                    sc = SEVERITY_COLORS.get(s, MEDIUM_GRAY)
                    st.markdown(
                        f"""<div style="border-left:3px solid {sc};padding:8px 12px;
                        margin-bottom:8px;background:{CREAM};border-radius:0 6px 6px 0">
                        <span style="font-size:0.72rem;color:{MEDIUM_GRAY}">
                          {q['response_id']} · {q['location']} · NPS {q.get('nps_score','—')} ·
                          <strong style="color:{sc}">Sev {s}</strong>
                        </span><br>
                        <span style="font-size:0.85rem;color:{DARK}">{q['text'][:260]}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.caption(f"**Owner:** {issue['recommended_owner']}")

            with exp_r:
                # Per-subcategory root cause + suggested action
                sub_details = issue.get("subcategory_details", [])
                if sub_details:
                    st.markdown(
                        f'<div style="font-weight:600;font-size:0.85rem;color:{DARK};'
                        f'margin-bottom:6px">Root Causes & Suggested Actions</div>',
                        unsafe_allow_html=True,
                    )
                    for sd in sub_details:
                        if not sd["root_cause"] and not sd["suggested_action"]:
                            continue
                        st.markdown(
                            f'<div style="background:{CREAM};border-radius:6px;'
                            f'padding:10px 12px;margin-bottom:8px;">'
                            f'<div style="font-weight:600;font-size:0.78rem;color:{DARK};'
                            f'margin-bottom:4px">{sd["subcategory"]} '
                            f'<span style="font-weight:400;color:{MEDIUM_GRAY}">· {sd["count"]} responses</span></div>'
                            + (
                                f'<div style="font-size:0.78rem;color:{DARK};margin-bottom:3px">'
                                f'<span style="color:{MEDIUM_GRAY};font-weight:600">Root cause: </span>'
                                f'{sd["root_cause"]}</div>'
                                if sd["root_cause"] else ""
                            )
                            + (
                                f'<div style="font-size:0.78rem;color:{DARK}">'
                                f'<span style="color:{MEDIUM_GRAY};font-weight:600">Action: </span>'
                                f'{sd["suggested_action"]}</div>'
                                if sd["suggested_action"] else ""
                            )
                            + "</div>",
                            unsafe_allow_html=True,
                        )


def render_control_charts_tab(df: pd.DataFrame, chart_df: pd.DataFrame) -> None:
    from config import OPERATIONAL_CATEGORIES

    st.markdown(
        section_header(
            "Statistical Process Control",
            "Weekly p-charts per issue category · UCL = p̄ + 3σ · Weeks above UCL flagged in red",
        ),
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ How to read this chart", expanded=False):
        st.markdown(
            """
**What is a p-chart?**
A p-chart (proportion chart) is a statistical process control tool that tracks whether a process is behaving consistently over time. Here, each chart monitors the weekly *rate* of a given issue category — the share of all responses that week mentioning that issue.

**Key elements:**
- **Centerline (p̄)** — the weighted average issue rate across all historical weeks. This is the baseline "normal" level for that category.
- **UCL / LCL** (Upper / Lower Control Limit) — set at ±3 standard deviations from the centerline. Under normal variation, ~99.7% of weeks should fall within these bounds.
- **Red points** — weeks where the issue rate exceeded the UCL. These are statistically unlikely under normal conditions and warrant investigation.
- **Run above centerline** — consecutive weeks above p̄ can signal a sustained shift even if no single week breaks the UCL.

**Important notes:**
- Control limits are calculated from the **full dataset** (all available weeks), not the sidebar date filter. This ensures baselines are stable and not distorted by short windows.
- A week above the UCL is a signal to *investigate*, not a confirmed problem. It means the rate was unusually high — the cause may be operational, seasonal, or a one-off event.
- A week below the LCL (issue rate unusually low) is generally a positive signal.
            """
        )

    # ── Signal directory ──────────────────────────────────────────────────────
    import io as _sio
    _sig_json = _compute_signal_directory(
        df.to_json(orient="split", date_format="iso"),
        chart_df.to_json(orient="split", date_format="iso"),
    )
    _sig_df = pd.read_json(_sio.StringIO(_sig_json), orient="split")

    # "Active" = signal occurred within the last 30 days of available data
    if not _sig_df.empty and "latest_signal" in _sig_df.columns:
        _data_max = pd.Timestamp(df["response_date"].max()).normalize()
        _cutoff = _data_max - pd.Timedelta(days=30)

        def _week_to_date(wl):
            try:
                return pd.to_datetime(str(wl) + "-1", format="%G-W%V-%u")
            except Exception:
                return pd.NaT

        _sig_df["_latest_date"] = _sig_df["latest_signal"].apply(_week_to_date)
        _sig_df = _sig_df[_sig_df["_latest_date"] >= _cutoff].drop(columns=["_latest_date"])

    if _sig_df.empty:
        st.success("✅ No active out-of-control signals in the last 30 days of data.")
    else:
        _n = len(_sig_df)
        st.warning(
            f"**{_n} active out-of-control signal{'s' if _n != 1 else ''}** "
            f"(within 30 days of {_data_max.strftime('%b %d, %Y')}) — "
            "use the dropdowns below to drill into any combination."
        )
        st.dataframe(
            _sig_df.rename(columns={
                "issue_category": "Category",
                "location": "Location",
                "signal_weeks": "Signal Weeks",
                "latest_signal": "Latest Signal",
            }),
            use_container_width=True,
            hide_index=True,
            height=213,
            column_config={
                "Signal Weeks": st.column_config.NumberColumn(
                    "Signal Weeks",
                    help="Number of weeks (within the last 30 days of data) where the issue rate exceeded the UCL.",
                ),
                "Latest Signal": st.column_config.TextColumn(
                    "Latest Signal",
                    help="Most recent ISO week with an above-UCL observation.",
                ),
            },
        )

    col_sel, col_loc = st.columns([2, 2])
    with col_sel:
        selected_cat = st.selectbox(
            "Issue category",
            options=OPERATIONAL_CATEGORIES,
            index=0,
        )
    with col_loc:
        loc_options = ["All locations"] + sorted(df["member_location"].dropna().unique().tolist())
        selected_loc = st.selectbox("Location filter", options=loc_options, index=0)

    # active_chart_df tracks whichever dataset backs the visible chart,
    # so the spike summary always reflects what's shown.
    active_chart_df = chart_df
    spike_scope_label = "Across all locations"

    if selected_loc == "All locations":
        fig = build_control_chart(selected_cat, chart_df)
    else:
        loc_rates = location_weekly_rates(df, selected_cat, selected_loc)
        if loc_rates.empty:
            st.warning(f"No data found for **{selected_cat}** at **{selected_loc}**.")
            fig = build_control_chart(selected_cat, chart_df)
            st.caption("Showing network-level chart instead.")
        else:
            loc_chart = compute_control_limits(loc_rates)
            loc_chart = detect_run_above_centerline(loc_chart)

            if not loc_chart["has_limits"].any():
                # Insufficient location volume — overlay network-level limits as reference
                net_ref = (
                    chart_df[chart_df["issue_category"] == selected_cat][
                        ["week_label", "p_bar", "ucl", "lcl", "has_limits"]
                    ].copy()
                )
                loc_chart = loc_chart.drop(
                    columns=["p_bar", "ucl", "lcl", "has_limits"], errors="ignore"
                )
                loc_chart = loc_chart.merge(net_ref, on="week_label", how="left")
                loc_chart["above_ucl"] = (
                    loc_chart["issue_rate"] > loc_chart["ucl"].fillna(float("inf"))
                )
                st.caption(
                    "⚠ Insufficient weekly volume for location-specific limits "
                    "(need ≥12 weeks with n≥5). Network-level limits shown as reference."
                )

            fig = build_control_chart(selected_cat, loc_chart)
            fig.update_layout(
                title=dict(
                    text=f"<b>{selected_cat}</b> — {selected_loc} Weekly Issue Rate"
                )
            )
            active_chart_df = loc_chart
            spike_scope_label = selected_loc

    st.plotly_chart(fig, use_container_width=True)

    # Spike summary table — scoped to whatever dataset backs the visible chart
    spikes = spike_summary(active_chart_df)
    if not spikes.empty:
        st.markdown("---")
        st.markdown(
            section_header(
                "Out-of-Control Signals",
                f"Weeks where the issue rate exceeded the UCL — {spike_scope_label}",
            ),
            unsafe_allow_html=True,
        )
        all_spikes = spikes.copy()
        all_spikes["issue_rate_fmt"] = (all_spikes["issue_rate"] * 100).round(1).astype(str) + "%"
        all_spikes["ucl_fmt"] = (all_spikes["ucl"] * 100).round(1).astype(str) + "%"
        all_spikes["p_bar_fmt"] = (all_spikes["p_bar"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            all_spikes[["issue_category", "week_label", "issue_rate_fmt", "ucl_fmt", "p_bar_fmt"]]
            .rename(columns={
                "issue_category": "Category",
                "week_label": "Week",
                "issue_rate_fmt": "Rate",
                "ucl_fmt": "UCL",
                "p_bar_fmt": "Centerline",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No out-of-control signals detected.")


def render_alerts_tab(
    df: pd.DataFrame,
    all_alerts: list[dict],
    chart_df: pd.DataFrame,
) -> None:
    by_tier = alerts_by_tier(all_alerts)

    # Daily alerts
    day_alerts = alerts_for_day(df)
    if day_alerts:
        st.markdown(
            section_header(
                "Today's Critical Responses",
                f"{len(day_alerts)} severity-5 / high-churn response(s) on the most recent date",
            ),
            unsafe_allow_html=True,
        )
        for a in day_alerts:
            st.markdown(
                alert_box(
                    "Critical",
                    f"{a['response_id']} · {a['member_location']} · NPS {a.get('nps_score','—')}",
                    f"<em>{a.get('feedback_text','')[:200]}</em>",
                    meta=f"{a['trigger']} · Owner: {a['recommended_owner']}",
                ),
                unsafe_allow_html=True,
            )
        st.markdown("---")

    # Tier sections
    tier_config = [
        ("Critical", "🔴 Critical", "Require immediate action"),
        ("Warning", "🟡 Warning", "Investigate this week"),
        ("Watchlist", "🟠 Watchlist", "Monitor closely"),
        ("Location", "📍 Location Flags", "Location-specific systematic issues"),
    ]

    for tier_key, tier_label, tier_desc in tier_config:
        alerts = by_tier.get(tier_key, [])
        count = len(alerts)
        with st.expander(f"{tier_label} — {count} alert{'s' if count != 1 else ''} · {tier_desc}", expanded=(tier_key == "Critical")):
            if not alerts:
                st.caption("No alerts in this tier for the current period.")
            for a in alerts:
                locs = a.get("affected_locations", [])
                loc_str = ", ".join(locs[:4]) + ("…" if len(locs) > 4 else "")
                body = a.get("detail") or a.get("trigger", "")
                meta_parts = [
                    f"Responses: {a.get('response_count',0)}",
                    f"Avg severity: {a.get('avg_severity',0):.1f}",
                    f"Locations: {loc_str}",
                    f"Owner: {a.get('recommended_owner','')}",
                ]
                if a.get("deviation_score") is not None:
                    meta_parts.append(f"Deviation: {a['deviation_score']:.0%} above avg")
                st.markdown(
                    alert_box(
                        tier_key,
                        f"{a['category']}",
                        body[:300],
                        meta=" · ".join(meta_parts),
                    ),
                    unsafe_allow_html=True,
                )


def render_segments_tab(
    df: pd.DataFrame,
    health_df: pd.DataFrame,
    dev_df: pd.DataFrame,
) -> None:
    st.markdown(
        section_header("Member Segments", "NPS, severity, and issue profiles by location"),
        unsafe_allow_html=True,
    )

    tab_health, tab_dev = st.tabs(
        ["Location Health", "Issue Deviation"]
    )

    with tab_health:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            if not health_df.empty:
                # Color code by health score
                health_display = health_df.copy()
                health_display["Score"] = health_display["health_score"].apply(
                    lambda s: f"{s:.0f}/100"
                )
                health_display["Status"] = health_display["health_score"].apply(
                    lambda s: "🔴 At Risk" if s < 45 else "🟡 Monitor" if s < 60 else "🟢 Healthy"
                )
                st.dataframe(
                    health_display[["member_location", "Score", "Status", "avg_nps",
                                    "avg_severity", "detractor_rate", "high_severity_rate",
                                    "total_responses"]]
                    .rename(columns={
                        "member_location": "Location",
                        "avg_nps": "Avg NPS",
                        "avg_severity": "Avg Severity",
                        "detractor_rate": "Detractor Rate",
                        "high_severity_rate": "High Sev Rate",
                        "total_responses": "Responses",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Detractor Rate": st.column_config.NumberColumn(
                            "Detractor Rate",
                            help=(
                                "Share of responses with NPS 0–6 (Detractors) out of all "
                                "responses for that location. Higher = more dissatisfied members."
                            ),
                        ),
                        "High Sev Rate": st.column_config.NumberColumn(
                            "High Sev Rate",
                            help=(
                                "Share of responses at this location with a severity score of 4 or 5 "
                                "out of all responses for that location. These represent serious issues "
                                "involving trust, access, billing failures, or critical incidents. "
                                "A value of 0.35 means 35% of that location's responses were high-severity."
                            ),
                        ),
                    },
                )
        with col_r:
            nps_loc = nps_by_location(df)
            if not nps_loc.empty:
                fig = px.bar(
                    nps_loc.sort_values("nps"),
                    x="nps",
                    y="location",
                    orientation="h",
                    color="nps",
                    color_continuous_scale=["#EF4444", "#F59E0B", "#22C55E"],
                    title="NPS by Location",
                    labels={"nps": "NPS", "location": ""},
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor=WHITE,
                    paper_bgcolor=WHITE,
                    margin=dict(t=40, b=20, l=0, r=20),
                    coloraxis_showscale=False,
                )
                fig.add_vline(x=0, line_dash="dash", line_color=MEDIUM_GRAY, line_width=1)
                st.plotly_chart(fig, use_container_width=True)

    with tab_dev:
        if not dev_df.empty:
            # Heatmap first
            pivot = dev_df.pivot_table(
                index="member_location",
                columns="issue_category",
                values="deviation_score",
                fill_value=0,
            )
            if not pivot.empty:
                fig = px.imshow(
                    pivot,
                    color_continuous_scale=["#EFF6FF", "#DBEAFE", WARNING_BORDER, CRITICAL_BORDER],
                    title="Deviation Score Heatmap (location × category)",
                    labels={"color": "Deviation"},
                    aspect="auto",
                )
                fig.update_layout(
                    height=420,
                    margin=dict(t=50, b=10, l=100, r=20),
                    paper_bgcolor=WHITE,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Flagged table below
            flagged = dev_df[dev_df["is_flagged"]].copy()
            flagged["deviation_pct"] = (flagged["deviation_score"] * 100).round(1)
            if not flagged.empty:
                st.markdown(
                    f'<div style="color:{CRITICAL_BORDER};font-weight:600;font-size:0.85rem;'
                    f'margin-bottom:12px">⚠ {len(flagged)} location–category combinations flagged '
                    f'(≥50% above network average, n≥3)</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    flagged[["member_location", "issue_category", "loc_count",
                              "loc_rate", "network_rate", "deviation_pct"]]
                    .rename(columns={
                        "member_location": "Location",
                        "issue_category": "Category",
                        "loc_count": "Count",
                        "loc_rate": "Location Rate",
                        "network_rate": "Network Rate",
                        "deviation_pct": "Deviation (%)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Deviation (%)": st.column_config.NumberColumn(
                            "Deviation (%)",
                            help=(
                                "How much this location's issue rate differs from the network average. "
                                "Calculated as (location rate − network rate) ÷ network rate × 100. "
                                "A value of +75 means this location reports that issue 75% more often "
                                "than the network average. Flagged when ≥50% above average with at "
                                "least 3 responses."
                            ),
                        ),
                    },
                )
        else:
            st.info("Run analysis to see location deviation data.")



def render_responses_tab(df: pd.DataFrame) -> None:
    """Filterable table of every response with all AI-generated fields."""
    st.markdown(
        section_header(
            "All Responses",
            "Full response log with AI classification — filter by category, subcategory, "
            "sentiment, severity, or segment",
        ),
        unsafe_allow_html=True,
    )

    # ── Filter controls ───────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        from config import ISSUE_CATEGORIES as _ALL_CATS
        # Always show all defined categories so filtering by e.g. "Other" works
        # even when no matching rows exist in the current date/location filter.
        sel_cats = st.multiselect(
            "Category", options=_ALL_CATS, default=[], placeholder="All categories", key="resp_cat"
        )
    with fc2:
        sub_source = df[df["issue_category"].isin(sel_cats)] if sel_cats else df
        sub_opts = sorted(sub_source["issue_subcategory"].dropna().unique().tolist())
        sel_subs = st.multiselect(
            "Subcategory", options=sub_opts, default=[], placeholder="All subcategories", key="resp_sub"
        )
    with fc3:
        sent_opts = sorted(df["sentiment"].dropna().unique().tolist())
        sel_sents = st.multiselect(
            "Sentiment", options=sent_opts, default=[], placeholder="All sentiments", key="resp_sent"
        )
    with fc4:
        sev_range = st.select_slider(
            "Severity range", options=[1, 2, 3, 4, 5], value=(1, 5), key="resp_sev"
        )
        sev_min, sev_max = sev_range

    fc5, fc6 = st.columns(2)
    with fc5:
        seg_opts = sorted(df["nps_segment"].dropna().unique().tolist())
        sel_segs = st.multiselect(
            "NPS segment", options=seg_opts, default=[], placeholder="All segments", key="resp_seg"
        )
    with fc6:
        churn_only = st.checkbox("Churn risk only", value=False, key="resp_churn")

    # ── Apply filters ─────────────────────────────────────────────────────────
    display = df.copy()
    if sel_cats:
        display = display[display["issue_category"].isin(sel_cats)]
    if sel_subs:
        display = display[display["issue_subcategory"].isin(sel_subs)]
    if sel_sents:
        display = display[display["sentiment"].isin(sel_sents)]
    if sel_segs:
        display = display[display["nps_segment"].isin(sel_segs)]
    if "severity_score" in display.columns:
        display = display[
            (display["severity_score"] >= sev_min) & (display["severity_score"] <= sev_max)
        ]
    if churn_only and "churn_risk" in display.columns:
        display = display[display["churn_risk"]]

    st.caption(f"Showing **{len(display):,}** of **{len(df):,}** responses")

    if display.empty:
        st.info("No responses match the current filters.")
        return

    # ── Prepare display table ─────────────────────────────────────────────────
    show_cols = [
        "response_id", "response_date", "member_location", "nps_score",
        "nps_segment", "issue_category", "issue_subcategory", "sentiment",
        "severity_score", "churn_risk", "recommended_owner",
        "feedback_text", "root_cause_hypothesis", "suggested_operational_action",
    ]
    show_cols = [c for c in show_cols if c in display.columns]
    tbl = display[show_cols].copy()

    if "response_date" in tbl.columns:
        tbl["response_date"] = tbl["response_date"].dt.date
    if "churn_risk" in tbl.columns:
        tbl["churn_risk"] = tbl["churn_risk"].map({True: "⚠ Yes", False: "No"})

    sort_col = "response_date" if "response_date" in tbl.columns else tbl.columns[0]
    tbl = tbl.sort_values(sort_col, ascending=False)

    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "response_id": st.column_config.TextColumn("ID", width="small"),
            "response_date": st.column_config.DateColumn("Date", width="small"),
            "member_location": st.column_config.TextColumn("Location", width="small"),
            "nps_score": st.column_config.NumberColumn("NPS", width="small"),
            "nps_segment": st.column_config.TextColumn("Segment", width="small"),
            "issue_category": st.column_config.TextColumn("Category", width="medium"),
            "issue_subcategory": st.column_config.TextColumn("Subcategory", width="medium"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "severity_score": st.column_config.NumberColumn("Severity", width="small", format="%d ★"),
            "churn_risk": st.column_config.TextColumn("Churn Risk", width="small"),
            "recommended_owner": st.column_config.TextColumn("Owner", width="small"),
            "feedback_text": st.column_config.TextColumn("Feedback", width="large"),
            "root_cause_hypothesis": st.column_config.TextColumn("Root Cause", width="large"),
            "suggested_operational_action": st.column_config.TextColumn(
                "Suggested Action", width="large"
            ),
        },
    )

    # Export
    st.download_button(
        "⬇ Download filtered responses (CSV)",
        data=tbl.to_csv(index=False),
        file_name="responses_filtered.csv",
        mime="text/csv",
    )


def render_promoter_tab(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "Promoter Insights",
            "What 9–10 scorers consistently praise — protecting and scaling what works",
        ),
        unsafe_allow_html=True,
    )

    promoters = df[df["nps_segment"] == "Promoter"]
    total_promoters = len(promoters)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        prom_nps = compute_nps(promoters)
        st.metric("Promoter Responses", total_promoters)
        drivers = promoter_drivers(df, max_quotes=2)
        if drivers:
            driver_df = pd.DataFrame([
                {"Category": d["category"], "Subcategory": d["subcategory"], "Count": d["count"]}
                for d in drivers
            ])
            fig = px.bar(
                driver_df,
                x="Count",
                y="Subcategory",
                orientation="h",
                color_discrete_sequence=["#22C55E"],
                title="Top Promoter Subcategories",
            )
            fig.update_layout(
                height=320,
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                margin=dict(t=40, b=10, l=0, r=20),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(
            f'<div style="font-weight:600;font-size:0.85rem;color:{DARK};margin-bottom:8px">'
            f"Promoter Quotes</div>",
            unsafe_allow_html=True,
        )
        drivers = promoter_drivers(df, max_quotes=2)
        shown = 0
        for d in drivers:
            for q in d.get("example_quotes", []):
                if shown >= 6:
                    break
                st.markdown(
                    f"""<div style="border-left:3px solid #22C55E;padding:8px 12px;
                    margin-bottom:8px;background:{CREAM};border-radius:0 6px 6px 0">
                    <span style="font-size:0.72rem;color:{MEDIUM_GRAY}">
                      {q['response_id']} · {q['location']} · NPS {q.get('nps_score','—')}
                    </span><br>
                    <span style="font-size:0.85rem;color:{DARK}">{q['text'][:220]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                shown += 1



def render_export_tab(
    df: pd.DataFrame,
    all_alerts: list[dict],
    health_df: pd.DataFrame,
    dev_df: pd.DataFrame,
) -> None:
    st.markdown(
        section_header("Export", "Download analysis outputs in CSV or Excel format"),
        unsafe_allow_html=True,
    )

    day_labels = sorted(df["day_label"].dropna().unique().tolist(), reverse=True)
    week_labels = sorted(df["week_label"].dropna().unique().tolist(), reverse=True)
    most_recent_day = day_labels[0] if day_labels else None
    most_recent_week = week_labels[0] if week_labels else None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Enriched Dataset")
        st.caption(f"All {len(df)} responses with AI analysis fields.")
        sub1, sub2 = st.columns(2)
        with sub1:
            st.download_button(
                "Download CSV",
                data=enriched_csv(df),
                file_name="preventativescan_feedback_enriched.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with sub2:
            st.download_button(
                "Download Excel",
                data=enriched_excel(df),
                file_name="preventativescan_feedback_enriched.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.markdown("#### Active Alerts")
        st.caption(f"{len(all_alerts)} alerts across all tiers.")
        sub3, sub4 = st.columns(2)
        with sub3:
            st.download_button(
                "Download CSV",
                data=alerts_csv(all_alerts),
                file_name="preventativescan_alerts.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with sub4:
            st.download_button(
                "Download Excel",
                data=alerts_excel(all_alerts),
                file_name="preventativescan_alerts.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.markdown("#### Location Analysis")
        st.caption("Health scores and deviation scores by location.")
        sub5, sub6 = st.columns(2)
        with sub5:
            st.download_button(
                "Download CSV",
                data=location_analysis_csv(health_df, dev_df),
                file_name="preventativescan_location_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with sub6:
            st.download_button(
                "Download Excel",
                data=location_analysis_excel(health_df, dev_df),
                file_name="preventativescan_location_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col2:
        st.markdown("#### Daily Digest")
        if most_recent_day:
            sel_day = st.selectbox(
                "Select date",
                options=day_labels,
                index=0,
                key="export_day",
                format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y"),
            )
            dd = daily_digest(df, sel_day)
            sub7, sub8 = st.columns(2)
            with sub7:
                st.download_button(
                    "Download CSV",
                    data=daily_digest_csv(dd),
                    file_name=f"daily_digest_{sel_day}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with sub8:
                st.download_button(
                    "Download Excel",
                    data=daily_digest_excel(dd),
                    file_name=f"daily_digest_{sel_day}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        st.markdown("#### Weekly Digest")
        if most_recent_week:
            sel_week = st.selectbox(
                "Select week",
                options=week_labels,
                index=0,
                key="export_week",
            )
            wd = weekly_digest(df, all_alerts=all_alerts, target_week=sel_week)
            sub9, sub10 = st.columns(2)
            with sub9:
                st.download_button(
                    "Download CSV",
                    data=weekly_digest_csv(wd),
                    file_name=f"weekly_digest_{sel_week}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with sub10:
                st.download_button(
                    "Download Excel",
                    data=weekly_digest_excel(wd),
                    file_name=f"weekly_digest_{sel_week}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Page header
    st.markdown(
        f'<h1 style="margin-bottom:2px">PreventativeScan</h1>'
        f'<p style="color:{MEDIUM_GRAY};font-size:0.95rem;margin-top:0;margin-bottom:1.5rem">'
        f"Automated Feedback Analysis Tool · Member Experience &amp; Operations</p>",
        unsafe_allow_html=True,
    )

    # Load raw CSV — use uploaded file if one is stored in widget state,
    # otherwise fall back to the default dataset on disk.
    # The file_uploader widget (key="csv_upload") lives inside render_sidebar,
    # but Streamlit persists widget values in session_state so we can read it
    # here before render_sidebar is called.
    _upload = st.session_state.get("csv_upload")
    with st.spinner("Loading data…"):
        if _upload is not None:
            df_raw, load_warnings = _load_from_upload(_upload.getvalue())
        else:
            df_raw, load_warnings = _load_raw()

    if load_warnings:
        with st.expander("⚠ Data quality notices", expanded=False):
            for w in load_warnings:
                st.caption(f"• {w}")

    # Check analysis state up front so we can enrich df before filtering
    stats = cache_stats(CACHE_PATH)
    analysis_ready = stats["total_cached"] > 0

    # Build enriched df (AI columns merged from cache) — falls back to raw if cache empty
    import io as _io
    df_raw_json = df_raw.to_json(orient="split", date_format="iso")
    if analysis_ready:
        _enriched_json = _get_enriched_df_json(df_raw_json, stats["total_cached"])
        df_enriched = pd.read_json(_io.StringIO(_enriched_json), orient="split")
        df_enriched["response_date"] = pd.to_datetime(df_enriched["response_date"])
        df_enriched["week_start"] = pd.to_datetime(df_enriched["week_start"])
        if "churn_risk" in df_enriched.columns:
            df_enriched["churn_risk"] = df_enriched["churn_risk"].astype(bool)
        if "severity_score" in df_enriched.columns:
            df_enriched["severity_score"] = (
                pd.to_numeric(df_enriched["severity_score"], errors="coerce")
                .fillna(2).astype(int)
            )
    else:
        df_enriched = df_raw.copy()

    # Sidebar + filters — pass enriched df so the filtered result carries analysis columns
    df_filtered, filters = render_sidebar(df_enriched)

    # Run analysis if triggered
    if filters["api_key"]:
        maybe_run_analysis(df_raw, filters["api_key"])

    if not analysis_ready:
        st.warning(
            "Analysis has not been run yet. Enter your Anthropic API key in the sidebar "
            "and click **Run analysis** to classify all responses."
        )
        st.stop()

    # Compute all metrics from the enriched df
    with st.spinner("Computing metrics…"):
        enriched_for_metrics = df_enriched.to_json(orient="split", date_format="iso")
        metrics_cache = _compute_all_metrics(enriched_for_metrics)
        m = _restore_metrics(metrics_cache)

    # For filtered views we recompute on-the-fly (fast — no API calls)
    chart_df = m["chart_df"]
    all_alerts = m["all_alerts"]
    health_df = m["health"]
    dev_df = m["dev"]
    persist_df = m["persist"]

    # ── Persistent critical alert banner ────────────────────────────────────
    critical_alerts = [a for a in all_alerts if a.get("tier") == "Critical"]
    if critical_alerts:
        cat_names = list(dict.fromkeys(a["category"] for a in critical_alerts))[:3]
        extras = len(critical_alerts) - len(cat_names)
        cat_str = ", ".join(cat_names) + (f" +{extras} more" if extras > 0 else "")
        st.error(
            f"🔴 **{len(critical_alerts)} Critical Alert{'s' if len(critical_alerts) > 1 else ''}**"
            f" — {cat_str}. Open the **🚨 Alerts** tab for details and recommended actions."
        )

    # Tabs
    tabs = st.tabs([
        "📋 Briefing",
        "🔍 Top Issues",
        "🗒 Responses",
        "📉 Control Charts",
        "🚨 Alerts",
        "🗺 Segments",
        "⭐ Promoters",
        "⬇ Export",
    ])

    with tabs[0]:
        render_briefing_tab(df_filtered, all_alerts, dev_df, persist_df, df_enriched)
    with tabs[1]:
        render_top_issues_tab(df_filtered)
    with tabs[2]:
        render_responses_tab(df_filtered)
    with tabs[3]:
        render_control_charts_tab(df_enriched, chart_df)
    with tabs[4]:
        render_alerts_tab(df_filtered, all_alerts, chart_df)
    with tabs[5]:
        render_segments_tab(df_filtered, health_df, dev_df)
    with tabs[6]:
        render_promoter_tab(df_filtered)
    with tabs[7]:
        render_export_tab(df_filtered, all_alerts, health_df, dev_df)


if __name__ == "__main__":
    main()
