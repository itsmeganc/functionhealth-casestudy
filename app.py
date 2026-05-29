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

def _get_resolution_steps(
    cat: str,
    owner: str,
    top_subcats: list[tuple[str, int]],   # [(subcategory, count), ...]
    avg_sev: float,
    n_total: int,
    cache_key: str,
    loc: str | None = None,
) -> list[str]:
    """
    Return 3–5 specific, actionable resolution steps for the aggregate issue pattern.
    Successful results are cached in session_state. Failures are NOT cached so the
    next render will retry. Returns an empty list if no API key is available.
    """
    ss_key = f"_steps_{cache_key}"
    cached = st.session_state.get(ss_key)
    # Only use cache if it holds a non-empty result (don't cache failures)
    if cached:
        return cached

    api_key = st.session_state.get("_api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return []

    subcats_str = ", ".join(
        f"{sub} ({cnt} response{'s' if cnt != 1 else ''})"
        for sub, cnt in top_subcats[:5]
    ) or "not specified"

    loc_line = f"Location: {loc}\n" if loc else ""

    prompt = f"""You are an operational advisor for PreventativeScan, a preventative health scanning company.

A dashboard has flagged the following recurring issue pattern. Write exactly 3 to 5 numbered action steps to resolve it.

Rules for each step:
- Name the specific responsible team (use the Responsible Team provided)
- State exactly what to do — no vague verbs like "escalate", "review", "monitor", or "follow up"
- State what metric or outcome to measure to confirm improvement
- Give a concrete timeframe (e.g. "within 48 hours", "by end of week", "weekly for 4 weeks")

Format strictly — one step per line, nothing else:
1. [Team]: [Specific action] — measure [metric] within [timeframe].

Issue context:
Category: {cat}
{loc_line}Responsible team: {owner}
Total flagged responses: {n_total}
Average severity: {avg_sev}/5.0
Top subcategories: {subcats_str}

Write only the numbered steps. No preamble, no intro sentence, no summary."""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        with st.spinner("Generating action steps…"):
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
        raw = msg.content[0].text.strip()
        # Accept any non-blank line that starts with a digit or "•/-"
        steps = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] in "-•*")
        ]
        if steps:
            st.session_state[ss_key] = steps
        return steps
    except Exception as e:
        st.warning(f"Could not generate action steps: {e}")
        return []


def _get_location_health_steps(
    loc: str,
    status: str,
    health_score: float,
    avg_nps: float | None,
    avg_sev: float,
    detractor_rate: float,
    flagged_issues: list[str],
    top_categories: list[tuple[str, int]],
    n_total: int,
    cache_key: str,
) -> list[str]:
    """
    Return 3–5 location-level action steps. API-synthesised when key is available;
    template-based fallback otherwise. Successful results cached in session_state.
    """
    ss_key = f"_loc_health_{cache_key}"
    cached = st.session_state.get(ss_key)
    if cached:
        return cached

    api_key = st.session_state.get("_api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")

    if api_key:
        flagged_str = ", ".join(flagged_issues) if flagged_issues else "none"
        top_cats_str = (
            ", ".join(f"{c} ({n} response{'s' if n != 1 else ''})" for c, n in top_categories[:5])
            or "none"
        )
        nps_str = f"{avg_nps:.0f}" if avg_nps is not None else "N/A"
        prompt = f"""You are an operational advisor for PreventativeScan, a preventative health scanning company.

Given this location's performance summary, write exactly 3 to 5 numbered action steps.

Rules for each step:
- Name the responsible team
- State exactly what to do — no vague verbs like "escalate", "review", "monitor", or "follow up"
- State the metric or outcome to measure
- Give a concrete timeframe
- For healthy locations (status "Healthy"): focus on maintaining quality and sharing best practices
- For at-risk locations: focus on specific remediation with urgency

Format strictly — one step per line, nothing else:
1. [Team]: [Specific action] — measure [metric] within [timeframe].

Location context:
Location: {loc}
Status: {status} (health score {health_score:.0f}/100)
NPS: {nps_str}
Avg Severity: {avg_sev}/5.0
Detractor Rate: {detractor_rate:.0%}
Total Responses: {n_total}
Flagged Issues: {flagged_str}
Top Issue Categories: {top_cats_str}

Write only the numbered steps. No preamble, no summary."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            with st.spinner(f"Generating action steps for {loc}…"):
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = msg.content[0].text.strip()
            steps = [
                line.strip() for line in raw.splitlines()
                if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] in "-•*")
            ]
            if steps:
                st.session_state[ss_key] = steps
                return steps
        except Exception as e:
            st.warning(f"Could not generate action steps: {e}")

    # ── Template fallback ─────────────────────────────────────────────────────
    from config import CATEGORY_OWNER_MAP as _HLOC_MAP
    steps: list[str] = []
    i = 1
    if flagged_issues:
        for _fcat in flagged_issues[:2]:
            _fowner = _HLOC_MAP.get(_fcat, "Operations")
            steps.append(
                f"{i}. {_fowner}: Conduct a root-cause analysis for elevated {_fcat} "
                f"complaints at {loc} — identify the top 2 contributing factors and "
                f"present a remediation plan within 2 weeks."
            )
            i += 1
    if detractor_rate >= 0.35:
        steps.append(
            f"{i}. Member Experience: Directly contact the {int(detractor_rate * n_total)} "
            f"Detractor respondents from {loc} — offer resolution and track re-contact "
            f"completion rate weekly for 4 weeks."
        )
        i += 1
    if avg_sev >= 3.5 and not flagged_issues:
        if top_categories:
            _towner = _HLOC_MAP.get(top_categories[0][0], "Operations")
            steps.append(
                f"{i}. {_towner}: Address the top issue category at {loc} "
                f"({top_categories[0][0]}, {top_categories[0][1]} responses) — "
                f"implement process change and track weekly severity average for 4 weeks."
            )
            i += 1
    if status == "Healthy" and avg_nps is not None:
        steps.append(
            f"{i}. Member Experience: Document the {loc} member journey as a best-practice "
            f"template (NPS {avg_nps:.0f}, avg severity {avg_sev}) — share with the 3 "
            f"lowest-scoring locations within 30 days."
        )
        i += 1
    if not steps:
        steps.append(
            f"1. Operations: Review {loc}'s current workflows against network benchmarks — "
            f"identify one improvement opportunity and report findings within 2 weeks."
        )
    st.session_state[ss_key] = steps
    return steps


def _render_location_profile_detail(
    loc: str,
    df_full: pd.DataFrame,
    health_row: pd.Series | None,
    dev_df: pd.DataFrame,
    page_key: str,
) -> None:
    """Full-width detail panel for a single location's health profile."""
    from config import CATEGORY_OWNER_MAP as _LMAP, ALERT_EXCLUDE_CATEGORIES as _LEXCL

    PAGE_SIZE = 5

    # ── Data prep ─────────────────────────────────────────────────────────────
    loc_df = df_full[df_full["member_location"] == loc].copy()

    health_score = float(health_row["health_score"]) if health_row is not None else 50.0
    avg_nps = health_row["avg_nps"] if health_row is not None else None
    avg_sev = float(health_row["avg_severity"]) if health_row is not None else (
        loc_df["severity_score"].mean() if not loc_df.empty else 0.0
    )
    det_rate = float(health_row["detractor_rate"]) if health_row is not None else 0.0
    high_sev_rate = float(health_row["high_severity_rate"]) if health_row is not None else 0.0
    n_total = int(health_row["total_responses"]) if health_row is not None else len(loc_df)

    if health_score < 45:
        status = "At Risk"
        status_color = CRITICAL_BORDER
    elif health_score < 60:
        status = "Monitor"
        status_color = WARNING_BORDER
    else:
        status = "Healthy"
        status_color = "#22C55E"

    # Flagged issues for this location
    loc_flagged = (
        dev_df[(dev_df["member_location"] == loc) & (dev_df["is_flagged"])]
        .sort_values("deviation_score", ascending=False)
        if not dev_df.empty and "is_flagged" in dev_df.columns
        else pd.DataFrame()
    )

    # Top issue categories for this location (operational only)
    top_cats: list[tuple[str, int]] = []
    if "issue_category" in loc_df.columns:
        top_cats = list(
            loc_df[~loc_df["issue_category"].isin(_LEXCL)]
            .groupby("issue_category")["response_id"]
            .count()
            .sort_values(ascending=False)
            .head(5)
            .items()
        )

    # ── Health summary box ────────────────────────────────────────────────────
    nps_display = f"{avg_nps:.0f}" if avg_nps is not None else "—"
    summary_text = (
        f"{loc} is <strong>{status}</strong> with a health score of "
        f"<strong>{health_score:.0f}/100</strong>. "
        f"NPS: {nps_display} · Avg severity: {avg_sev:.1f}/5.0 · "
        f"Detractor rate: {det_rate:.0%} · High-severity rate: {high_sev_rate:.0%} · "
        f"{n_total} total responses."
    )
    st.markdown(
        f'<div style="background:{CREAM};border-left:4px solid {status_color};'
        f"border-radius:0 6px 6px 0;padding:12px 16px;margin-bottom:16px;line-height:1.6\">"
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.05em;'
        f'color:{status_color};text-transform:uppercase;margin-bottom:6px">Location Health</div>'
        f'<div style="font-size:0.85rem;color:{DARK}">{summary_text}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Two-column: flagged issues + top categories ───────────────────────────
    _c1, _c2 = st.columns(2)
    with _c1:
        if not loc_flagged.empty:
            st.markdown("**⚠ Flagged Issues**")
            for _, _fr in loc_flagged.iterrows():
                _dpct = int(_fr["deviation_score"] * 100)
                st.markdown(
                    f'<div style="font-size:0.82rem;padding:4px 0;'
                    f'border-bottom:1px solid {MEDIUM_GRAY}33">'
                    f'<span style="color:{CRITICAL_BORDER};font-weight:600">+{_dpct}%</span> '
                    f'{_fr["issue_category"]} · {int(_fr["loc_count"])} responses</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**✅ No Flagged Issues**")
            st.caption("No issue categories are significantly above the network average.")

    with _c2:
        if top_cats:
            st.markdown("**Top Issue Categories**")
            _max_cnt = top_cats[0][1] if top_cats else 1
            for _cat, _cnt in top_cats:
                _bar_pct = int(_cnt / _max_cnt * 100)
                st.markdown(
                    f'<div style="font-size:0.82rem;padding:4px 0;'
                    f'border-bottom:1px solid {MEDIUM_GRAY}33">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<span>{_cat}</span>'
                    f'<span style="color:{MEDIUM_GRAY}">{_cnt}</span></div>'
                    f'<div style="height:4px;background:{MEDIUM_GRAY}22;border-radius:2px;margin-top:3px">'
                    f'<div style="height:4px;width:{_bar_pct}%;background:{status_color};border-radius:2px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # ── Action Steps ─────────────────────────────────────────────────────────
    _lh_steps = _get_location_health_steps(
        loc=loc, status=status, health_score=health_score,
        avg_nps=float(avg_nps) if avg_nps is not None else None,
        avg_sev=avg_sev, detractor_rate=det_rate,
        flagged_issues=[r["issue_category"] for _, r in loc_flagged.iterrows()],
        top_categories=top_cats, n_total=n_total,
        cache_key="_".join(loc.split()),
    )
    if _lh_steps:
        st.markdown("**Suggested Action Steps**")
        for _i, _step in enumerate(_lh_steps, 1):
            st.markdown(_step if _step[0].isdigit() else f"{_i}. {_step}")
        st.markdown("")

    # ── Recent High-Severity Feedback (severity 4–5, paginated) ─────────────
    if "severity_score" in loc_df.columns:
        _sev_df = (
            loc_df[loc_df["severity_score"] >= 4]
            .sort_values(["severity_score", "response_date"], ascending=[False, False])
            .reset_index(drop=True)
        )
        if "has_feedback" in _sev_df.columns:
            _sev_df = _sev_df[_sev_df["has_feedback"]].reset_index(drop=True)
        elif "feedback_text" in _sev_df.columns:
            _sev_df = _sev_df[_sev_df["feedback_text"].str.strip().ne("")].reset_index(drop=True)
    else:
        _sev_df = pd.DataFrame()

    _sev_total = len(_sev_df)
    st.markdown(
        f"**High-Severity Feedback** (severity 4–5) · "
        f"{_sev_total} response{'s' if _sev_total != 1 else ''}"
    )

    if _sev_total == 0:
        st.caption("No severity 4–5 responses for this location.")
    else:
        if page_key not in st.session_state:
            st.session_state[page_key] = 0
        _sev_pages = max(1, (_sev_total + PAGE_SIZE - 1) // PAGE_SIZE)
        _sev_page = min(st.session_state[page_key], _sev_pages - 1)
        st.session_state[page_key] = _sev_page
        _sev_start = _sev_page * PAGE_SIZE
        _sev_end = min(_sev_start + PAGE_SIZE, _sev_total)
        _sev_rows = list(_sev_df.iloc[_sev_start:_sev_end].iterrows())

        _html_parts = []
        for _si, (_, _sr) in enumerate(_sev_rows):
            _date_s = str(_sr.get("response_date", ""))[:10]
            _fb = str(_sr.get("feedback_text", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _sep = f"border-bottom:1px solid {MEDIUM_GRAY}44;" if _si < len(_sev_rows) - 1 else ""
            _sc = CRITICAL_BORDER if int(_sr["severity_score"]) == 5 else WARNING_BORDER
            _html_parts.append(
                f'<div style="padding:7px 0;{_sep}">'
                f'<div style="font-size:0.76rem;font-weight:600;color:{DARK};margin-bottom:2px">'
                f'{_sr["response_id"]} · NPS {_sr.get("nps_score", "—")} · '
                f'<span style="color:{_sc}">Severity {int(_sr["severity_score"])}</span> · '
                f'{_sr.get("issue_category", "—")} · {_date_s}</div>'
                f'<div style="font-size:0.82rem;color:#4B5563;line-height:1.45">{_fb}</div>'
                f'</div>'
            )
        st.markdown("".join(_html_parts), unsafe_allow_html=True)

        if _sev_pages > 1:
            _pp1, _pp2, _pp3 = st.columns([1, 2, 1])
            with _pp1:
                if _sev_page > 0:
                    if st.button("← Prev", key=f"prev_{page_key}"):
                        st.session_state[page_key] -= 1
                        st.rerun()
            with _pp2:
                st.caption(f"Page {_sev_page + 1} of {_sev_pages} · {_sev_total} total")
            with _pp3:
                if _sev_page < _sev_pages - 1:
                    if st.button("Next →", key=f"next_{page_key}"):
                        st.session_state[page_key] += 1
                        st.rerun()


def _render_loc_issue_detail(
    loc: str,
    cat: str,
    dev_row: pd.Series,
    df_full: pd.DataFrame,
    page_key: str,
    persist_weeks: int | None = None,
) -> None:
    """Render expanded content for a single location issue card.
    df_full must be the complete enriched dataset (not date-filtered) so all
    responses for this location + category are available.
    """
    PAGE_SIZE = 5

    loc_df = df_full[
        (df_full["member_location"] == loc) & (df_full["issue_category"] == cat)
    ].sort_values("severity_score", ascending=False).reset_index(drop=True)

    # ── Compute values ────────────────────────────────────────────────────────
    n_total = len(loc_df)
    avg_sev = round(loc_df["severity_score"].mean(), 1) if n_total > 0 else 0.0

    # Top 2 unique root causes from highest-severity responses
    root_causes: list[str] = []
    if "root_cause_hypothesis" in loc_df.columns:
        root_causes = (
            loc_df["root_cause_hypothesis"]
            .dropna()
            .loc[lambda s: s.str.strip() != ""]
            .drop_duplicates()
            .head(2)
            .tolist()
        )

    # ── Highlighted summary block ─────────────────────────────────────────────
    if persist_weeks is not None:
        summary_text = (
            f"{loc} has had {cat} ranking as a persistent top-3 issue for "
            f"{persist_weeks} consecutive weeks — {n_total} responses, "
            f"average severity {avg_sev}/5.0. This pattern indicates a systemic "
            "operational problem, not random variation."
        )
    else:
        dev_pct = int(dev_row["deviation_score"] * 100)
        summary_text = (
            f"{loc} is running +{dev_pct}% above the network average for {cat} issues "
            f"across {n_total} responses — {dev_row['loc_rate']:.1%} issue rate vs. "
            f"{dev_row['network_rate']:.1%} network average, average severity {avg_sev}/5.0."
        )
    if root_causes:
        summary_text += " " + " ".join(root_causes)

    st.markdown(
        f'<div style="background:{CREAM};border-left:4px solid {WARNING_BORDER};'
        f"border-radius:0 6px 6px 0;padding:12px 16px;margin-bottom:16px;"
        f'line-height:1.6">'
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.05em;'
        f'color:{WARNING_BORDER};text-transform:uppercase;margin-bottom:6px">'
        f"Issue Summary</div>"
        f'<div style="font-size:0.85rem;color:{DARK}">{summary_text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


    # ── Resolution Steps ─────────────────────────────────────────────────────
    # Build top-subcategory list (used by both API path and fallback)
    from config import CATEGORY_OWNER_MAP as _OWNER_MAP
    _owner = _OWNER_MAP.get(cat, "Member Experience")
    _top_subcats: list[tuple[str, int]] = []
    if "issue_subcategory" in loc_df.columns:
        _sub_counts = (
            loc_df.groupby("issue_subcategory")["response_id"]
            .count()
            .sort_values(ascending=False)
            .head(5)
        )
        _top_subcats = list(_sub_counts.items())

    _steps_cache_key = f"loc_{'_'.join(loc.split())}_{cat.split('/')[0].strip().replace(' ', '_')}"
    _steps = _get_resolution_steps(
        cat=cat, owner=_owner, top_subcats=_top_subcats,
        avg_sev=avg_sev, n_total=n_total,
        cache_key=_steps_cache_key, loc=loc,
    )

    # Fallback: scrape best suggested_operational_action per top subcategory
    if not _steps and "suggested_operational_action" in loc_df.columns and _top_subcats:
        _seen: set[str] = set()
        for _sub, _ in _top_subcats:
            _candidates = (
                loc_df[loc_df["issue_subcategory"] == _sub]
                .sort_values("severity_score", ascending=False)["suggested_operational_action"]
                .dropna().loc[lambda s: s.str.strip() != ""].tolist()
            )
            for _c in _candidates:
                _c_clean = _c.strip()
                if _c_clean not in _seen:
                    _steps.append(_c_clean)
                    _seen.add(_c_clean)
                    break
            if len(_steps) >= 5:
                break

    if _steps:
        st.markdown("**Suggested Resolution Steps**")
        for _i, _step in enumerate(_steps, 1):
            # Fallback steps may not have a leading number; add one if missing
            st.markdown(_step if _step[0].isdigit() else f"{_i}. {_step}")
        st.markdown("")

    # ── Paginated Feedback ────────────────────────────────────────────────────
    if "has_feedback" in loc_df.columns:
        _fb_df = loc_df[loc_df["has_feedback"]].reset_index(drop=True)
    elif "feedback_text" in loc_df.columns:
        _fb_df = loc_df[loc_df["feedback_text"].str.strip().ne("")].reset_index(drop=True)
    else:
        _fb_df = pd.DataFrame()

    total = len(_fb_df)
    st.markdown(
        f"**Associated Feedback** · {total} response{'s' if total != 1 else ''} with comments"
    )

    if total == 0:
        st.caption("No written feedback available for this location and category.")
        return

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(st.session_state[page_key], total_pages - 1)
    st.session_state[page_key] = page

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_rows = list(_fb_df.iloc[start:end].iterrows())

    # Render all page responses as one HTML block — eliminates inter-widget margins
    _fb_html = []
    for i, (_, row) in enumerate(page_rows):
        date_str = str(row.get("response_date", ""))[:10]
        fb_text = str(row.get("feedback_text", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        sep_style = f"border-bottom:1px solid {MEDIUM_GRAY}44;" if i < len(page_rows) - 1 else ""
        _fb_html.append(
            f'<div style="padding:7px 0;{sep_style}">'
            f'<div style="font-size:0.76rem;font-weight:600;color:{DARK};margin-bottom:2px">'
            f"{row['response_id']} · NPS {row.get('nps_score', '—')} · "
            f"Severity {row['severity_score']} · {date_str}</div>"
            f'<div style="font-size:0.82rem;color:#4B5563;line-height:1.45">{fb_text}</div>'
            f"</div>"
        )
    st.markdown("".join(_fb_html), unsafe_allow_html=True)

    if total_pages > 1:
        _p1, _p2, _p3 = st.columns([1, 2, 1])
        with _p1:
            if page > 0:
                if st.button("← Prev", key=f"prev_{page_key}"):
                    st.session_state[page_key] -= 1
                    st.rerun()
        with _p2:
            st.caption(f"Page {page + 1} of {total_pages} · {total} total")
        with _p3:
            if page < total_pages - 1:
                if st.button("Next →", key=f"next_{page_key}"):
                    st.session_state[page_key] += 1
                    st.rerun()


def _render_cat_issue_detail(
    cat: str,
    rank: int,
    df_full: pd.DataFrame,
    page_key: str,
) -> None:
    """Render expanded content for a top issue category card.
    df_full must be the complete enriched dataset so all responses for this
    category are included regardless of date filter.
    """
    PAGE_SIZE = 5

    cat_df = (
        df_full[df_full["issue_category"] == cat]
        .sort_values("severity_score", ascending=False)
        .reset_index(drop=True)
    )

    n_total = len(cat_df)
    avg_sev = round(cat_df["severity_score"].mean(), 1) if n_total > 0 else 0.0
    det_count = int((cat_df["nps_segment"] == "Detractor").sum()) if "nps_segment" in cat_df.columns else 0
    det_pct = round(det_count / n_total * 100, 1) if n_total > 0 else 0.0
    churn_count = int(cat_df["churn_risk"].sum()) if "churn_risk" in cat_df.columns else 0

    # Top 2 unique root causes from highest-severity responses
    root_causes: list[str] = []
    if "root_cause_hypothesis" in cat_df.columns:
        root_causes = (
            cat_df["root_cause_hypothesis"]
            .dropna()
            .loc[lambda s: s.str.strip() != ""]
            .drop_duplicates()
            .head(2)
            .tolist()
        )

    # ── Highlighted summary block ─────────────────────────────────────────────
    summary_text = (
        f"{cat} is the #{rank} most reported issue category with {n_total} total responses — "
        f"average severity {avg_sev}/5.0, {det_pct:.0f}% detractor rate"
        + (f", {churn_count} churn-risk response{'s' if churn_count != 1 else ''}" if churn_count > 0 else "")
        + "."
    )
    if root_causes:
        summary_text += " " + " ".join(root_causes)

    st.markdown(
        f'<div style="background:{CREAM};border-left:4px solid {WARNING_BORDER};'
        f"border-radius:0 6px 6px 0;padding:12px 16px;margin-bottom:16px;"
        f'line-height:1.6">'
        f'<div style="font-size:0.75rem;font-weight:700;letter-spacing:0.05em;'
        f'color:{WARNING_BORDER};text-transform:uppercase;margin-bottom:6px">'
        f"Issue Summary</div>"
        f'<div style="font-size:0.85rem;color:{DARK}">{summary_text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Resolution Steps ─────────────────────────────────────────────────────
    from config import CATEGORY_OWNER_MAP as _CAT_OWNER_MAP
    _cat_owner = _CAT_OWNER_MAP.get(cat, "Member Experience")
    _cat_top_subcats: list[tuple[str, int]] = []
    if "issue_subcategory" in cat_df.columns:
        _cat_sub_counts = (
            cat_df.groupby("issue_subcategory")["response_id"]
            .count()
            .sort_values(ascending=False)
            .head(5)
        )
        _cat_top_subcats = list(_cat_sub_counts.items())

    _cat_steps_cache_key = f"cat_{cat.split('/')[0].strip().replace(' ', '_')}"
    _cat_steps = _get_resolution_steps(
        cat=cat, owner=_cat_owner, top_subcats=_cat_top_subcats,
        avg_sev=avg_sev, n_total=n_total, cache_key=_cat_steps_cache_key,
    )

    # Fallback: scrape best suggested_operational_action per top subcategory
    if not _cat_steps and "suggested_operational_action" in cat_df.columns and _cat_top_subcats:
        _cat_seen: set[str] = set()
        for _sub, _ in _cat_top_subcats:
            _candidates = (
                cat_df[cat_df["issue_subcategory"] == _sub]
                .sort_values("severity_score", ascending=False)["suggested_operational_action"]
                .dropna().loc[lambda s: s.str.strip() != ""].tolist()
            )
            for _c in _candidates:
                _c_clean = _c.strip()
                if _c_clean not in _cat_seen:
                    _cat_steps.append(_c_clean)
                    _cat_seen.add(_c_clean)
                    break
            if len(_cat_steps) >= 5:
                break

    if _cat_steps:
        st.markdown("**Suggested Resolution Steps**")
        for _i, _step in enumerate(_cat_steps, 1):
            st.markdown(_step if _step[0].isdigit() else f"{_i}. {_step}")
        st.markdown("")

    # ── Paginated Feedback ────────────────────────────────────────────────────
    if "has_feedback" in cat_df.columns:
        _fb_df = cat_df[cat_df["has_feedback"]].reset_index(drop=True)
    elif "feedback_text" in cat_df.columns:
        _fb_df = cat_df[cat_df["feedback_text"].str.strip().ne("")].reset_index(drop=True)
    else:
        _fb_df = pd.DataFrame()

    total = len(_fb_df)
    st.markdown(
        f"**Associated Feedback** · {total} response{'s' if total != 1 else ''} with comments"
    )

    if total == 0:
        st.caption("No written feedback available for this category.")
        return

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(st.session_state[page_key], total_pages - 1)
    st.session_state[page_key] = page

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_rows = list(_fb_df.iloc[start:end].iterrows())

    # Render all page responses as one HTML block — eliminates inter-widget margins
    _fb_html = []
    for i, (_, row) in enumerate(page_rows):
        date_str = str(row.get("response_date", ""))[:10]
        fb_text = str(row.get("feedback_text", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        sep_style = f"border-bottom:1px solid {MEDIUM_GRAY}44;" if i < len(page_rows) - 1 else ""
        _fb_html.append(
            f'<div style="padding:7px 0;{sep_style}">'
            f'<div style="font-size:0.76rem;font-weight:600;color:{DARK};margin-bottom:2px">'
            f"{row['response_id']} · {row.get('member_location', '—')} · "
            f"NPS {row.get('nps_score', '—')} · Severity {row['severity_score']} · {date_str}</div>"
            f'<div style="font-size:0.82rem;color:#4B5563;line-height:1.45">{fb_text}</div>'
            f"</div>"
        )
    st.markdown("".join(_fb_html), unsafe_allow_html=True)

    if total_pages > 1:
        _p1, _p2, _p3 = st.columns([1, 2, 1])
        with _p1:
            if page > 0:
                if st.button("← Prev", key=f"prev_{page_key}"):
                    st.session_state[page_key] -= 1
                    st.rerun()
        with _p2:
            st.caption(f"Page {page + 1} of {total_pages} · {total} total")
        with _p3:
            if page < total_pages - 1:
                if st.button("Next →", key=f"next_{page_key}"):
                    st.session_state[page_key] += 1
                    st.rerun()


def render_briefing_tab(
    df: pd.DataFrame,
    all_alerts: list[dict],
    dev_df: pd.DataFrame,
    persist_df: pd.DataFrame,
    df_full: pd.DataFrame,
    health_df: pd.DataFrame,
) -> None:
    """Combined Briefing + Overview tab.
    df       = sidebar-filtered data (KPI strip, daily drill-down, NPS dist, location sections)
    df_full  = full enriched dataset, unfiltered — used only for the two 6-month trend charts
    """
    if df.empty:
        st.info("No responses match the current filters.")
        return

    from config import (
        ALERT_EXCLUDE_CATEGORIES as _EXCL,
        OPERATIONAL_CATEGORIES as _OP_CATS,
        SUBCATEGORY_MAP as _SUBCAT_MAP,
        CATEGORY_OWNER_MAP as _COWNER_MAP,
    )

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

    # ── TEAM FILTER (scopes Location Issues + Top Issue Categories) ─────────
    _team_options = sorted({
        t.strip()
        for owner in _COWNER_MAP.values()
        for t in owner.split(" / ")
        if owner != "N/A"
    })
    _tf_col, _ = st.columns([2, 5])
    with _tf_col:
        _selected_teams = st.multiselect(
            "Team view",
            options=_team_options,
            default=[],
            placeholder="All teams",
            key="brief_team_filter",
            help=(
                "Filter Location Issues and Top Issue Categories to only show "
                "problems owned by one or more selected teams. "
                "Leave empty to see all."
            ),
        )

    # ── 2. LOCATION ISSUES REQUIRING INVESTIGATION ───────────────────────────
    if not dev_df.empty and "is_flagged" in dev_df.columns:
        _flagged = dev_df[dev_df["is_flagged"]].copy()
        # Apply team filter — keep categories where ANY selected team appears in the owner string
        if _selected_teams:
            _flagged = _flagged[
                _flagged["issue_category"].map(
                    lambda _c: any(
                        _t in _COWNER_MAP.get(_c, "") for _t in _selected_teams
                    )
                )
            ].copy()
        if not _flagged.empty:
            st.markdown(
                section_header(
                    "📍 Location Issues Requiring Investigation",
                    "Locations where a specific issue rate is ≥50% above the network average "
                    "(min 3 responses)",
                ),
                unsafe_allow_html=True,
            )
            _sel_key = "brief_loc_selected"
            if _sel_key not in st.session_state:
                st.session_state[_sel_key] = None

            ncols = min(3, len(_flagged))
            loc_cols = st.columns(ncols)
            _top3 = list(_flagged.head(3).iterrows())

            for i, (_, _row) in enumerate(_top3):
                _dev_pct = int(_row["deviation_score"] * 100)
                _sev_color = CRITICAL_BORDER if _dev_pct > 100 else WARNING_BORDER
                _loc = _row["member_location"]
                _cat = _row["issue_category"]
                _card_id = f"{_loc}||{_cat}"
                _is_open = st.session_state[_sel_key] == _card_id

                with loc_cols[i]:
                    st.markdown(
                        f'<div style="background:{CREAM};border:1px solid {_sev_color};'
                        f"border-top:3px solid {_sev_color};border-radius:8px;"
                        f'padding:14px 16px;margin-bottom:4px">'
                        f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                        f"{_loc}</div>"
                        f'<div style="color:{_sev_color};font-weight:600;font-size:0.8rem;'
                        f'margin-top:3px">{_cat}</div>'
                        f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                        f'<span style="font-size:1.4rem;font-weight:700;color:{_sev_color}">'
                        f"+{_dev_pct}%</span> above network average</div>"
                        f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:6px">'
                        f"{int(_row['loc_count'])} responses · "
                        f"Location rate: {_row['loc_rate'] * 100:.1f}% · "
                        f"Network avg: {_row['network_rate'] * 100:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    _btn_label = "▲ Hide details" if _is_open else "▼ View details"
                    if st.button(_btn_label, key=f"loc_btn_{i}", use_container_width=True):
                        st.session_state[_sel_key] = None if _is_open else _card_id
                        st.rerun()

            # Full-width detail panel — rendered outside columns
            _active = st.session_state.get(_sel_key)
            if _active:
                _active_rows = [
                    _r for _, _r in _top3
                    if f"{_r['member_location']}||{_r['issue_category']}" == _active
                ]
                if _active_rows:
                    _ar = _active_rows[0]
                    _al = _ar["member_location"]
                    _ac = _ar["issue_category"]
                    _page_key = (
                        "brief_loc_"
                        + "_".join(_al.split())
                        + "_"
                        + _ac.split("/")[0].strip().replace(" ", "_")
                        + "_page"
                    )
                    st.markdown("---")
                    _render_loc_issue_detail(_al, _ac, _ar, df_full, _page_key)

            # ── Additional flagged locations (4th onward) ─────────────────────
            _remaining = list(_flagged.iloc[3:].iterrows())
            if _remaining:
                with st.expander(
                    f"{len(_remaining)} more flagged location issues",
                    expanded=False,
                ):
                    _rem_sel_key = "brief_loc_rem_selected"
                    if _rem_sel_key not in st.session_state:
                        st.session_state[_rem_sel_key] = None

                    # Render in rows of 3
                    for _rem_row_start in range(0, len(_remaining), 3):
                        _rem_row = _remaining[_rem_row_start: _rem_row_start + 3]
                        _rem_cols = st.columns(3)

                        for _rci, (_, _rrow) in enumerate(_rem_row):
                            _rdev_pct = int(_rrow["deviation_score"] * 100)
                            _rsev_color = CRITICAL_BORDER if _rdev_pct > 100 else WARNING_BORDER
                            _rloc = _rrow["member_location"]
                            _rcat = _rrow["issue_category"]
                            _rcard_id = f"{_rloc}||{_rcat}"
                            _ris_open = st.session_state[_rem_sel_key] == _rcard_id

                            with _rem_cols[_rci]:
                                st.markdown(
                                    f'<div style="background:{CREAM};border:1px solid {_rsev_color};'
                                    f"border-top:3px solid {_rsev_color};border-radius:8px;"
                                    f'padding:14px 16px;margin-bottom:4px">'
                                    f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                                    f"{_rloc}</div>"
                                    f'<div style="color:{_rsev_color};font-weight:600;font-size:0.8rem;'
                                    f'margin-top:3px">{_rcat}</div>'
                                    f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                                    f'<span style="font-size:1.4rem;font-weight:700;color:{_rsev_color}">'
                                    f"+{_rdev_pct}%</span> above network average</div>"
                                    f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:6px">'
                                    f"{int(_rrow['loc_count'])} responses · "
                                    f"Location rate: {_rrow['loc_rate'] * 100:.1f}% · "
                                    f"Network avg: {_rrow['network_rate'] * 100:.1f}%</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                                _rbtn_label = "▲ Hide details" if _ris_open else "▼ View details"
                                if st.button(
                                    _rbtn_label,
                                    key=f"loc_rem_btn_{_rem_row_start + _rci}",
                                    use_container_width=True,
                                ):
                                    st.session_state[_rem_sel_key] = (
                                        None if _ris_open else _rcard_id
                                    )
                                    st.rerun()

                        # Full-width detail panel — below whichever row holds the active card
                        _rem_active = st.session_state.get(_rem_sel_key)
                        if _rem_active and any(
                            f"{_rrow['member_location']}||{_rrow['issue_category']}" == _rem_active
                            for _, _rrow in _rem_row
                        ):
                            _rem_ar = next(
                                _rrow for _, _rrow in _rem_row
                                if f"{_rrow['member_location']}||{_rrow['issue_category']}" == _rem_active
                            )
                            _rem_al = _rem_ar["member_location"]
                            _rem_ac = _rem_ar["issue_category"]
                            _rem_page_key = (
                                "brief_loc_rem_"
                                + "_".join(_rem_al.split())
                                + "_"
                                + _rem_ac.split("/")[0].strip().replace(" ", "_")
                                + "_page"
                            )
                            st.markdown("---")
                            _render_loc_issue_detail(
                                _rem_al, _rem_ac, _rem_ar, df_full, _rem_page_key
                            )


    # ── LOCATION PROFILES (collapsible) ─────────────────────────────────────
    if not health_df.empty:
        with st.expander("📍 Location Profiles", expanded=False):
            st.caption("Health summary and action steps for every location — sorted worst to best")

            _lp_sel_key = "brief_profile_selected"
            if _lp_sel_key not in st.session_state:
                st.session_state[_lp_sel_key] = None

            _lp_list = list(health_df.sort_values("health_score").iterrows())

            for _lp_row_start in range(0, len(_lp_list), 3):
                _lp_row_items = _lp_list[_lp_row_start: _lp_row_start + 3]
                _lp_cols = st.columns(3)

                for _lp_ci, (_, _hr) in enumerate(_lp_row_items):
                    _ploc = _hr["member_location"]
                    _phscore = float(_hr["health_score"])
                    _pnps = _hr["avg_nps"]
                    _psev = float(_hr["avg_severity"])
                    _pdet = float(_hr["detractor_rate"])
                    _pn = int(_hr["total_responses"])

                    if _phscore < 45:
                        _pstatus = "🔴 At Risk"
                        _pcolor = CRITICAL_BORDER
                    elif _phscore < 60:
                        _pstatus = "🟡 Monitor"
                        _pcolor = WARNING_BORDER
                    else:
                        _pstatus = "🟢 Healthy"
                        _pcolor = "#22C55E"

                    _pflagged_count = (
                        len(dev_df[(dev_df["member_location"] == _ploc) & dev_df["is_flagged"]])
                        if not dev_df.empty and "is_flagged" in dev_df.columns else 0
                    )
                    _pcard_id = _ploc
                    _pis_open = st.session_state[_lp_sel_key] == _pcard_id

                    with _lp_cols[_lp_ci]:
                        _nps_disp = f"{_pnps:.0f}" if _pnps is not None else "—"
                        st.markdown(
                            f'<div style="background:{CREAM};border:1px solid {_pcolor};'
                            f"border-top:3px solid {_pcolor};border-radius:8px;"
                            f'padding:14px 16px;margin-bottom:4px">'
                            f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                            f"{_ploc}</div>"
                            f'<div style="color:{_pcolor};font-weight:600;font-size:0.78rem;'
                            f'margin-top:3px">{_pstatus} · {_phscore:.0f}/100</div>'
                            f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                            f"NPS {_nps_disp} · Avg sev {_psev:.1f} · {_pdet:.0%} detractors</div>"
                            f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:4px">'
                            f"{_pn} responses"
                            + (
                                f' · <span style="color:{CRITICAL_BORDER};font-weight:600">'
                                f"⚠ {_pflagged_count} flagged</span>"
                                if _pflagged_count > 0 else ""
                            )
                            + f"</div></div>",
                            unsafe_allow_html=True,
                        )
                        _lp_btn = "▲ Hide profile" if _pis_open else "▼ View profile"
                        if st.button(
                            _lp_btn,
                            key=f"lp_btn_{_lp_row_start + _lp_ci}",
                            use_container_width=True,
                        ):
                            st.session_state[_lp_sel_key] = (
                                None if _pis_open else _pcard_id
                            )
                            st.rerun()

                # Expand detail below whichever row contains the active card
                _lp_active = st.session_state.get(_lp_sel_key)
                if _lp_active and any(
                    _hr["member_location"] == _lp_active for _, _hr in _lp_row_items
                ):
                    _lp_hrow = health_df[health_df["member_location"] == _lp_active]
                    _lp_hrow_series = _lp_hrow.iloc[0] if not _lp_hrow.empty else None
                    _lp_page_key = f"lp_page_{'_'.join(_lp_active.split())}"
                    st.markdown("---")
                    _render_location_profile_detail(
                        loc=_lp_active,
                        df_full=df_full,
                        health_row=_lp_hrow_series,
                        dev_df=dev_df,
                        page_key=_lp_page_key,
                    )
                    st.markdown("---")

    # ── TOP ISSUE CATEGORIES ─────────────────────────────────────────────────
    # Uses the same ranking logic as the Top Issues tab: categories ranked by
    # avg_severity (primary signal of operational impact), with count and
    # detractor rate as supporting context on each card.
    _all_issues = top_issues(df_full, n=20)   # fetch all; slice below
    # Apply same team filter
    if _selected_teams:
        _all_issues = [
            _iss for _iss in _all_issues
            if any(_t in _COWNER_MAP.get(_iss["category"], "") for _t in _selected_teams)
        ]
    _top3_issues = _all_issues[:3]
    _remaining_issues = _all_issues[3:]
    if _top3_issues:
        st.markdown("---")
        st.markdown(
            section_header(
                "🔥 Top Issue Categories",
                "Ranked by average severity across all responses — highest operational impact",
            ),
            unsafe_allow_html=True,
        )
        _cat_sel_key = "brief_cat_selected"
        if _cat_sel_key not in st.session_state:
            st.session_state[_cat_sel_key] = None

        _ctcols = min(3, len(_top3_issues))
        cat_cols = st.columns(_ctcols)

        for i, _issue in enumerate(_top3_issues):
            _ccat = _issue["category"]
            _ccard_id = _ccat
            _cis_open = st.session_state[_cat_sel_key] == _ccard_id
            _cavg = float(_issue["avg_severity"])
            _csev_color = CRITICAL_BORDER if _cavg >= 4.0 else WARNING_BORDER

            with cat_cols[i]:
                st.markdown(
                    f'<div style="background:{CREAM};border:1px solid {_csev_color};'
                    f"border-top:3px solid {_csev_color};border-radius:8px;"
                    f'padding:14px 16px;margin-bottom:4px">'
                    f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                    f"{_ccat}</div>"
                    f'<div style="color:{_csev_color};font-weight:600;font-size:0.8rem;'
                    f'margin-top:3px">Avg severity {_cavg}</div>'
                    f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                    f'<span style="font-size:1.4rem;font-weight:700;color:{_csev_color}">'
                    f"{_issue['count']}</span> responses</div>"
                    f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:6px">'
                    f"{_issue['detractor_pct']:.0f}% detractor rate · "
                    f"{_issue['churn_risk_count']} churn risk"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                _cbtn_label = "▲ Hide details" if _cis_open else "▼ View details"
                if st.button(_cbtn_label, key=f"cat_btn_{i}", use_container_width=True):
                    st.session_state[_cat_sel_key] = None if _cis_open else _ccard_id
                    st.rerun()

        # Full-width detail panel — rendered outside columns
        _cactive = st.session_state.get(_cat_sel_key)
        if _cactive:
            _cactive_items = [
                (idx, iss) for idx, iss in enumerate(_top3_issues)
                if iss["category"] == _cactive
            ]
            if _cactive_items:
                _crank, _ = _cactive_items[0]
                _cpage_key = (
                    "brief_cat_"
                    + _cactive.split("/")[0].strip().replace(" ", "_")
                    + "_page"
                )
                st.markdown("---")
                _render_cat_issue_detail(_cactive, _crank + 1, df_full, _cpage_key)

        # ── Additional issue categories (4th onward) ──────────────────────────
        if _remaining_issues:
            with st.expander(
                f"{len(_remaining_issues)} more issue categories",
                expanded=False,
            ):
                _cat_rem_sel_key = "brief_cat_rem_selected"
                if _cat_rem_sel_key not in st.session_state:
                    st.session_state[_cat_rem_sel_key] = None

                for _rem_row_start in range(0, len(_remaining_issues), 3):
                    _rem_cat_row = _remaining_issues[_rem_row_start: _rem_row_start + 3]
                    _rem_cat_cols = st.columns(3)

                    for _rci, _rissue in enumerate(_rem_cat_row):
                        _rccat = _rissue["category"]
                        _rccard_id = _rccat
                        _rcis_open = st.session_state[_cat_rem_sel_key] == _rccard_id
                        _rcavg = float(_rissue["avg_severity"])
                        _rcsev_color = (
                            "#22C55E" if _rcavg < 3.0
                            else CRITICAL_BORDER if _rcavg >= 4.0
                            else WARNING_BORDER
                        )

                        with _rem_cat_cols[_rci]:
                            st.markdown(
                                f'<div style="background:{CREAM};border:1px solid {_rcsev_color};'
                                f"border-top:3px solid {_rcsev_color};border-radius:8px;"
                                f'padding:14px 16px;margin-bottom:4px">'
                                f'<div style="font-weight:700;color:{DARK};font-size:0.9rem">'
                                f"{_rccat}</div>"
                                f'<div style="color:{_rcsev_color};font-weight:600;font-size:0.8rem;'
                                f'margin-top:3px">Avg severity {_rcavg}</div>'
                                f'<div style="margin-top:8px;font-size:0.8rem;color:{DARK}">'
                                f'<span style="font-size:1.4rem;font-weight:700;color:{_rcsev_color}">'
                                f"{_rissue['count']}</span> responses</div>"
                                f'<div style="color:{MEDIUM_GRAY};font-size:0.75rem;margin-top:6px">'
                                f"{_rissue['detractor_pct']:.0f}% detractor rate · "
                                f"{_rissue['churn_risk_count']} churn risk"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                            _rcbtn_label = "▲ Hide details" if _rcis_open else "▼ View details"
                            if st.button(
                                _rcbtn_label,
                                key=f"cat_rem_btn_{_rem_row_start + _rci}",
                                use_container_width=True,
                            ):
                                st.session_state[_cat_rem_sel_key] = (
                                    None if _rcis_open else _rccard_id
                                )
                                st.rerun()

                    # Full-width detail panel — below whichever row holds the active card
                    _rc_active = st.session_state.get(_cat_rem_sel_key)
                    if _rc_active and any(
                        iss["category"] == _rc_active for iss in _rem_cat_row
                    ):
                        _rc_rank = next(
                            3 + _all_issues.index(iss)
                            for iss in _rem_cat_row
                            if iss["category"] == _rc_active
                        )
                        _rc_page_key = (
                            "brief_cat_rem_"
                            + _rc_active.split("/")[0].strip().replace(" ", "_")
                            + "_page"
                        )
                        st.markdown("---")
                        _render_cat_issue_detail(_rc_active, _rc_rank + 1, df_full, _rc_page_key)

    # ── 3. DAILY DRILL-DOWN ───────────────────────────────────────────────────
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
                    height=max(180, len(cat_order) * 24 + 40),
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
                        _crit_action = (c.get("suggested_action") or "").strip()
                        _crit_body = f"<em>{c.get('feedback_text', '')}</em>"
                        if _crit_action:
                            _crit_body += (
                                f'<div style="font-size:0.78rem;margin-top:7px;'
                                f'padding-top:7px;border-top:1px solid rgba(0,0,0,0.08)">'
                                f'<span style="font-weight:700">Suggested action:</span> '
                                f'{_crit_action}</div>'
                            )
                        st.markdown(
                            alert_box(
                                "Critical",
                                f"{c['response_id']} · {c['member_location']} · NPS {c.get('nps_score', '—')}",
                                _crit_body,
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


    # ── KPI STRIP ────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    nps_val = nps["nps"]
    c1.metric(
        "NPS Score",
        f"{nps_val:.0f}" if nps_val is not None else "—",
        delta=(
            f"{_prior_nps_delta:+.1f} pts {_delta_label}"
            if _prior_nps_delta is not None
            else None
        ),
    )
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

    # ── CHARTS (collapsible) ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Charts", expanded=False):
        # ── Build 6-month unfiltered slice for trend charts ──────────────────
        six_mo_cutoff = df_full["response_date"].max() - pd.Timedelta(weeks=26)
        df_6mo = df_full[df_full["response_date"] >= six_mo_cutoff].copy()
        six_mo_label = "Last 6 months · unaffected by date / location filters"

        # ── NPS DISTRIBUTION + RESPONSE VOLUME ───────────────────────────────
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

        # ── SENTIMENT OVER TIME + ISSUE VELOCITY (side by side) ─────────────
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

        # ── ISSUE DISTRIBUTION (filtered period) ─────────────────────────────
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

    # Make API key available to helper functions via session_state
    if filters["api_key"]:
        st.session_state["_api_key"] = filters["api_key"]

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
        render_briefing_tab(df_filtered, all_alerts, dev_df, persist_df, df_enriched, health_df)
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
