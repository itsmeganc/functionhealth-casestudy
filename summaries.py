"""
Daily and weekly digest generation for the PreventativeScan Feedback Tool.

Daily digest: intended for the Member Experience team's morning review.
Weekly digest: intended for Operations' trend review and leadership reporting.

Both return structured dicts that the UI renders and export.py serialises.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from config import OPERATIONAL_CATEGORIES, ALERT_EXCLUDE_CATEGORIES
from metrics import (
    compute_nps,
    top_issues,
    promoter_drivers,
    weekly_summary,
    weekly_category_rates,
    location_health_scores,
    location_deviation_scores,
    issue_velocity,
    severity_trajectory,
)
from alerts import daily_critical_alerts, alerts_by_tier


# ── Daily digest ──────────────────────────────────────────────────────────────

def daily_digest(df: pd.DataFrame, target_date: str | None = None) -> dict[str, Any]:
    """
    Structured daily digest for a given calendar date (YYYY-MM-DD).
    Defaults to the most recent date in df.

    Returns a dict with the following keys:
        date, total_responses, new_critical_count, avg_severity,
        nps_snapshot, top_categories (list), critical_responses (list),
        notable_quotes (list), churn_risk_count, has_data (bool)
    """
    if "day_label" not in df.columns:
        return _empty_daily(target_date or "unknown")

    if target_date is None:
        target_date = df["day_label"].dropna().max()

    day_df = df[df["day_label"] == target_date].copy()

    if day_df.empty:
        return _empty_daily(target_date)

    nps = compute_nps(day_df)
    criticals = daily_critical_alerts(day_df, target_date)

    # Top 3 operational categories for the day
    op_day = day_df[~day_df["issue_category"].isin(ALERT_EXCLUDE_CATEGORIES)]
    top3_cats = (
        op_day.groupby("issue_category")["response_id"]
        .count()
        .nlargest(3)
        .reset_index()
        .rename(columns={"response_id": "count"})
        .to_dict("records")
    )

    # Notable quotes: severity >= 4, up to 5, has feedback text
    notable = (
        day_df[day_df["has_feedback"] & (day_df["severity_score"] >= 4)]
        .sort_values("severity_score", ascending=False)
        .head(5)
    )
    notable_quotes = [
        {
            "response_id": r["response_id"],
            "nps_score": int(r["nps_score"]) if pd.notna(r.get("nps_score")) else None,
            "location": r["member_location"],
            "category": r.get("issue_category", ""),
            "severity": int(r["severity_score"]),
            "text": r["feedback_text"],
        }
        for _, r in notable.iterrows()
    ]

    return {
        "has_data": True,
        "date": target_date,
        "total_responses": len(day_df),
        "new_critical_count": len(criticals),
        "avg_severity": round(day_df["severity_score"].mean(), 2),
        "nps_snapshot": nps,
        "top_categories": top3_cats,
        "critical_responses": criticals,
        "notable_quotes": notable_quotes,
        "churn_risk_count": int(day_df["churn_risk"].sum()),
        "detractor_count": int((day_df["nps_segment"] == "Detractor").sum()),
        "promoter_count": int((day_df["nps_segment"] == "Promoter").sum()),
    }


def _empty_daily(target_date: str) -> dict[str, Any]:
    return {
        "has_data": False,
        "date": target_date,
        "total_responses": 0,
        "new_critical_count": 0,
        "avg_severity": None,
        "nps_snapshot": None,
        "top_categories": [],
        "critical_responses": [],
        "notable_quotes": [],
        "churn_risk_count": 0,
        "detractor_count": 0,
        "promoter_count": 0,
    }


def daily_digest_range(
    df: pd.DataFrame, start_date: str, end_date: str
) -> list[dict[str, Any]]:
    """
    Generate a daily digest for each date in [start_date, end_date].
    Useful for exporting a multi-day view or populating a date picker.
    """
    if "day_label" not in df.columns:
        return []
    dates_in_range = sorted(
        d for d in df["day_label"].dropna().unique()
        if start_date <= d <= end_date
    )
    return [daily_digest(df, d) for d in dates_in_range]


# ── Weekly digest ─────────────────────────────────────────────────────────────

def weekly_digest(
    df: pd.DataFrame,
    chart_df: pd.DataFrame | None = None,
    all_alerts: list[dict] | None = None,
    target_week: str | None = None,
) -> dict[str, Any]:
    """
    Structured weekly digest for an ISO week label (e.g. "2025-W42").
    Defaults to the most recent week in df.

    Parameters
    ----------
    df          : enriched feedback DataFrame
    chart_df    : output of control_charts.compute_control_limits() — used for
                  control chart status summary. Pass None to skip.
    all_alerts  : flat list of alert dicts from alerts.generate_alerts() — used
                  to surface this week's active alerts. Pass None to skip.
    target_week : ISO week string e.g. "2025-W42". Defaults to most recent.

    Returns a dict with:
        week_label, total_responses, nps_score, nps_delta,
        avg_severity, severity_delta,
        top_5_issues (list), control_chart_status (dict),
        location_flags (list), promoter_highlights (list),
        weekly_alerts_by_tier (dict)
    """
    if "week_label" not in df.columns:
        return _empty_weekly(target_week or "unknown")

    if target_week is None:
        target_week = df["week_label"].dropna().max()

    week_df = df[df["week_label"] == target_week].copy()
    if week_df.empty:
        return _empty_weekly(target_week)

    # ── NPS delta: compare to prior week ─────────────────────────────────────
    all_weeks = sorted(df["week_label"].dropna().unique().tolist())
    prior_week = _prior_week_label(target_week, all_weeks)
    prior_df = df[df["week_label"] == prior_week] if prior_week else pd.DataFrame()

    nps_this = compute_nps(week_df)
    nps_prior = compute_nps(prior_df) if not prior_df.empty else None
    nps_delta = (
        round(nps_this["nps"] - nps_prior["nps"], 1)
        if nps_prior and nps_this["nps"] is not None and nps_prior["nps"] is not None
        else None
    )

    # ── Severity delta ────────────────────────────────────────────────────────
    avg_sev_this = round(week_df["severity_score"].mean(), 2)
    avg_sev_prior = round(prior_df["severity_score"].mean(), 2) if not prior_df.empty else None
    sev_delta = (
        round(avg_sev_this - avg_sev_prior, 2) if avg_sev_prior is not None else None
    )

    # ── Top 5 issues ──────────────────────────────────────────────────────────
    top5 = top_issues(week_df, n=5)

    # ── Control chart status ──────────────────────────────────────────────────
    cc_status: dict[str, str] = {}
    if chart_df is not None and not chart_df.empty:
        week_chart = chart_df[chart_df["week_label"] == target_week]
        for _, row in week_chart.iterrows():
            cat = row["issue_category"]
            if not row.get("has_limits", False):
                cc_status[cat] = "insufficient_data"
            elif row.get("above_ucl", False):
                cc_status[cat] = "above_ucl"
            elif row.get("below_lcl", False):
                cc_status[cat] = "below_lcl"
            elif row.get("run_above_cl", False):
                cc_status[cat] = "run_above_cl"
            elif row["issue_rate"] > row.get("p_bar", 0):
                cc_status[cat] = "above_centerline"
            else:
                cc_status[cat] = "normal"

    # ── Location flags this week ──────────────────────────────────────────────
    location_flags: list[dict] = []
    if all_alerts:
        location_flags = [a for a in all_alerts if a.get("tier") == "Location"]

    # ── Promoter highlights ───────────────────────────────────────────────────
    prom = promoter_drivers(week_df, max_quotes=1)[:2]

    # ── Weekly alerts by tier ─────────────────────────────────────────────────
    weekly_alerts: dict[str, list[dict]] = {}
    if all_alerts:
        weekly_alerts = alerts_by_tier(
            [a for a in all_alerts if a.get("type") != "daily"]
        )

    return {
        "has_data": True,
        "week_label": target_week,
        "total_responses": len(week_df),
        "nps_score": nps_this.get("nps"),
        "nps_delta": nps_delta,
        "avg_severity": avg_sev_this,
        "severity_delta": sev_delta,
        "top_5_issues": top5,
        "control_chart_status": cc_status,
        "location_flags": location_flags,
        "promoter_highlights": prom,
        "weekly_alerts_by_tier": weekly_alerts,
        "prior_week": prior_week,
    }


def _empty_weekly(week_label: str) -> dict[str, Any]:
    return {
        "has_data": False,
        "week_label": week_label,
        "total_responses": 0,
        "nps_score": None,
        "nps_delta": None,
        "avg_severity": None,
        "severity_delta": None,
        "top_5_issues": [],
        "control_chart_status": {},
        "location_flags": [],
        "promoter_highlights": [],
        "weekly_alerts_by_tier": {},
        "prior_week": None,
    }


def _prior_week_label(target_week: str, all_weeks: list[str]) -> str | None:
    """Return the week label immediately before target_week in all_weeks."""
    try:
        idx = all_weeks.index(target_week)
        return all_weeks[idx - 1] if idx > 0 else None
    except ValueError:
        return None


def weekly_digest_all(
    df: pd.DataFrame,
    chart_df: pd.DataFrame | None = None,
    all_alerts: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Generate a weekly digest for every week present in df."""
    weeks = sorted(df["week_label"].dropna().unique().tolist())
    return [weekly_digest(df, chart_df, all_alerts, w) for w in weeks]
