"""
Alert generation for the PreventativeScan Feedback Analysis Tool.

Alert tiers
-----------
Critical   — act today
Warning    — investigate this week
Watchlist  — monitor closely
Location   — location-specific systematic or persistent issue

Each alert record is a dict with a consistent schema so the UI can render
all tiers from a single list without branching on type.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal

import pandas as pd

from config import (
    ALERT_CRITICAL_SEVERITY,
    ALERT_CRITICAL_AVG_SEVERITY,
    ALERT_CRITICAL_AVG_SEVERITY_ABS,
    ALERT_CRITICAL_MIN_RESPONSES,
    ALERT_WARNING_AVG_SEVERITY,
    ALERT_WARNING_MIN_RESPONSES,
    ALERT_WARNING_DETRACTOR_RATE,
    ALERT_WARNING_CONSECUTIVE_WEEKS,
    ALERT_WATCHLIST_LOW_VOLUME_MAX,
    ALERT_WATCHLIST_SEVERITY_FLOOR,
    ALERT_EXCLUDE_CATEGORIES,
    LOCATION_DEVIATION_THRESHOLD,
    LOCATION_DEVIATION_MIN_COUNT,
    DAILY_CRITICAL_SEVERITY,
    DAILY_CHURN_RISK_MIN_SEVERITY,
)

AlertTier = Literal["Critical", "Warning", "Watchlist", "Location"]


@dataclass
class Alert:
    tier: AlertTier
    category: str
    trigger: str                          # human-readable trigger description
    affected_locations: list[str]
    response_count: int
    avg_severity: float
    example_ids: list[str]
    recommended_owner: str
    detail: str = ""
    subcategory: str = ""
    deviation_score: float | None = None  # for Location alerts

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _example_ids(df: pd.DataFrame, n: int = 3) -> list[str]:
    top = df.sort_values("severity_score", ascending=False).head(n)
    return top["response_id"].tolist()


def _locations(df: pd.DataFrame) -> list[str]:
    return sorted(df["member_location"].dropna().unique().tolist())


def _owner(category: str) -> str:
    from config import CATEGORY_OWNER_MAP
    return CATEGORY_OWNER_MAP.get(category, "Member Experience")


def _dedup(alerts: list[Alert]) -> list[Alert]:
    """
    Within each tier, keep only the highest-priority alert per category.
    Priority order: Critical > Warning > Watchlist.
    Location alerts are always kept (may have multiple per category).
    """
    seen: dict[tuple, Alert] = {}
    tier_priority = {"Critical": 0, "Warning": 1, "Watchlist": 2, "Location": -1}

    result = []
    for a in alerts:
        if a.tier == "Location":
            result.append(a)
            continue
        key = a.category
        if key not in seen or tier_priority[a.tier] < tier_priority[seen[key].tier]:
            seen[key] = a

    result.extend(seen.values())
    result.sort(key=lambda a: (tier_priority.get(a.tier, 3), -a.response_count))
    return result


# ── Alert generators ──────────────────────────────────────────────────────────

def _critical_individual(df: pd.DataFrame) -> list[Alert]:
    """Any response with severity_score == 5."""
    sev5 = df[df["severity_score"] == ALERT_CRITICAL_SEVERITY]
    alerts = []
    for cat, grp in sev5.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        alerts.append(
            Alert(
                tier="Critical",
                category=cat,
                trigger=f"Severity-5 response(s) present ({len(grp)} total)",
                affected_locations=_locations(grp),
                response_count=len(grp),
                avg_severity=round(grp["severity_score"].mean(), 2),
                example_ids=_example_ids(grp),
                recommended_owner=_owner(cat),
                detail=(
                    f"{len(grp)} response(s) with severity 5 in this category. "
                    "Immediate review recommended."
                ),
            )
        )
    return alerts


def _critical_ucl_spike(
    df: pd.DataFrame, spike_df: pd.DataFrame
) -> list[Alert]:
    """Category above UCL this period with avg severity >= threshold."""
    if spike_df.empty:
        return []

    alerts = []
    recent_spikes = spike_df.copy()

    for cat, spike_grp in recent_spikes.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        cat_df = df[df["issue_category"] == cat]
        if cat_df.empty:
            continue
        avg_sev = cat_df["severity_score"].mean()
        if avg_sev >= ALERT_CRITICAL_AVG_SEVERITY and len(cat_df) >= ALERT_CRITICAL_MIN_RESPONSES:
            weeks = sorted(spike_grp["week_label"].dropna().tolist())
            alerts.append(
                Alert(
                    tier="Critical",
                    category=cat,
                    trigger=(
                        f"Category above UCL in week(s) {', '.join(weeks)} "
                        f"with avg severity {avg_sev:.1f}"
                    ),
                    affected_locations=_locations(cat_df),
                    response_count=len(cat_df),
                    avg_severity=round(avg_sev, 2),
                    example_ids=_example_ids(cat_df),
                    recommended_owner=_owner(cat),
                    detail=(
                        "Statistical spike detected above upper control limit. "
                        "High average severity suggests systemic issue."
                    ),
                )
            )
    return alerts


def _critical_high_avg_severity(df: pd.DataFrame) -> list[Alert]:
    """Category avg severity >= 4 with at least 2 responses."""
    alerts = []
    for cat, grp in df.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        avg_sev = grp["severity_score"].mean()
        if avg_sev >= ALERT_CRITICAL_AVG_SEVERITY_ABS and len(grp) >= ALERT_CRITICAL_MIN_RESPONSES:
            alerts.append(
                Alert(
                    tier="Critical",
                    category=cat,
                    trigger=(
                        f"Avg severity {avg_sev:.1f} across {len(grp)} response(s)"
                    ),
                    affected_locations=_locations(grp),
                    response_count=len(grp),
                    avg_severity=round(avg_sev, 2),
                    example_ids=_example_ids(grp),
                    recommended_owner=_owner(cat),
                    detail=(
                        "Average severity at or above 4.0 indicates sustained serious "
                        "issues affecting member trust or access."
                    ),
                )
            )
    return alerts


def _warning_avg_severity(df: pd.DataFrame) -> list[Alert]:
    alerts = []
    for cat, grp in df.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        avg_sev = grp["severity_score"].mean()
        if avg_sev >= ALERT_WARNING_AVG_SEVERITY and len(grp) >= ALERT_WARNING_MIN_RESPONSES:
            alerts.append(
                Alert(
                    tier="Warning",
                    category=cat,
                    trigger=f"Avg severity {avg_sev:.1f} across {len(grp)} response(s)",
                    affected_locations=_locations(grp),
                    response_count=len(grp),
                    avg_severity=round(avg_sev, 2),
                    example_ids=_example_ids(grp),
                    recommended_owner=_owner(cat),
                )
            )
    return alerts


def _warning_detractor_rate(df: pd.DataFrame) -> list[Alert]:
    alerts = []
    for cat, grp in df.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        if len(grp) < ALERT_WARNING_MIN_RESPONSES:
            continue
        det_rate = (grp["nps_segment"] == "Detractor").mean()
        if det_rate >= ALERT_WARNING_DETRACTOR_RATE:
            alerts.append(
                Alert(
                    tier="Warning",
                    category=cat,
                    trigger=f"Detractor rate {det_rate:.0%} (≥{ALERT_WARNING_DETRACTOR_RATE:.0%} threshold)",
                    affected_locations=_locations(grp),
                    response_count=len(grp),
                    avg_severity=round(grp["severity_score"].mean(), 2),
                    example_ids=_example_ids(grp),
                    recommended_owner=_owner(cat),
                )
            )
    return alerts


def _warning_run_above_cl(chart_df: pd.DataFrame, df: pd.DataFrame) -> list[Alert]:
    """Category above centerline for N consecutive weeks."""
    if "run_above_cl" not in chart_df.columns:
        return []
    run_cats = chart_df[chart_df["run_above_cl"]]["issue_category"].unique()
    alerts = []
    for cat in run_cats:
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        cat_df = df[df["issue_category"] == cat]
        alerts.append(
            Alert(
                tier="Warning",
                category=cat,
                trigger=(
                    f"Above centerline for ≥{ALERT_WARNING_CONSECUTIVE_WEEKS} "
                    "consecutive weeks"
                ),
                affected_locations=_locations(cat_df),
                response_count=len(cat_df),
                avg_severity=round(cat_df["severity_score"].mean(), 2),
                example_ids=_example_ids(cat_df),
                recommended_owner=_owner(cat),
                detail=(
                    "Issue rate has been above its historical average for multiple "
                    "consecutive weeks — trend warrants investigation."
                ),
            )
        )
    return alerts


def _watchlist_above_cl_below_ucl(chart_df: pd.DataFrame, df: pd.DataFrame) -> list[Alert]:
    """Category above centerline but below UCL this period."""
    if chart_df.empty or "p_bar" not in chart_df.columns:
        return []

    # Most recent week per category
    recent = (
        chart_df[chart_df["has_limits"].fillna(False)]
        .sort_values("week_start")
        .groupby("issue_category")
        .last()
        .reset_index()
    )
    above_cl = recent[
        (recent["issue_rate"] > recent["p_bar"]) & (~recent["above_ucl"].fillna(False))
    ]

    alerts = []
    for _, row in above_cl.iterrows():
        cat = row["issue_category"]
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        cat_df = df[df["issue_category"] == cat]
        alerts.append(
            Alert(
                tier="Watchlist",
                category=cat,
                trigger="Rate above centerline but below UCL (most recent week)",
                affected_locations=_locations(cat_df),
                response_count=len(cat_df),
                avg_severity=round(cat_df["severity_score"].mean(), 2),
                example_ids=_example_ids(cat_df),
                recommended_owner=_owner(cat),
            )
        )
    return alerts


def _watchlist_low_volume_high_severity(df: pd.DataFrame) -> list[Alert]:
    """Low volume but includes severity >= floor — easy to overlook."""
    alerts = []
    for cat, grp in df.groupby("issue_category"):
        if cat in ALERT_EXCLUDE_CATEGORIES:
            continue
        high_sev = grp[grp["severity_score"] >= ALERT_WATCHLIST_SEVERITY_FLOOR]
        if 0 < len(grp) <= ALERT_WATCHLIST_LOW_VOLUME_MAX and not high_sev.empty:
            alerts.append(
                Alert(
                    tier="Watchlist",
                    category=cat,
                    trigger=(
                        f"Low volume ({len(grp)} responses) but {len(high_sev)} "
                        f"with severity ≥{ALERT_WATCHLIST_SEVERITY_FLOOR}"
                    ),
                    affected_locations=_locations(grp),
                    response_count=len(grp),
                    avg_severity=round(grp["severity_score"].mean(), 2),
                    example_ids=_example_ids(high_sev),
                    recommended_owner=_owner(cat),
                )
            )
    return alerts


# ── Location-specific alerts ──────────────────────────────────────────────────

def _location_deviation_alerts(deviation_df: pd.DataFrame, df: pd.DataFrame) -> list[Alert]:
    """
    Alert when a location's category rate is LOCATION_DEVIATION_THRESHOLD
    above the network average AND has >= LOCATION_DEVIATION_MIN_COUNT responses.
    """
    if deviation_df.empty:
        return []

    flagged = deviation_df[deviation_df["is_flagged"]].copy()
    alerts = []

    for _, row in flagged.iterrows():
        loc = row["member_location"]
        cat = row["issue_category"]
        cat_loc_df = df[
            (df["member_location"] == loc) & (df["issue_category"] == cat)
        ]
        dev_pct = round(row["deviation_score"] * 100, 1)
        alerts.append(
            Alert(
                tier="Location",
                category=cat,
                trigger=(
                    f"{loc}: {dev_pct}% above network avg rate for this category"
                ),
                affected_locations=[loc],
                response_count=int(row["loc_count"]),
                avg_severity=round(cat_loc_df["severity_score"].mean(), 2)
                if not cat_loc_df.empty
                else 0.0,
                example_ids=_example_ids(cat_loc_df),
                recommended_owner=_owner(cat),
                detail=(
                    f"Location rate: {row['loc_rate']:.1%}, "
                    f"Network rate: {row['network_rate']:.1%}. "
                    "Investigate whether this is a location-specific operational failure."
                ),
                deviation_score=float(row["deviation_score"]),
            )
        )
    return alerts


def _location_persistence_alerts(
    persistence_df: pd.DataFrame, df: pd.DataFrame
) -> list[Alert]:
    """
    Alert when a location has been a top-3 contributor to a category
    for N+ consecutive weeks.
    """
    if persistence_df.empty:
        return []

    alerts = []
    for _, row in persistence_df.iterrows():
        loc = row["member_location"]
        cat = row["issue_category"]
        cat_loc_df = df[
            (df["member_location"] == loc) & (df["issue_category"] == cat)
        ]
        alerts.append(
            Alert(
                tier="Location",
                category=cat,
                trigger=(
                    f"{loc}: persistent top-3 contributor to this category "
                    f"for {row['max_consecutive_weeks']} consecutive weeks"
                ),
                affected_locations=[loc],
                response_count=int(row["weeks_in_top3"]),
                avg_severity=round(cat_loc_df["severity_score"].mean(), 2)
                if not cat_loc_df.empty
                else 0.0,
                example_ids=_example_ids(cat_loc_df),
                recommended_owner=_owner(cat),
                detail=(
                    f"Appeared in top 3 locations for '{cat}' in "
                    f"{row['weeks_in_top3']} weeks total "
                    f"({row['max_consecutive_weeks']} consecutive). "
                    "This indicates a systemic location-specific problem, not a one-off spike."
                ),
            )
        )
    return alerts



# ── Daily alert processing ────────────────────────────────────────────────────

def daily_critical_alerts(
    df: pd.DataFrame, target_date: str | None = None
) -> list[dict]:
    """
    Scan a single day's responses for severity-5 or high-severity churn-risk
    responses. Returns a list of alert dicts suitable for the Today panel.

    target_date: "YYYY-MM-DD". Defaults to the most recent day_label in df.
    """
    if "day_label" not in df.columns:
        return []

    if target_date is None:
        target_date = df["day_label"].dropna().max()

    day_df = df[df["day_label"] == target_date].copy()
    if day_df.empty:
        return []

    # Severity-5 responses
    sev5 = day_df[day_df["severity_score"] >= DAILY_CRITICAL_SEVERITY]
    # High-severity churn-risk (severity >= 4 and churn_risk=True)
    churn_high = day_df[
        day_df["churn_risk"] & (day_df["severity_score"] >= DAILY_CHURN_RISK_MIN_SEVERITY)
    ]

    flagged = pd.concat([sev5, churn_high]).drop_duplicates("response_id")
    if flagged.empty:
        return []

    alerts_out = []
    for _, row in flagged.iterrows():
        trigger_parts = []
        if row["severity_score"] >= DAILY_CRITICAL_SEVERITY:
            trigger_parts.append(f"Severity {row['severity_score']}")
        if row["churn_risk"] and row["severity_score"] >= DAILY_CHURN_RISK_MIN_SEVERITY:
            trigger_parts.append("Churn risk flagged")

        alerts_out.append({
            "tier": "Critical",
            "type": "daily",
            "response_id": row["response_id"],
            "category": row.get("issue_category", "Unknown"),
            "subcategory": row.get("issue_subcategory", ""),
            "trigger": " | ".join(trigger_parts),
            "member_location": row["member_location"],
            "nps_score": int(row["nps_score"]) if pd.notna(row.get("nps_score")) else None,
            "severity_score": int(row["severity_score"]),
            "churn_risk": bool(row.get("churn_risk", False)),
            "feedback_text": row["feedback_text"],
            "recommended_owner": row.get("recommended_owner", "Member Experience"),
            "suggested_action": row.get("suggested_operational_action", ""),
            "date": target_date,
        })

    return alerts_out


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_alerts(
    df: pd.DataFrame,
    chart_df: pd.DataFrame | None = None,
    spike_df: pd.DataFrame | None = None,
    deviation_df: pd.DataFrame | None = None,
    persistence_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Generate all alerts and return as a list of dicts (serialisable for export).

    Parameters
    ----------
    df            : enriched feedback DataFrame (post-analysis)
    chart_df      : output of control_charts.compute_control_limits(), optionally
                    enriched with detect_run_above_centerline()
    spike_df      : output of control_charts.spike_summary()
    deviation_df  : output of metrics.location_deviation_scores()
    persistence_df: output of metrics.location_persistence_flags()
    """
    raw_alerts: list[Alert] = []

    # ── Critical ──────────────────────────────────────────────────────────────
    raw_alerts.extend(_critical_individual(df))
    raw_alerts.extend(_critical_high_avg_severity(df))
    if spike_df is not None and not spike_df.empty:
        raw_alerts.extend(_critical_ucl_spike(df, spike_df))

    # ── Warning ───────────────────────────────────────────────────────────────
    raw_alerts.extend(_warning_avg_severity(df))
    if chart_df is not None:
        raw_alerts.extend(_warning_run_above_cl(chart_df, df))

    # ── Watchlist ─────────────────────────────────────────────────────────────
    if chart_df is not None:
        raw_alerts.extend(_watchlist_above_cl_below_ucl(chart_df, df))
    raw_alerts.extend(_watchlist_low_volume_high_severity(df))

    # ── Location ──────────────────────────────────────────────────────────────
    if deviation_df is not None:
        raw_alerts.extend(_location_deviation_alerts(deviation_df, df))
    if persistence_df is not None:
        raw_alerts.extend(_location_persistence_alerts(persistence_df, df))

    deduped = _dedup(raw_alerts)
    return [a.to_dict() for a in deduped]


def alerts_for_day(df: pd.DataFrame, target_date: str | None = None) -> list[dict]:
    """
    Convenience wrapper: returns only the daily Critical alerts for a single date.
    Intended for the Today's Summary panel.
    """
    return daily_critical_alerts(df, target_date)


def alerts_by_tier(alerts: list[dict]) -> dict[str, list[dict]]:
    """Partition a flat alerts list into a dict keyed by tier."""
    out: dict[str, list[dict]] = {
        "Critical": [], "Warning": [], "Watchlist": [], "Location": []
    }
    for a in alerts:
        tier = a.get("tier", "Watchlist")
        out.setdefault(tier, []).append(a)
    return out
