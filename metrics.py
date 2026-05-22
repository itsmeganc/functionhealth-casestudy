"""
Metrics computation for the PreventativeScan Feedback Analysis Tool.

Computes:
    - NPS (overall, by location, by week)
    - Top issue summary with example quotes
    - Weekly aggregations
    - Promoter driver analysis
    - Sentiment distribution
    - Issue velocity (week-over-week rate change)
    - Severity trajectory per category
    - Churn-risk concentration by location and category
    - Issue co-occurrence matrix
    - Location deviation scoring
    - Location health scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    NPS_PROMOTER_MIN,
    NPS_PASSIVE_MIN,
    OPERATIONAL_CATEGORIES,
    LOCATION_DEVIATION_THRESHOLD,
    LOCATION_DEVIATION_MIN_COUNT,
    LOCATION_PERSISTENCE_WEEKS,
    LOCATION_HEALTH_SEVERITY_FLOOR,
    VELOCITY_WEEKS_LOOKBACK,
    COOCCURRENCE_MIN_COUNT,
    ALERT_EXCLUDE_CATEGORIES,
)


# ── NPS ───────────────────────────────────────────────────────────────────────

def compute_nps(df: pd.DataFrame) -> dict:
    """Compute overall NPS and segment counts from scored responses."""
    scored = df[df["nps_score"].notna()].copy()
    if scored.empty:
        return {"nps": None, "promoters": 0, "passives": 0, "detractors": 0, "total": 0}

    scores = scored["nps_score"].astype(int)
    promoters = (scores >= NPS_PROMOTER_MIN).sum()
    passives = ((scores >= NPS_PASSIVE_MIN) & (scores < NPS_PROMOTER_MIN)).sum()
    detractors = (scores < NPS_PASSIVE_MIN).sum()
    total = len(scored)

    nps = round(100 * (promoters - detractors) / total, 1)
    return {
        "nps": nps,
        "promoters": int(promoters),
        "passives": int(passives),
        "detractors": int(detractors),
        "total": int(total),
        "promoter_pct": round(100 * promoters / total, 1),
        "detractor_pct": round(100 * detractors / total, 1),
    }


def nps_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """NPS score and segment breakdown per location."""
    results = []
    for loc, grp in df[df["nps_score"].notna()].groupby("member_location"):
        stats = compute_nps(grp)
        stats["location"] = loc
        results.append(stats)
    out = pd.DataFrame(results).sort_values("nps", ascending=False)
    return out[["location", "nps", "promoters", "passives", "detractors", "total",
                "promoter_pct", "detractor_pct"]].reset_index(drop=True)


def nps_by_week(df: pd.DataFrame) -> pd.DataFrame:
    """NPS score per ISO week."""
    results = []
    scored = df[df["nps_score"].notna() & df["week_start"].notna()]
    for week_start, grp in scored.groupby("week_start"):
        stats = compute_nps(grp)
        stats["week_start"] = week_start
        stats["week_label"] = grp["week_label"].iloc[0]
        results.append(stats)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("week_start").reset_index(drop=True)


# ── Top issues ────────────────────────────────────────────────────────────────

def top_issues(
    df: pd.DataFrame,
    n: int = 5,
    exclude_positive: bool = True,
    max_quotes: int = 3,
) -> list[dict]:
    """
    Return the top N issue categories by response count, each with:
        category, count, pct_of_total, avg_severity, detractor_pct,
        churn_risk_count, subcategories (breakdown), example_quotes,
        recommended_owner
    """
    working = df.copy()
    if exclude_positive:
        working = working[~working["issue_category"].isin(ALERT_EXCLUDE_CATEGORIES)]

    total = len(working)
    if total == 0:
        return []

    grouped = (
        working.groupby("issue_category")
        .agg(
            count=("response_id", "count"),
            avg_severity=("severity_score", "mean"),
            churn_risk_count=("churn_risk", "sum"),
            detractor_count=(
                "nps_segment",
                lambda s: (s == "Detractor").sum(),
            ),
        )
        .reset_index()
    )

    grouped["pct_of_total"] = (grouped["count"] / total * 100).round(1)
    grouped["avg_severity"] = grouped["avg_severity"].round(2)
    grouped["detractor_pct"] = (
        grouped["detractor_count"] / grouped["count"] * 100
    ).round(1)

    top = grouped.nlargest(n, "avg_severity")

    result = []
    for _, row in top.iterrows():
        cat = row["issue_category"]
        cat_df = working[working["issue_category"] == cat]

        # Subcategory breakdown with root cause + suggested action from
        # the highest-severity response in each subcategory
        sub_detail = []
        for sub, grp in cat_df.groupby("issue_subcategory"):
            best = grp.sort_values("severity_score", ascending=False).iloc[0]
            sub_detail.append({
                "subcategory": sub,
                "count": len(grp),
                "root_cause": best.get("root_cause_hypothesis", "") or "",
                "suggested_action": best.get("suggested_operational_action", "") or "",
            })
        sub_detail.sort(key=lambda x: x["count"], reverse=True)
        sub_detail = sub_detail[:5]
        sub_counts = {s["subcategory"]: s["count"] for s in sub_detail}

        # Example quotes: prefer higher severity, then non-empty feedback
        quote_df = (
            cat_df[cat_df["has_feedback"]]
            .sort_values("severity_score", ascending=False)
            .head(max_quotes)
        )
        quotes = [
            {
                "response_id": r["response_id"],
                "nps_score": r["nps_score"],
                "location": r["member_location"],
                "date": str(r["response_date"].date()) if pd.notna(r["response_date"]) else "",
                "text": r["feedback_text"],
                "severity": r["severity_score"],
            }
            for _, r in quote_df.iterrows()
        ]

        from config import CATEGORY_OWNER_MAP
        result.append({
            "category": cat,
            "count": int(row["count"]),
            "pct_of_total": float(row["pct_of_total"]),
            "avg_severity": float(row["avg_severity"]),
            "detractor_pct": float(row["detractor_pct"]),
            "churn_risk_count": int(row["churn_risk_count"]),
            "subcategories": sub_counts,
            "subcategory_details": sub_detail,
            "example_quotes": quotes,
            "recommended_owner": CATEGORY_OWNER_MAP.get(cat, "Member Experience"),
        })

    return result


# ── Promoter driver analysis ──────────────────────────────────────────────────

def promoter_drivers(df: pd.DataFrame, max_quotes: int = 2) -> list[dict]:
    """
    Analyse what 9–10 scorers consistently praise.
    Returns top subcategories by frequency among Promoters.
    """
    promoters = df[df["nps_segment"] == "Promoter"]
    if promoters.empty:
        return []

    grouped = (
        promoters[promoters["has_feedback"]]
        .groupby(["issue_category", "issue_subcategory"])
        .agg(count=("response_id", "count"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(10)
    )

    result = []
    for _, row in grouped.iterrows():
        mask = (
            (promoters["issue_category"] == row["issue_category"])
            & (promoters["issue_subcategory"] == row["issue_subcategory"])
            & promoters["has_feedback"]
        )
        quotes = [
            {"response_id": r["response_id"], "text": r["feedback_text"],
             "nps_score": r["nps_score"], "location": r["member_location"]}
            for _, r in promoters[mask].head(max_quotes).iterrows()
        ]
        result.append({
            "category": row["issue_category"],
            "subcategory": row["issue_subcategory"],
            "count": int(row["count"]),
            "example_quotes": quotes,
        })
    return result


# ── Weekly aggregations ───────────────────────────────────────────────────────

def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-week aggregate table: response count, avg NPS, avg severity,
    detractor count, top 3 categories by count.
    """
    valid = df[df["week_start"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for week_start, grp in valid.groupby("week_start"):
        scored = grp[grp["nps_score"].notna()]
        top3 = (
            grp[~grp["issue_category"].isin(ALERT_EXCLUDE_CATEGORIES)]
            .groupby("issue_category")["response_id"]
            .count()
            .nlargest(3)
            .index.tolist()
        )
        rows.append({
            "week_start": week_start,
            "week_label": grp["week_label"].iloc[0],
            "response_count": len(grp),
            "avg_nps_score": round(scored["nps_score"].astype(float).mean(), 1)
            if not scored.empty
            else None,
            "avg_severity": round(grp["severity_score"].mean(), 2),
            "detractor_count": (grp["nps_segment"] == "Detractor").sum(),
            "promoter_count": (grp["nps_segment"] == "Promoter").sum(),
            "churn_risk_count": grp["churn_risk"].sum(),
            "top_categories": ", ".join(top3),
        })

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)


def weekly_category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (week_start, issue_category) pair: count, total responses that
    week, and issue_rate = count / total.

    Only includes categories in OPERATIONAL_CATEGORIES.
    Weeks with zero occurrences of a category are included as 0-rate rows so
    p-chart baselines are accurate.
    """
    valid = df[df["week_start"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    weekly_totals = valid.groupby("week_start")["response_id"].count().rename("weekly_total")

    cat_df = valid[valid["issue_category"].isin(OPERATIONAL_CATEGORIES)]
    counts = (
        cat_df.groupby(["week_start", "issue_category"])["response_id"]
        .count()
        .reset_index(name="count")
    )

    # Build a complete grid: every (week, category) combination
    all_weeks = valid["week_start"].unique()
    all_cats = OPERATIONAL_CATEGORIES
    index = pd.MultiIndex.from_product(
        [all_weeks, all_cats], names=["week_start", "issue_category"]
    )
    full = pd.DataFrame(index=index).reset_index()
    full = full.merge(counts, on=["week_start", "issue_category"], how="left")
    full["count"] = full["count"].fillna(0).astype(int)
    full = full.merge(weekly_totals, on="week_start", how="left")

    full["issue_rate"] = full.apply(
        lambda r: r["count"] / r["weekly_total"] if r["weekly_total"] > 0 else 0.0,
        axis=1,
    )

    # Add week_label for display
    week_label_map = (
        valid.drop_duplicates("week_start")
        .set_index("week_start")["week_label"]
        .to_dict()
    )
    full["week_label"] = full["week_start"].map(week_label_map)

    return full.sort_values(["issue_category", "week_start"]).reset_index(drop=True)


# ── Issue velocity ────────────────────────────────────────────────────────────

def issue_velocity(weekly_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Week-over-week change in issue rate per category.
    Adds columns: rate_change, pct_change, is_accelerating.
    """
    out = weekly_rates.copy().sort_values(["issue_category", "week_start"])

    out["rate_change"] = out.groupby("issue_category")["issue_rate"].diff(
        VELOCITY_WEEKS_LOOKBACK
    )
    out["pct_change"] = out.groupby("issue_category")["issue_rate"].pct_change(
        VELOCITY_WEEKS_LOOKBACK
    )
    out["is_accelerating"] = out["rate_change"] > 0
    return out


# ── Severity trajectory ───────────────────────────────────────────────────────

def severity_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average severity per category per week, with a simple linear trend
    direction flag (rising / stable / falling).
    """
    valid = df[df["week_start"].notna() & df["issue_category"].isin(OPERATIONAL_CATEGORIES)]
    traj = (
        valid.groupby(["week_start", "week_label", "issue_category"])["severity_score"]
        .mean()
        .reset_index(name="avg_severity")
        .sort_values(["issue_category", "week_start"])
    )

    def _trend(series: pd.Series) -> str:
        if len(series) < 2:
            return "stable"
        slope = np.polyfit(range(len(series)), series, 1)[0]
        if slope > 0.05:
            return "rising"
        if slope < -0.05:
            return "falling"
        return "stable"

    trend_map = (
        traj.groupby("issue_category")["avg_severity"]
        .apply(_trend)
        .rename("severity_trend")
    )
    traj = traj.merge(trend_map, on="issue_category", how="left")
    return traj


# ── Churn-risk concentration ──────────────────────────────────────────────────

def churn_risk_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (location, category) pair, count churn_risk=True responses and
    compute the churn_risk_rate within that cell.
    """
    churn = df[df["churn_risk"]].copy()
    if churn.empty:
        return pd.DataFrame(
            columns=["member_location", "issue_category", "churn_count", "total_in_cell", "churn_rate"]
        )

    cell_totals = (
        df.groupby(["member_location", "issue_category"])["response_id"]
        .count()
        .rename("total_in_cell")
    )
    churn_counts = (
        churn.groupby(["member_location", "issue_category"])["response_id"]
        .count()
        .rename("churn_count")
    )
    result = (
        pd.concat([cell_totals, churn_counts], axis=1)
        .reset_index()
        .fillna(0)
    )
    result["churn_count"] = result["churn_count"].astype(int)
    result["churn_rate"] = (result["churn_count"] / result["total_in_cell"]).round(3)
    return result.sort_values("churn_count", ascending=False).reset_index(drop=True)


# ── Issue co-occurrence ───────────────────────────────────────────────────────

def issue_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how often each pair of issue categories appears together in the
    same (location, week) cell. Returns a tidy dataframe of pairs.
    """
    op = df[df["issue_category"].isin(OPERATIONAL_CATEGORIES)].copy()
    op["cell"] = op["member_location"] + "|" + op["week_label"].fillna("")

    pairs: dict[tuple, int] = {}
    for _, cell_df in op.groupby("cell"):
        cats = cell_df["issue_category"].unique().tolist()
        if len(cats) < 2:
            continue
        for i, a in enumerate(cats):
            for b in cats[i + 1 :]:
                key = tuple(sorted([a, b]))
                pairs[key] = pairs.get(key, 0) + 1

    rows = [
        {"category_a": k[0], "category_b": k[1], "co_count": v}
        for k, v in pairs.items()
        if v >= COOCCURRENCE_MIN_COUNT
    ]
    if not rows:
        return pd.DataFrame(columns=["category_a", "category_b", "co_count"])
    return pd.DataFrame(rows).sort_values("co_count", ascending=False).reset_index(drop=True)


# ── Location deviation scoring ────────────────────────────────────────────────

def location_deviation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (location, category) pair, compute how much that location's
    category rate deviates from the network average.

    Columns: member_location, issue_category, loc_count, loc_rate,
             network_rate, deviation_score, is_flagged
    """
    op = df[df["issue_category"].isin(OPERATIONAL_CATEGORIES)].copy()
    total_by_loc = df.groupby("member_location")["response_id"].count().rename("loc_total")
    network_total = len(df)

    # Location-level category counts
    loc_counts = (
        op.groupby(["member_location", "issue_category"])["response_id"]
        .count()
        .reset_index(name="loc_count")
    )
    loc_counts = loc_counts.merge(total_by_loc, on="member_location")
    loc_counts["loc_rate"] = loc_counts["loc_count"] / loc_counts["loc_total"]

    # Network-level category rates
    net_counts = (
        op.groupby("issue_category")["response_id"]
        .count()
        .rename("net_count")
        .reset_index()
    )
    net_counts["network_rate"] = net_counts["net_count"] / network_total
    loc_counts = loc_counts.merge(net_counts[["issue_category", "network_rate"]], on="issue_category")

    # Deviation score: relative deviation from network average
    loc_counts["deviation_score"] = (
        (loc_counts["loc_rate"] - loc_counts["network_rate"])
        / loc_counts["network_rate"].replace(0, np.nan)
    ).round(3)

    loc_counts["is_flagged"] = (
        (loc_counts["deviation_score"] >= LOCATION_DEVIATION_THRESHOLD)
        & (loc_counts["loc_count"] >= LOCATION_DEVIATION_MIN_COUNT)
    )

    return loc_counts.sort_values("deviation_score", ascending=False).reset_index(drop=True)


def location_persistence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify (location, category) pairs where the location has been in the
    top-3 contributors to that category for N+ consecutive weeks.
    """
    op = df[df["issue_category"].isin(OPERATIONAL_CATEGORIES) & df["week_start"].notna()].copy()

    # Per week, rank locations by count within each category
    cell_counts = (
        op.groupby(["week_start", "issue_category", "member_location"])["response_id"]
        .count()
        .reset_index(name="count")
    )
    cell_counts["rank"] = cell_counts.groupby(["week_start", "issue_category"])[
        "count"
    ].rank(method="min", ascending=False)
    top3 = cell_counts[cell_counts["rank"] <= 3].copy()

    results = []
    for (loc, cat), grp in top3.groupby(["member_location", "issue_category"]):
        weeks = sorted(grp["week_start"].unique())
        # Detect consecutive streaks
        max_streak = 1
        streak = 1
        for i in range(1, len(weeks)):
            delta = (weeks[i] - weeks[i - 1]).days
            if delta <= 14:  # allow up to 2-week gap (accounts for sparse data)
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= LOCATION_PERSISTENCE_WEEKS:
            results.append({
                "member_location": loc,
                "issue_category": cat,
                "max_consecutive_weeks": max_streak,
                "weeks_in_top3": len(weeks),
            })

    if not results:
        return pd.DataFrame(
            columns=["member_location", "issue_category", "max_consecutive_weeks", "weeks_in_top3"]
        )
    return (
        pd.DataFrame(results)
        .sort_values("max_consecutive_weeks", ascending=False)
        .reset_index(drop=True)
    )


def location_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite health score per location (lower = worse).
    Components: avg NPS (inverted), avg severity, detractor rate,
    pct responses with severity >= LOCATION_HEALTH_SEVERITY_FLOOR.
    """
    results = []
    for loc, grp in df.groupby("member_location"):
        scored = grp[grp["nps_score"].notna()]
        n = len(grp)
        avg_nps = scored["nps_score"].astype(float).mean() if not scored.empty else None
        avg_sev = grp["severity_score"].mean()
        det_rate = (grp["nps_segment"] == "Detractor").sum() / n if n > 0 else 0
        high_sev_rate = (grp["severity_score"] >= LOCATION_HEALTH_SEVERITY_FLOOR).sum() / n

        # Composite (0–100, higher = healthier)
        # avg_nps maps 1–10 → 0–1; severity maps 1–5 → 1–0 (inverted)
        nps_component = (avg_nps / 10.0) if avg_nps is not None else 0.5
        sev_component = 1.0 - ((avg_sev - 1) / 4.0)
        det_component = 1.0 - det_rate
        high_sev_component = 1.0 - high_sev_rate

        health_score = round(
            100
            * (
                0.35 * nps_component
                + 0.30 * sev_component
                + 0.20 * det_component
                + 0.15 * high_sev_component
            ),
            1,
        )
        results.append({
            "member_location": loc,
            "health_score": health_score,
            "avg_nps": round(avg_nps, 2) if avg_nps is not None else None,
            "avg_severity": round(avg_sev, 2),
            "detractor_rate": round(det_rate, 3),
            "high_severity_rate": round(high_sev_rate, 3),
            "total_responses": n,
        })

    return (
        pd.DataFrame(results)
        .sort_values("health_score")
        .reset_index(drop=True)
    )


# ── Week-over-week deltas ─────────────────────────────────────────────────────

def week_over_week_deltas(df: pd.DataFrame) -> dict:
    """
    Compute week-over-week changes for key metrics using the two most recent weeks
    present in df.  Returns a dict; all values are None when < 2 weeks exist.
    """
    weekly_nps = nps_by_week(df)

    result: dict = {
        "nps_current": None, "nps_delta": None,
        "detractor_current_pct": None, "detractor_delta_pp": None,
        "promoter_current_pct": None, "promoter_delta_pp": None,
        "volume_current": None, "volume_delta": None,
        "severity_current": None, "severity_delta": None,
        "current_week_label": None, "prior_week_label": None,
    }

    if len(weekly_nps) >= 2:
        latest = weekly_nps.iloc[-1]
        prior = weekly_nps.iloc[-2]
        result.update(
            {
                "nps_current": round(float(latest["nps"]), 1),
                "nps_delta": round(float(latest["nps"] - prior["nps"]), 1),
                "detractor_current_pct": round(float(latest["detractor_pct"]), 1),
                "detractor_delta_pp": round(
                    float(latest["detractor_pct"] - prior["detractor_pct"]), 1
                ),
                "promoter_current_pct": round(float(latest["promoter_pct"]), 1),
                "promoter_delta_pp": round(
                    float(latest["promoter_pct"] - prior["promoter_pct"]), 1
                ),
                "volume_current": int(latest["total"]),
                "volume_delta": int(latest["total"] - prior["total"]),
                "current_week_label": str(latest["week_label"]),
                "prior_week_label": str(prior["week_label"]),
            }
        )

    # Severity WoW — weekly average severity
    if "severity_score" in df.columns and "week_label" in df.columns:
        sev_weekly = (
            df[df["severity_score"].notna() & df["week_label"].notna()]
            .groupby("week_label")["severity_score"]
            .mean()
            .sort_index()
        )
        if len(sev_weekly) >= 2:
            result["severity_current"] = round(float(sev_weekly.iloc[-1]), 2)
            result["severity_delta"] = round(
                float(sev_weekly.iloc[-1] - sev_weekly.iloc[-2]), 2
            )

    return result


# ── Daily aggregations ───────────────────────────────────────────────────────

def nps_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """NPS score per calendar day."""
    results = []
    scored = df[df["nps_score"].notna() & df["day_label"].notna()]
    for day_label, grp in scored.groupby("day_label"):
        stats = compute_nps(grp)
        stats["day_label"] = day_label
        stats["response_date"] = grp["response_date"].dt.date.iloc[0]
        results.append(stats)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("day_label").reset_index(drop=True)


def daily_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (day_label, issue_category): count and avg severity.
    Only includes OPERATIONAL_CATEGORIES. Zero-count days are NOT filled
    (daily granularity makes a full grid impractically large).
    """
    valid = df[
        df["day_label"].notna()
        & df["issue_category"].isin(OPERATIONAL_CATEGORIES)
    ]
    if valid.empty:
        return pd.DataFrame(
            columns=["day_label", "issue_category", "count", "avg_severity"]
        )
    return (
        valid.groupby(["day_label", "issue_category"])
        .agg(count=("response_id", "count"), avg_severity=("severity_score", "mean"))
        .reset_index()
        .sort_values(["day_label", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-calendar-day aggregate: response count, avg NPS score, avg severity,
    detractor count, promoter count, churn-risk count, top 3 categories,
    count of severity-5 responses.
    """
    valid = df[df["day_label"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for day_label, grp in valid.groupby("day_label"):
        scored = grp[grp["nps_score"].notna()]
        top3 = (
            grp[~grp["issue_category"].isin(ALERT_EXCLUDE_CATEGORIES)]
            .groupby("issue_category")["response_id"]
            .count()
            .nlargest(3)
            .index.tolist()
        )
        rows.append({
            "day_label": day_label,
            "response_date": grp["response_date"].dt.date.iloc[0],
            "response_count": len(grp),
            "avg_nps_score": round(scored["nps_score"].astype(float).mean(), 1)
            if not scored.empty
            else None,
            "avg_severity": round(grp["severity_score"].mean(), 2),
            "detractor_count": (grp["nps_segment"] == "Detractor").sum(),
            "promoter_count": (grp["nps_segment"] == "Promoter").sum(),
            "churn_risk_count": int(grp["churn_risk"].sum()),
            "critical_count": int((grp["severity_score"] == 5).sum()),
            "top_categories": ", ".join(top3),
        })

    return pd.DataFrame(rows).sort_values("day_label").reset_index(drop=True)


# ── Sentiment breakdown ───────────────────────────────────────────────────────

def sentiment_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment distribution per category."""
    return (
        df.groupby(["issue_category", "sentiment"])["response_id"]
        .count()
        .reset_index(name="count")
        .sort_values(["issue_category", "count"], ascending=[True, False])
    )


def sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly sentiment distribution."""
    valid = df[df["week_start"].notna()]
    return (
        valid.groupby(["week_start", "week_label", "sentiment"])["response_id"]
        .count()
        .reset_index(name="count")
        .sort_values(["week_start", "sentiment"])
    )
