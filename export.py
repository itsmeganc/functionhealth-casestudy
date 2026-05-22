"""
Download/export generation for the PreventativeScan Feedback Analysis Tool.

All functions return bytes objects ready to pass to st.download_button.

Supported formats:
    - CSV  (always available)
    - Excel (.xlsx via openpyxl, with basic formatting)
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _to_excel(sheets: dict[str, pd.DataFrame], freeze_header: bool = True) -> bytes:
    """
    Write multiple DataFrames to a single Excel workbook.
    sheets: {sheet_name: dataframe}
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=safe_name, index=False)

            ws = writer.sheets[safe_name]

            # Auto-width columns (approximate)
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(cell.value)) if cell.value else 0 for cell in col_cells),
                    default=10,
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(
                    max_len + 4, 60
                )

            # Freeze top row
            if freeze_header:
                ws.freeze_panes = "A2"

    return buf.getvalue()


def _flatten_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column of lists/dicts to JSON strings for CSV compatibility."""
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].apply(
            lambda v: str(v) if isinstance(v, (list, dict)) else v
        )
    return df


# ── Enriched full dataset ─────────────────────────────────────────────────────

def enriched_csv(df: pd.DataFrame) -> bytes:
    """
    All feedback rows with every AI analysis column included.
    Columns are reordered for readability: identifiers first, then NPS,
    then AI fields, then metadata.
    """
    preferred_order = [
        "response_id", "response_date", "day_label", "week_label",
        "member_location", "nps_score", "nps_segment",
        "has_feedback", "feedback_text",
        "issue_category", "issue_subcategory", "issue_driver",
        "sentiment", "severity_score", "churn_risk",
        "root_cause_hypothesis", "recommended_owner",
        "suggested_operational_action",
        "analysis_timestamp",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    remainder = [c for c in df.columns if c not in cols]
    return _to_csv(df[cols + remainder])


def enriched_excel(df: pd.DataFrame) -> bytes:
    """Enriched dataset as a formatted Excel file."""
    preferred_order = [
        "response_id", "response_date", "member_location", "nps_score",
        "nps_segment", "has_feedback", "feedback_text",
        "issue_category", "issue_subcategory", "issue_driver",
        "sentiment", "severity_score", "churn_risk",
        "root_cause_hypothesis", "recommended_owner",
        "suggested_operational_action",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    remainder = [c for c in df.columns if c not in cols]
    export_df = df[cols + remainder].copy()
    # Booleans don't render well in Excel
    if "churn_risk" in export_df.columns:
        export_df["churn_risk"] = export_df["churn_risk"].map({True: "Yes", False: "No"})
    return _to_excel({"Feedback Data": export_df})


# ── Daily digest ──────────────────────────────────────────────────────────────

def daily_digest_csv(digest: dict[str, Any]) -> bytes:
    """
    Flatten a daily digest dict into a two-sheet-style CSV.
    Sheet 1: summary row. Sheet 2: critical responses.
    For CSV we just concatenate with a blank row between sections.
    """
    lines = []

    # Summary block
    summary_row = {
        "date": digest.get("date"),
        "total_responses": digest.get("total_responses"),
        "new_critical_count": digest.get("new_critical_count"),
        "avg_severity": digest.get("avg_severity"),
        "nps_score": digest.get("nps_snapshot", {}).get("nps") if digest.get("nps_snapshot") else None,
        "detractor_count": digest.get("detractor_count"),
        "promoter_count": digest.get("promoter_count"),
        "churn_risk_count": digest.get("churn_risk_count"),
    }
    summary_df = pd.DataFrame([summary_row])
    lines.append("=== DAILY SUMMARY ===")
    lines.append(summary_df.to_csv(index=False))

    # Top categories block
    top_cats = digest.get("top_categories", [])
    if top_cats:
        lines.append("\n=== TOP CATEGORIES TODAY ===")
        lines.append(pd.DataFrame(top_cats).to_csv(index=False))

    # Critical responses block
    criticals = digest.get("critical_responses", [])
    if criticals:
        lines.append("\n=== CRITICAL RESPONSES ===")
        crit_df = pd.DataFrame(criticals)
        lines.append(_flatten_list_col(crit_df, "affected_locations").to_csv(index=False))

    # Notable quotes block
    quotes = digest.get("notable_quotes", [])
    if quotes:
        lines.append("\n=== NOTABLE QUOTES (Severity >= 4) ===")
        lines.append(pd.DataFrame(quotes).to_csv(index=False))

    return "\n".join(lines).encode("utf-8")


def daily_digest_excel(digest: dict[str, Any]) -> bytes:
    """Daily digest as a multi-sheet Excel workbook."""
    sheets: dict[str, pd.DataFrame] = {}

    summary_row = {
        "Date": digest.get("date"),
        "Total Responses": digest.get("total_responses"),
        "New Criticals": digest.get("new_critical_count"),
        "Avg Severity": digest.get("avg_severity"),
        "NPS Score": digest.get("nps_snapshot", {}).get("nps") if digest.get("nps_snapshot") else None,
        "Detractors": digest.get("detractor_count"),
        "Promoters": digest.get("promoter_count"),
        "Churn Risk": digest.get("churn_risk_count"),
    }
    sheets["Summary"] = pd.DataFrame([summary_row])

    top_cats = digest.get("top_categories", [])
    if top_cats:
        sheets["Top Categories"] = pd.DataFrame(top_cats)

    criticals = digest.get("critical_responses", [])
    if criticals:
        crit_df = pd.DataFrame(criticals)
        sheets["Critical Responses"] = _flatten_list_col(crit_df, "affected_locations")

    quotes = digest.get("notable_quotes", [])
    if quotes:
        sheets["Notable Quotes"] = pd.DataFrame(quotes)

    return _to_excel(sheets)


# ── Weekly digest ─────────────────────────────────────────────────────────────

def weekly_digest_csv(digest: dict[str, Any]) -> bytes:
    """Weekly digest as a multi-section CSV."""
    lines = []

    summary_row = {
        "week_label": digest.get("week_label"),
        "total_responses": digest.get("total_responses"),
        "nps_score": digest.get("nps_score"),
        "nps_delta": digest.get("nps_delta"),
        "avg_severity": digest.get("avg_severity"),
        "severity_delta": digest.get("severity_delta"),
        "prior_week": digest.get("prior_week"),
    }
    lines.append("=== WEEKLY SUMMARY ===")
    lines.append(pd.DataFrame([summary_row]).to_csv(index=False))

    top5 = digest.get("top_5_issues", [])
    if top5:
        lines.append("\n=== TOP 5 ISSUES ===")
        top5_flat = [
            {
                "category": t["category"],
                "count": t["count"],
                "pct_of_total": t["pct_of_total"],
                "avg_severity": t["avg_severity"],
                "detractor_pct": t["detractor_pct"],
                "churn_risk_count": t["churn_risk_count"],
                "recommended_owner": t["recommended_owner"],
            }
            for t in top5
        ]
        lines.append(pd.DataFrame(top5_flat).to_csv(index=False))

    cc_status = digest.get("control_chart_status", {})
    if cc_status:
        lines.append("\n=== CONTROL CHART STATUS ===")
        cc_df = pd.DataFrame(
            [{"category": k, "status": v} for k, v in cc_status.items()]
        )
        lines.append(cc_df.to_csv(index=False))

    loc_flags = digest.get("location_flags", [])
    if loc_flags:
        lines.append("\n=== LOCATION FLAGS ===")
        lf_df = pd.DataFrame(loc_flags)
        lines.append(_flatten_list_col(lf_df, "affected_locations").to_csv(index=False))

    return "\n".join(lines).encode("utf-8")


def weekly_digest_excel(digest: dict[str, Any]) -> bytes:
    """Weekly digest as a multi-sheet Excel workbook."""
    sheets: dict[str, pd.DataFrame] = {}

    summary_row = {
        "Week": digest.get("week_label"),
        "Total Responses": digest.get("total_responses"),
        "NPS Score": digest.get("nps_score"),
        "NPS Delta vs Prior Week": digest.get("nps_delta"),
        "Avg Severity": digest.get("avg_severity"),
        "Severity Delta vs Prior Week": digest.get("severity_delta"),
        "Prior Week": digest.get("prior_week"),
    }
    sheets["Summary"] = pd.DataFrame([summary_row])

    top5 = digest.get("top_5_issues", [])
    if top5:
        sheets["Top 5 Issues"] = pd.DataFrame([
            {
                "Category": t["category"],
                "Count": t["count"],
                "% of Total": t["pct_of_total"],
                "Avg Severity": t["avg_severity"],
                "Detractor %": t["detractor_pct"],
                "Churn Risk Count": t["churn_risk_count"],
                "Recommended Owner": t["recommended_owner"],
            }
            for t in top5
        ])

    cc_status = digest.get("control_chart_status", {})
    if cc_status:
        sheets["Control Chart Status"] = pd.DataFrame(
            [{"Category": k, "Status": v} for k, v in cc_status.items()]
        )

    loc_flags = digest.get("location_flags", [])
    if loc_flags:
        lf_df = pd.DataFrame(loc_flags)
        sheets["Location Flags"] = _flatten_list_col(lf_df, "affected_locations")

    prom = digest.get("promoter_highlights", [])
    if prom:
        sheets["Promoter Highlights"] = pd.DataFrame([
            {
                "Category": p["category"],
                "Subcategory": p["subcategory"],
                "Count": p["count"],
            }
            for p in prom
        ])

    return _to_excel(sheets)


# ── Alerts export ─────────────────────────────────────────────────────────────

def alerts_csv(alerts: list[dict[str, Any]]) -> bytes:
    """All active alerts as a flat CSV, sorted by tier priority."""
    if not alerts:
        return b"tier,category,trigger,affected_locations,response_count,avg_severity,recommended_owner,detail\n"
    tier_order = {"Critical": 0, "Warning": 1, "Watchlist": 2, "Location": 3}
    df = pd.DataFrame(alerts)
    df = _flatten_list_col(df, "affected_locations")
    df = _flatten_list_col(df, "example_ids")
    if "tier" in df.columns:
        df["_sort"] = df["tier"].map(tier_order).fillna(4)
        df = df.sort_values("_sort").drop(columns=["_sort"])
    return _to_csv(df)


def alerts_excel(alerts: list[dict[str, Any]]) -> bytes:
    """Active alerts as a multi-sheet Excel workbook, one sheet per tier."""
    tier_order = ["Critical", "Warning", "Watchlist", "Location"]
    sheets: dict[str, pd.DataFrame] = {}
    for tier in tier_order:
        tier_alerts = [a for a in alerts if a.get("tier") == tier]
        if tier_alerts:
            df = pd.DataFrame(tier_alerts)
            df = _flatten_list_col(df, "affected_locations")
            df = _flatten_list_col(df, "example_ids")
            sheets[tier] = df
    if not sheets:
        sheets["No Alerts"] = pd.DataFrame([{"message": "No active alerts for this period."}])
    return _to_excel(sheets)


# ── Location analysis export ──────────────────────────────────────────────────

def location_analysis_csv(
    health_scores: pd.DataFrame,
    deviation_scores: pd.DataFrame,
) -> bytes:
    """Location health scores and deviation scores combined as CSV."""
    lines = ["=== LOCATION HEALTH SCORES ==="]
    lines.append(_to_csv(health_scores).decode("utf-8"))
    lines.append("\n=== LOCATION DEVIATION SCORES ===")
    lines.append(_to_csv(deviation_scores).decode("utf-8"))
    return "\n".join(lines).encode("utf-8")


def location_analysis_excel(
    health_scores: pd.DataFrame,
    deviation_scores: pd.DataFrame,
) -> bytes:
    """Location analysis as a multi-sheet Excel workbook."""
    return _to_excel({
        "Location Health Scores": health_scores,
        "Deviation Scores": deviation_scores,
    })
