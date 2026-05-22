"""
Robust CSV loader for the PreventativeScan NPS feedback file.

The source file uses a non-standard outer-quoted format where every row,
including the header, is wrapped in an extra layer of double quotes and inner
quotes are doubled. Standard pd.read_csv() produces a single-column result.

Strategy:
  1. Feed the raw file to csv.reader — it correctly strips the outer quotes,
     turning each row into a single string like:
         RESP_001,3,"Tried to reschedule...",2026-01-20,New York
  2. If the result is 1-column, re-parse each cell as an inner CSV row.
  3. Fall back to standard parsing if the file is already well-formed.
"""

import csv
import io
import logging
from pathlib import Path

import pandas as pd

from config import NPS_PROMOTER_MIN, NPS_PASSIVE_MIN

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "response_id",
    "nps_score",
    "feedback_text",
    "response_date",
    "member_location",
}


def _classify_nps(score) -> str:
    if pd.isna(score):
        return "Unknown"
    s = int(score)
    if s >= NPS_PROMOTER_MIN:
        return "Promoter"
    if s >= NPS_PASSIVE_MIN:
        return "Passive"
    return "Detractor"


def _parse_rows(raw_content: str) -> list[list[str]]:
    """
    Parse raw file content, handling both standard and outer-quoted CSV formats.
    Returns a list of [header, row, row, ...] where each element is a list of
    5 field strings.
    """
    reader = csv.reader(io.StringIO(raw_content))
    outer_rows = [row for row in reader if row]

    if not outer_rows:
        return []

    # Detect outer-quoted format: all rows are single-element lists
    if all(len(r) == 1 for r in outer_rows):
        inner_rows = []
        for r in outer_rows:
            cell = r[0]
            if not cell.strip():
                continue
            try:
                parsed = next(csv.reader([cell]))
                inner_rows.append(parsed)
            except StopIteration:
                logger.warning("Could not parse inner row: %s", cell[:60])
        return inner_rows

    # Already well-formed
    return outer_rows


def _build_dataframe(rows: list[list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """
    Shared core: validate parsed rows and build the enriched DataFrame.
    Returns (df, warnings).
    """
    warnings: list[str] = []

    if not rows:
        raise ValueError("No rows could be parsed from the CSV file.")

    header = [col.strip().lower().replace(" ", "_") for col in rows[0]]

    if set(header) != EXPECTED_COLUMNS:
        missing = EXPECTED_COLUMNS - set(header)
        extra = set(header) - EXPECTED_COLUMNS
        raise ValueError(
            f"Column mismatch. Missing: {missing or 'none'}. "
            f"Unexpected: {extra or 'none'}."
        )

    df = pd.DataFrame(rows[1:], columns=header)

    # ── Type coercion ─────────────────────────────────────────────────────────
    df["nps_score"] = pd.to_numeric(df["nps_score"], errors="coerce")
    df["response_date"] = pd.to_datetime(df["response_date"], errors="coerce")
    df["feedback_text"] = df["feedback_text"].fillna("").str.strip()
    df["member_location"] = df["member_location"].fillna("Unknown").str.strip()

    # ── Validation ────────────────────────────────────────────────────────────
    n_bad_score = df["nps_score"].isna().sum()
    if n_bad_score:
        warnings.append(
            f"{n_bad_score} row(s) have unparseable NPS scores and will be "
            "excluded from NPS calculations."
        )

    out_of_range = (~df["nps_score"].isna()) & (
        (df["nps_score"] < 0) | (df["nps_score"] > 10)
    )
    if out_of_range.sum():
        warnings.append(
            f"{out_of_range.sum()} row(s) have NPS scores outside 0–10. "
            "These will be excluded from NPS calculations."
        )
        df.loc[out_of_range, "nps_score"] = pd.NA

    n_bad_date = df["response_date"].isna().sum()
    if n_bad_date:
        warnings.append(f"{n_bad_date} row(s) have unparseable dates.")

    dup_mask = df["response_id"].duplicated(keep="first")
    if dup_mask.sum():
        warnings.append(
            f"{dup_mask.sum()} duplicate response_id(s) found. "
            "Keeping the first occurrence of each."
        )
        df = df[~dup_mask].copy()

    normalised = df["feedback_text"].str.lower().str.replace(r"\s+", " ", regex=True)
    dup_text_mask = normalised.duplicated(keep="first") & (normalised != "")
    if dup_text_mask.sum():
        warnings.append(
            f"{dup_text_mask.sum()} response(s) have near-identical feedback text "
            "to an earlier row — possible data entry duplicates."
        )

    # ── Derived columns ───────────────────────────────────────────────────────
    df["nps_score"] = df["nps_score"].astype("Int64")
    df["nps_segment"] = df["nps_score"].apply(_classify_nps)
    df["has_feedback"] = df["feedback_text"].ne("")
    df["day_label"] = df["response_date"].apply(
        lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else None
    )
    df["week_label"] = df["response_date"].apply(
        lambda d: d.strftime("%G-W%V") if pd.notna(d) else None
    )
    df["week_start"] = df["response_date"].apply(
        lambda d: (d - pd.Timedelta(days=d.weekday())).normalize()
        if pd.notna(d)
        else pd.NaT
    )
    df = df.sort_values("response_date", na_position="last").reset_index(drop=True)

    logger.info(
        "Loaded %d responses (%d with feedback text, %d without). "
        "Date range: %s → %s.",
        len(df),
        df["has_feedback"].sum(),
        (~df["has_feedback"]).sum(),
        df["day_label"].min(),
        df["day_label"].max(),
    )
    return df, warnings


def load_csv(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Load the NPS feedback CSV from a file path."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {filepath}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        raw_content = fh.read()
    return _build_dataframe(_parse_rows(raw_content))


def load_csv_from_bytes(content: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Load the NPS feedback CSV from uploaded file bytes (e.g. from st.file_uploader)."""
    raw_content = content.decode("utf-8-sig")
    return _build_dataframe(_parse_rows(raw_content))


def validate_analysis_columns(df: pd.DataFrame) -> list[str]:
    """
    After AI analysis has been merged, verify required columns are present and
    return a list of any missing ones.
    """
    required = [
        "issue_category",
        "issue_subcategory",
        "issue_driver",
        "sentiment",
        "severity_score",
        "churn_risk",
        "root_cause_hypothesis",
        "recommended_owner",
        "suggested_operational_action",
    ]
    return [c for c in required if c not in df.columns]
