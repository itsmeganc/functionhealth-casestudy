"""
AI analysis module — classifies each NPS response using Claude and persists
results to a JSON cache so re-runs are instant.

Batching strategy: 10 responses per API call to keep costs low.
Empty feedback rows are classified deterministically without an API call.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import anthropic
import pandas as pd

from config import (
    ISSUE_CATEGORIES,
    SUBCATEGORY_MAP,
    ISSUE_DRIVERS,
    SENTIMENT_LABELS,
    SEVERITY_RUBRIC,
    CATEGORY_OWNER_MAP,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "analysis_cache.json"
BATCH_SIZE = 10
MAX_RETRIES = 2
RETRY_DELAY = 2.0  # seconds between retries


# ── Stub for empty feedback rows ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


NO_COMMENT_STUB = {
    "issue_category": "No Comment",
    "issue_subcategory": "No Comment",
    "issue_driver": "Not Applicable",
    "sentiment": "Neutral",
    "severity_score": 1,
    "churn_risk": False,
    "root_cause_hypothesis": "No feedback text provided.",
    "recommended_owner": "N/A",
    "suggested_operational_action": "No action required — member did not provide written feedback.",
}


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    subcategory_lines = "\n".join(
        f"  {cat}: {', '.join(subs)}"
        for cat, subs in SUBCATEGORY_MAP.items()
    )
    severity_lines = "\n".join(
        f"  {k}: {v}" for k, v in SEVERITY_RUBRIC.items()
    )
    owner_lines = "\n".join(
        f"  {cat}: {owner}" for cat, owner in CATEGORY_OWNER_MAP.items()
    )
    return f"""You are a data analyst for PreventativeScan, a preventive MRI screening company.
Classify each member NPS feedback response using ONLY the allowed values listed below.

══════════════════════════════════════════════
CATEGORY DEFINITIONS AND DISAMBIGUATION RULES
══════════════════════════════════════════════

"Scheduling / Availability"
Use for: booking, appointment availability, rescheduling, appointment changes by the company, confirmation issues.
Key test: The member's core frustration is about WHEN or WHETHER they could get an appointment.
Do NOT use if the root cause is a website or technology failure — use Technology / Portal instead.
Examples: "Couldn't get an appointment for 3 months" | "Tried to reschedule three times" | "Dallas location fully booked 10+ weeks out"

"Communication / Support Responsiveness"
Use for: slow replies, unanswered calls or emails, poor phone support, support handoff failures, being transferred with no resolution.
Key test: The member contacted support and was ignored, delayed, or bounced around.
Do NOT use for post-scan clinical questions about findings — use Care Coordination / Follow-up instead.
Examples: "Called twice, left messages, no one called back" | "Simple questions take 5–7 days to get answered" | "Transferred three times, then told I'd get a callback — never did"

"Results / Reports"
Use for: delayed results, missing reports, unclear or vague reports, difficulty accessing reports, frustration with results review.
Key test: The member's frustration is specifically about receiving, understanding, or accessing their scan results.
Examples: "Results took three weeks despite being told 7–10 days" | "Never got my results report" | "Portal said error loading document" (when specifically about the results document)

"Technology / Portal"
Use for: website crashes, booking system errors, portal failures, login or access problems, checkout or payment errors, online document errors.
Key test: Root cause is a product or engineering failure, not a capacity or staffing issue.
Disambiguate from Scheduling: "website kept crashing when I tried to book" → Technology / Portal; "no available appointments for 2 months" → Scheduling / Availability.
Examples: "Website kept crashing when I tried to book" | "Booking system showed my slot as available then gave an error" | "Portal keeps saying error loading document"

"Billing / Pricing / Fees"
Use for: refund delays or denials, unexpected fees, double charges, insurance confusion, price complaints.
HIGH PRIORITY: billing issues strongly predict churn — score severity accordingly.
Examples: "Was told I'd get a refund — 8 weeks later, nothing" | "Facility tried to charge me again when I'd already paid" | "Charged a $200 cancellation fee with no warning"

"Facility Experience"
Use for: cleanliness, comfort, temperature, parking, directions, waiting room condition, check-in logistics.
Key test: The issue is about the physical environment of the location.
Examples: "Lobby had trash overflowing from the bins" | "Facility was hard to find" | "Check-in was disorganized"

"Staff / Clinical Experience"
Use for: technician, radiologist, Scan Advisor, staff professionalism, bedside manner, clinical confidence, quality of in-person care.
This category applies to BOTH positive and negative staff comments.
Examples: "Technician made me feel comfortable" | "Staff was dismissive" | "Radiologist was knowledgeable and thorough"

"Care Coordination / Follow-up"
Use for: post-scan follow-up, questions about findings, needed handoff to a clinician, advisor availability, unclear explanation of scan scope or results.
Key test: The issue is clinically sensitive — the member needed qualified guidance after the scan, not just a generic support response.
Disambiguate from Communication: "couldn't reach support about a billing issue" → Communication / Support Responsiveness; "had questions about my scan findings and couldn't get a clinical response" → Care Coordination / Follow-up.
Examples: "Had questions about my results but can't get anyone to call me back" | "Still don't understand what parts of my body were scanned" | "Tried to reach my Scan Advisor about a concerning finding"

"Preparation / Expectations"
Use for: unclear prep instructions, conflicting pre-visit information, mismatch between what was marketed or promised and what was actually delivered.
Examples: "Got three different fasting instructions" | "Not what was marketed to me when I signed up" | "Multiple emails with conflicting information about what to bring"

"Value Perception"
Use for: comments where the primary issue is that the service did not feel worth the premium price, even if technically fine.
Key test: Member explicitly references price, cost, or premium expectations as the core issue.
Examples: "Decent experience but nothing exceptional for the price" | "Expected more wow factor for what I paid" | "Fine, just didn't feel particularly special for a premium service"

"Positive Feedback"
Use when: feedback is mainly praise. Use even if a minor issue is mentioned, as long as the overall tone is positive.
Examples: "Everything was smooth from start to finish" | "Worth every dollar" | "Perfect experience in every way" | "Very impressed with the care and professionalism"

"Other"
Use ONLY when feedback is meaningful but genuinely cannot be assigned to any named category above.
DO NOT default to Other when a more specific category applies.

"No Comment"
Use ONLY for truly blank or empty feedback text.

══════════════════════════════════════════════
CRITICAL RULE — AVOID "Other"
══════════════════════════════════════════════

Before assigning "Other", work through this checklist:
1. Does feedback mention a website, portal, or booking system error? → Technology / Portal
2. Does feedback mention a refund, charge, fee, cancellation policy, or billing? → Billing / Pricing / Fees
3. Is the overall tone positive or complimentary? → Positive Feedback
4. Does feedback mention post-scan questions, scan findings, or needing clinical guidance? → Care Coordination / Follow-up
5. Did the member contact support and get no response, slow response, or a handoff failure? → Communication / Support Responsiveness
6. Does feedback reference price, cost, or value relative to premium expectations? → Value Perception
7. Does feedback mention conflicting or unclear information before the visit? → Preparation / Expectations
8. Does feedback mention the physical facility, cleanliness, parking, or check-in? → Facility Experience

Only assign "Other" if every item above is answered NO.

══════════════════════════════════════════════
FEW-SHOT CLASSIFICATION EXAMPLES
══════════════════════════════════════════════

EXAMPLE 1
Input — nps_score: 5, feedback: "The scan itself was fine but getting scheduled was a nightmare. Website kept crashing when I tried to book. Eventually called but was on hold for 30+ minutes."
Reasoning: Primary barrier is the website crashing during the booking flow — a technology failure, not a capacity problem. Hold time is a secondary frustration.
Output — issue_category: "Technology / Portal", issue_subcategory: "Website crash", issue_driver: "System Error", sentiment: "Negative", severity_score: 3, churn_risk: false

EXAMPLE 2
Input — nps_score: 3, feedback: "Was told I'd receive a partial refund for a service issue. That was 8 weeks ago. No refund, no response to my follow-ups. Starting to think this was a lie."
Reasoning: Core issue is an unresolved refund with no communication over 8 weeks. Classic billing failure with strong churn signal.
Output — issue_category: "Billing / Pricing / Fees", issue_subcategory: "Refund delay", issue_driver: "Process Breakdown", sentiment: "Negative", severity_score: 4, churn_risk: true

EXAMPLE 3
Input — nps_score: 4, feedback: "Tried to reschedule due to being sick and the process was incredibly difficult. Couldn't do it online, had to call, got put on hold, then transferred, then told I'd get a callback. Never got one."
Reasoning: Member wanted to reschedule but was blocked by a support process failure — multiple transfers with no callback. Root cause is support breakdown, not appointment unavailability.
Output — issue_category: "Communication / Support Responsiveness", issue_subcategory: "Support handoff failure", issue_driver: "Poor Communication", sentiment: "Negative", severity_score: 3, churn_risk: false

EXAMPLE 4
Input — nps_score: 5, feedback: "I still don't understand what parts of my body were actually scanned. The information provided is vague and when I asked questions the staff couldn't clearly explain it."
Reasoning: Member needs clinical clarification about their scan — this is post-scan follow-up and clinical question routing. Staff inability to explain scan scope makes this clinically sensitive, not a generic support issue.
Output — issue_category: "Care Coordination / Follow-up", issue_subcategory: "Clinical question routing", issue_driver: "Poor Communication", sentiment: "Negative", severity_score: 3, churn_risk: false

EXAMPLE 5
Input — nps_score: 10, feedback: "Perfect experience in every way. Worth every dollar."
Reasoning: Unambiguously positive, no issues mentioned. Use Positive Feedback, not Other.
Output — issue_category: "Positive Feedback", issue_subcategory: "Strong value perception", issue_driver: "Positive Experience", sentiment: "Positive", severity_score: 1, churn_risk: false

EXAMPLE 6
Input — nps_score: 9, feedback: "Very impressed with the care and professionalism throughout."
Reasoning: Clear praise for staff and clinical experience. This is Positive Feedback, not Other.
Output — issue_category: "Positive Feedback", issue_subcategory: "Professional staff", issue_driver: "Positive Experience", sentiment: "Positive", severity_score: 1, churn_risk: false

EXAMPLE 7
Input — nps_score: 3, feedback: "Very disappointed. Results took three weeks to get back to me despite being told 7-10 days. When I called to check status I got transferred three times and no one could tell me anything."
Reasoning: Primary complaint is delayed results with an explicit time expectation mismatch. The support failure during the follow-up call is secondary. Classify by the primary issue.
Output — issue_category: "Results / Reports", issue_subcategory: "Delayed results", issue_driver: "Delay / Wait Time", sentiment: "Negative", severity_score: 3, churn_risk: false

══════════════════════════════════════════════
ALLOWED VALUES (use exactly as listed)
══════════════════════════════════════════════

ALLOWED ISSUE_CATEGORIES:
{chr(10).join(f'  - {c}' for c in ISSUE_CATEGORIES)}

ALLOWED SUBCATEGORIES (by category):
{subcategory_lines}

ALLOWED ISSUE_DRIVERS:
{chr(10).join(f'  - {d}' for d in ISSUE_DRIVERS)}

ALLOWED SENTIMENT VALUES: {', '.join(SENTIMENT_LABELS)}

SEVERITY RUBRIC (1–5):
{severity_lines}

RECOMMENDED OWNERS:
{owner_lines}

OUTPUT RULES:
- severity_score must be an integer 1–5 matching the rubric exactly.
- churn_risk is true only for severity 4–5 OR explicit statements of leaving/not returning.
- root_cause_hypothesis must be based solely on the feedback text — do not claim certainty.
- suggested_operational_action must be 1–2 sentences using action verbs: review, investigate, audit, compare, monitor, clarify, follow up, escalate. Do not give medical advice.
- Return ONLY valid JSON. No prose outside the JSON structure."""


def _build_batch_prompt(batch: list[dict]) -> str:
    items = "\n\n".join(
        f'{i + 1}. response_id: {row["response_id"]}\n'
        f'   nps_score: {row["nps_score"]}\n'
        f'   feedback: {row["feedback_text"]}'
        for i, row in enumerate(batch)
    )
    return (
        f"Analyze the following {len(batch)} NPS feedback response(s).\n\n"
        "Before classifying each response, check whether it clearly fits a specific "
        "named category (Technology / Portal, Billing / Pricing / Fees, Positive Feedback, "
        "Care Coordination / Follow-up, Results / Reports, etc.) before assigning 'Other'. "
        "Reserve 'Other' only for feedback that genuinely cannot be assigned to any named category.\n\n"
        "Return a JSON array where each element corresponds to the numbered input "
        "in the same order. Each element must include the field 'response_id' plus "
        "all required classification fields.\n\n"
        f"{items}\n\n"
        "Return ONLY a JSON array. Example structure:\n"
        '[\n'
        '  {\n'
        '    "response_id": "RESP_001",\n'
        '    "issue_category": "...",\n'
        '    "issue_subcategory": "...",\n'
        '    "issue_driver": "...",\n'
        '    "sentiment": "...",\n'
        '    "severity_score": 3,\n'
        '    "churn_risk": false,\n'
        '    "root_cause_hypothesis": "...",\n'
        '    "recommended_owner": "...",\n'
        '    "suggested_operational_action": "..."\n'
        '  }\n'
        ']'
    )


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from model output, tolerating surrounding prose."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _validate_result(result: dict) -> dict:
    """
    Coerce types and fill missing fields with safe defaults so a partial
    response never causes a downstream crash.
    """
    category = result.get("issue_category", "Other")
    if category not in ISSUE_CATEGORIES:
        category = "Other"

    subcategory = result.get("issue_subcategory", "Other")
    allowed_subs = SUBCATEGORY_MAP.get(category, ["Other"])
    if subcategory not in allowed_subs:
        subcategory = allowed_subs[0]

    driver = result.get("issue_driver", "Other")
    if driver not in ISSUE_DRIVERS:
        driver = "Other"

    sentiment = result.get("sentiment", "Neutral")
    if sentiment not in SENTIMENT_LABELS:
        sentiment = "Neutral"

    try:
        severity = int(result.get("severity_score", 2))
        severity = max(1, min(5, severity))
    except (TypeError, ValueError):
        severity = 2

    churn_risk = bool(result.get("churn_risk", False))
    owner = CATEGORY_OWNER_MAP.get(category, "Member Experience")

    return {
        "issue_category": category,
        "issue_subcategory": subcategory,
        "issue_driver": driver,
        "sentiment": sentiment,
        "severity_score": severity,
        "churn_risk": churn_risk,
        "root_cause_hypothesis": str(result.get("root_cause_hypothesis", ""))[:500],
        "recommended_owner": owner,
        "suggested_operational_action": str(
            result.get("suggested_operational_action", "")
        )[:500],
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path, "r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Cache file unreadable — starting fresh.")
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)


# ── Core analysis function ────────────────────────────────────────────────────

def run_analysis(
    df: pd.DataFrame,
    api_key: str,
    cache_path: Path = DEFAULT_CACHE_PATH,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Analyse all rows in df that have not been cached yet.

    - Rows with empty feedback_text are classified without an API call.
    - All other rows are batched in groups of BATCH_SIZE and sent to Claude.
    - Results are cached after each batch (crash-safe).
    - Returns df with the nine AI-output columns merged in.

    progress_callback(completed, total) is called after each batch if provided.
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt()
    cache = _load_cache(cache_path)

    # Rows that still need analysis
    needs_analysis = df[~df["response_id"].isin(cache)].copy()
    total_new = len(needs_analysis)
    completed = 0

    if total_new > 0:
        logger.info("Analysing %d new responses (batch size %d).", total_new, BATCH_SIZE)

    # ── Empty feedback: no API call ──────────────────────────────────────────
    empty_mask = needs_analysis["feedback_text"].eq("")
    for rid in needs_analysis.loc[empty_mask, "response_id"]:
        stub = NO_COMMENT_STUB.copy()
        stub["analysis_timestamp"] = _now_iso()
        cache[rid] = stub
    needs_analysis = needs_analysis[~empty_mask]

    # ── Batched API calls ────────────────────────────────────────────────────
    rows_list = needs_analysis.to_dict("records")
    for batch_start in range(0, len(rows_list), BATCH_SIZE):
        batch = rows_list[batch_start : batch_start + BATCH_SIZE]
        batch_ids = [r["response_id"] for r in batch]

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": _build_batch_prompt(batch)}],
                )
                raw_text = response.content[0].text
                results = _extract_json_array(raw_text)
                break
            except anthropic.RateLimitError:
                logger.warning("Rate limit hit — waiting %.1fs", RETRY_DELAY * 2)
                time.sleep(RETRY_DELAY * 2)
                results = []
            except Exception as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                if attempt == MAX_RETRIES:
                    results = []
                else:
                    time.sleep(RETRY_DELAY)

        # Map results back to response_ids
        result_by_id = {}
        for item in results:
            if isinstance(item, dict) and "response_id" in item:
                result_by_id[item["response_id"]] = _validate_result(item)

        # Fill any missing IDs with a stub so the cache is always complete
        ts = _now_iso()
        for rid in batch_ids:
            if rid not in result_by_id:
                logger.warning("No result returned for %s — using fallback stub.", rid)
                result_by_id[rid] = {
                    "issue_category": "Other",
                    "issue_subcategory": "Other",
                    "issue_driver": "Other",
                    "sentiment": "Neutral",
                    "severity_score": 2,
                    "churn_risk": False,
                    "root_cause_hypothesis": "Could not be classified.",
                    "recommended_owner": "Member Experience",
                    "suggested_operational_action": "Review manually.",
                    "_parse_error": True,
                }
            result_by_id[rid]["analysis_timestamp"] = ts
            cache[rid] = result_by_id[rid]

        _save_cache(cache, cache_path)
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, total_new)
        logger.debug("Batch complete: %d/%d", completed, total_new)

    # ── Merge cache into DataFrame ───────────────────────────────────────────
    analysis_records = [
        {"response_id": rid, **fields} for rid, fields in cache.items()
    ]
    analysis_df = pd.DataFrame(analysis_records)

    enriched = df.merge(analysis_df, on="response_id", how="left")

    # Any rows that ended up without analysis (shouldn't happen) get safe defaults
    for col, default in [
        ("issue_category", "Other"),
        ("issue_subcategory", "Other"),
        ("issue_driver", "Other"),
        ("sentiment", "Neutral"),
        ("severity_score", 2),
        ("churn_risk", False),
        ("root_cause_hypothesis", ""),
        ("recommended_owner", "Member Experience"),
        ("suggested_operational_action", ""),
    ]:
        if col not in enriched.columns:
            enriched[col] = default
        else:
            enriched[col] = enriched[col].fillna(default)

    enriched["severity_score"] = pd.to_numeric(
        enriched["severity_score"], errors="coerce"
    ).fillna(2).astype(int)
    enriched["churn_risk"] = enriched["churn_risk"].astype(bool)

    return enriched


def load_from_cache(
    df: pd.DataFrame, cache_path: Path = DEFAULT_CACHE_PATH
) -> pd.DataFrame:
    """
    Merge cached analysis results into df without making any API calls.
    Rows not in the cache get safe default values so callers never see NaNs
    in analysis columns.
    """
    cache = _load_cache(cache_path)

    if cache:
        analysis_records = [
            {"response_id": rid, **fields} for rid, fields in cache.items()
        ]
        analysis_df = pd.DataFrame(analysis_records)
        enriched = df.merge(analysis_df, on="response_id", how="left")
    else:
        enriched = df.copy()

    defaults = [
        ("issue_category", "Other"),
        ("issue_subcategory", "Other"),
        ("issue_driver", "Other"),
        ("sentiment", "Neutral"),
        ("severity_score", 2),
        ("churn_risk", False),
        ("root_cause_hypothesis", ""),
        ("recommended_owner", "Member Experience"),
        ("suggested_operational_action", ""),
        ("analysis_timestamp", ""),
    ]
    for col, default in defaults:
        if col not in enriched.columns:
            enriched[col] = default
        else:
            enriched[col] = enriched[col].fillna(default)

    enriched["severity_score"] = (
        pd.to_numeric(enriched["severity_score"], errors="coerce").fillna(2).astype(int)
    )
    enriched["churn_risk"] = enriched["churn_risk"].astype(bool)

    return enriched


def ids_analyzed_on(
    target_date: str, cache_path: Path = DEFAULT_CACHE_PATH
) -> list[str]:
    """
    Return response_ids whose analysis_timestamp falls on target_date (YYYY-MM-DD).
    Useful for showing "analyzed today" counts in the UI.
    """
    cache = _load_cache(cache_path)
    result = []
    for rid, fields in cache.items():
        ts = fields.get("analysis_timestamp", "")
        if ts.startswith(target_date):
            result.append(rid)
    return result


def cache_stats(cache_path: Path = DEFAULT_CACHE_PATH) -> dict:
    """Return summary statistics about the current cache."""
    cache = _load_cache(cache_path)
    total = len(cache)
    errors = sum(1 for v in cache.values() if v.get("_parse_error"))
    no_comment = sum(
        1 for v in cache.values() if v.get("issue_category") == "No Comment"
    )
    return {"total_cached": total, "parse_errors": errors, "no_comment": no_comment}
