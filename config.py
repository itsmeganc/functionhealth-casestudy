"""
All taxonomy constants, rubrics, mappings, and thresholds for the
PreventativeScan Automated Feedback Analysis Tool.
"""

ISSUE_CATEGORIES = [
    "Scheduling / Availability",
    "Communication / Support Responsiveness",
    "Results / Reports",
    "Technology / Portal",
    "Billing / Pricing / Fees",
    "Facility Experience",
    "Staff / Clinical Experience",
    "Care Coordination / Follow-up",
    "Preparation / Expectations",
    "Value Perception",
    "Positive Feedback",
    "Other",
    "No Comment",
]

SUBCATEGORY_MAP = {
    "Scheduling / Availability": [
        "Appointment availability",
        "Rescheduling difficulty",
        "Appointment changed by company",
        "Booking confirmation issue",
        "Location availability",
        "Booking abandonment",
        "Other scheduling issue",
    ],
    "Communication / Support Responsiveness": [
        "No response",
        "Slow response",
        "Unclear communication",
        "Phone support issue",
        "Email support issue",
        "Support handoff failure",
        "Other communication issue",
    ],
    "Results / Reports": [
        "Delayed results",
        "Missing report",
        "Report access issue",
        "Unclear results explanation",
        "Rushed results review",
        "Concerning finding follow-up",
        "Other results issue",
    ],
    "Technology / Portal": [
        "Booking system error",
        "Portal access issue",
        "Checkout or payment error",
        "Results document error",
        "System record mismatch",
        "Website crash",
        "Other technology issue",
    ],
    "Billing / Pricing / Fees": [
        "Refund delay",
        "Unexpected fee",
        "Double charge",
        "Insurance issue",
        "Pricing concern",
        "Cancellation policy concern",
        "Other billing issue",
    ],
    "Facility Experience": [
        "Cleanliness",
        "Comfort",
        "Directions or wayfinding",
        "Check-in process",
        "Waiting room",
        "Parking or location access",
        "Other facility issue",
    ],
    "Staff / Clinical Experience": [
        "Technician experience",
        "Radiologist experience",
        "Scan Advisor experience",
        "Staff professionalism",
        "Dismissive interaction",
        "Clinical confidence",
        "Other staff or clinical issue",
    ],
    "Care Coordination / Follow-up": [
        "Post-scan follow-up",
        "Clinical question routing",
        "Concerning finding escalation",
        "Advisor availability",
        "Next-step guidance",
        "Other care coordination issue",
    ],
    "Preparation / Expectations": [
        "Prep instructions unclear",
        "Conflicting information",
        "Expectation mismatch",
        "Pre-visit communication",
        "Marketing promise mismatch",
        "Other preparation issue",
    ],
    "Value Perception": [
        "Not worth price",
        "Expected more premium experience",
        "Service felt generic",
        "Strong value perception",
        "Other value perception issue",
    ],
    "Positive Feedback": [
        "Smooth end-to-end experience",
        "Fast results",
        "Professional staff",
        "Clinical reassurance",
        "Facility quality",
        "Strong value perception",
        "Other positive feedback",
    ],
    "Other": ["Other"],
    "No Comment": ["No Comment"],
}

ISSUE_DRIVERS = [
    "Delay / Wait Time",
    "Access / Availability",
    "Process Breakdown",
    "Poor Communication",
    "System Error",
    "Unexpected Cost",
    "Quality Concern",
    "Expectation Mismatch",
    "Positive Experience",
    "Other",
    "Not Applicable",
]

SENTIMENT_LABELS = ["Positive", "Neutral", "Mixed", "Negative"]

SEVERITY_RUBRIC = {
    1: "Positive feedback or very minor friction with no clear operational impact.",
    2: "Low-impact inconvenience or mild dissatisfaction.",
    3: "Meaningful friction that affected the member experience but does not appear urgent.",
    4: "Serious issue affecting trust, access, service completion, billing, or follow-up.",
    5: (
        "Critical issue involving major service failure, repeated failed support attempts, "
        "high churn risk, safety/trust concern, or severe billing/access problem."
    ),
}

CATEGORY_OWNER_MAP = {
    "Scheduling / Availability": "Operations / Scheduling Ops",
    "Communication / Support Responsiveness": "Member Experience / Support Ops",
    "Results / Reports": "Clinical Ops / Product",
    "Technology / Portal": "Product / Engineering",
    "Billing / Pricing / Fees": "Billing Ops / Member Experience",
    "Facility Experience": "Site Operations",
    "Staff / Clinical Experience": "Clinical Ops / Site Operations",
    "Care Coordination / Follow-up": "Clinical Ops / Member Experience",
    "Preparation / Expectations": "Member Experience / Operations",
    "Value Perception": "Member Experience / Product Strategy",
    "Positive Feedback": "Member Experience",
    "Other": "Member Experience",
    "No Comment": "N/A",
}

# ── NPS thresholds ──────────────────────────────────────────────────────────
NPS_PROMOTER_MIN = 9
NPS_PASSIVE_MIN = 7
# Detractor: 0–6

# ── Alert thresholds ────────────────────────────────────────────────────────
ALERT_CRITICAL_SEVERITY = 5
ALERT_CRITICAL_AVG_SEVERITY = 3.0      # avg severity threshold when above UCL
ALERT_CRITICAL_AVG_SEVERITY_ABS = 4.0  # avg severity threshold (absolute, no UCL needed)
ALERT_CRITICAL_MIN_RESPONSES = 2

ALERT_WARNING_AVG_SEVERITY = 3.5
ALERT_WARNING_MIN_RESPONSES = 2
ALERT_WARNING_DETRACTOR_RATE = 0.40    # 40%
ALERT_WARNING_CONSECUTIVE_WEEKS = 2

ALERT_WATCHLIST_LOW_VOLUME_MAX = 5     # "low volume" cutoff
ALERT_WATCHLIST_SEVERITY_FLOOR = 4    # severity floor for low-volume watchlist

# ── Control chart ────────────────────────────────────────────────────────────
CONTROL_CHART_MIN_WEEKS = 12   # minimum weeks of data before drawing UCL/LCL
CONTROL_CHART_MIN_N = 5        # minimum weekly responses for a valid p-chart point

# ── Location flags ───────────────────────────────────────────────────────────
LOCATION_DEVIATION_THRESHOLD = 0.50   # 50% above network average rate
LOCATION_DEVIATION_MIN_COUNT = 3      # minimum responses to flag deviation
LOCATION_PERSISTENCE_WEEKS = 3        # consecutive weeks in top-3 to flag persistence
LOCATION_HEALTH_SEVERITY_FLOOR = 4    # severity floor for health score weighting

# ── Additional analysis ──────────────────────────────────────────────────────
VELOCITY_WEEKS_LOOKBACK = 2           # weeks to compute rate-of-change over
COOCCURRENCE_MIN_COUNT = 2            # minimum co-occurrences to show in matrix

# ── Daily alert thresholds ───────────────────────────────────────────────────
DAILY_CRITICAL_SEVERITY = 5          # severity score that triggers same-day alert
DAILY_CHURN_RISK_MIN_SEVERITY = 4   # churn_risk=True responses at this severity also fire daily

# ── Categories excluded from detractor-rate alert logic ──────────────────────
ALERT_EXCLUDE_CATEGORIES = {"Positive Feedback", "No Comment"}

# ── Categories that are operational (not positive/neutral noise) ──────────────
OPERATIONAL_CATEGORIES = [
    c for c in ISSUE_CATEGORIES
    if c not in {"Positive Feedback", "No Comment", "Other"}
]
