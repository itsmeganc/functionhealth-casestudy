# PreventativeScan — Automated Feedback Analysis Tool

A Streamlit prototype that ingests NPS member feedback, uses Claude AI to classify every response, and surfaces emerging issues through an interactive dashboard with alerting, statistical process control, and location-level analysis.

Built with Claude Code for the PreventativeScan NPS Analyst Case Study.

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Place the data file

The default dataset should be at:

```
data/member_feedback.csv
```

The file is already included. Alternatively, upload any compatible CSV directly from the app sidebar.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## First-Time Usage

1. **Enter your API key** in the sidebar under "PreventativeScan API Key" (or the app reads it from `.env` automatically).
2. Click **"Run analysis"** — the app will classify all unanalyzed responses using Claude. Results are cached so subsequent loads are instant.
3. Use the **sidebar filters** (date range, location) to scope the dashboard.

> Analysis of 500 responses takes ~3–5 minutes on first run. A local cache (`cache/analysis_cache.json`) persists results across sessions so you only pay the API cost once.

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| **Briefing** | KPI strip (NPS, volume, promoters, detractors, avg severity) with prior-period deltas · daily drill-down · sentiment over time · issue velocity · issue distribution |
| **Top Issues** | Top 5 categories ranked by avg severity · subcategory donut charts · per-subcategory root cause & suggested action summaries · example quotes · owner routing |
| **Responses** | Filterable table of every response with all AI-generated fields |
| **Control Charts** | p-chart SPC per issue category · active out-of-control signal directory · location filtering |
| **Alerts** | Critical / Warning / Watchlist / Location alerts with recommended actions |
| **Segments** | Location Health scores · Issue Deviation (locations exceeding network avg by ≥50%) |
| **Promoters** | What promoters highlight most · promoter driver analysis |
| **Export** | Download enriched CSV/Excel, daily digest, weekly digest, alerts, location analysis |

---

## File Structure

```
├── app.py                  # Streamlit UI — all tab rendering
├── analyzer.py             # Claude API integration, batch analysis, cache management
├── alerts.py               # Alert generation (Critical / Warning / Watchlist / Location)
├── config.py               # Taxonomy constants, rubrics, thresholds
├── control_charts.py       # p-chart SPC computation and Plotly chart builder
├── export.py               # CSV/Excel export builders
├── loader.py               # CSV ingestion, validation, enrichment
├── metrics.py              # All pandas metric computations
├── summaries.py            # Daily and weekly digest generators
├── theme.py                # CSS injection, color constants, UI helpers
├── purge_other.py          # CLI utility to re-classify "Other" responses
├── data/
│   └── member_feedback.csv # Default dataset (500 responses)
├── cache/
│   └── analysis_cache.json # Persisted AI classifications (auto-created)
├── .env.example            # API key template
└── requirements.txt
```

---

## AI Classification Schema

Each response is classified by Claude with the following fields:

| Field | Description |
|---|---|
| `issue_category` | One of 12 operational categories (Scheduling, Communication, Billing, etc.) |
| `issue_subcategory` | Specific sub-type within the category |
| `issue_driver` | Root structural cause (Delay, Access, Process Breakdown, etc.) |
| `sentiment` | Positive / Neutral / Mixed / Negative |
| `severity_score` | 1–5 (1 = minor friction, 5 = critical failure / safety concern) |
| `churn_risk` | Boolean — high likelihood of member loss |
| `root_cause_hypothesis` | AI-generated hypothesis for the operational root cause |
| `recommended_owner` | Team responsible for resolution |
| `suggested_operational_action` | Specific recommended next step |

---

## Alert Logic

**Critical** — act today:
- Any severity-5 response present
- Category avg severity ≥ 4.0 (≥ 2 responses)
- UCL breach on p-chart + avg severity ≥ 3.0

**Warning** — investigate this week:
- Category avg severity ≥ 3.5
- Issue rate above centerline for ≥ 2 consecutive weeks (run rule)

**Watchlist** — monitor:
- Rate above centerline but below UCL (most recent week)
- Low volume (≤ 5 responses) but includes severity ≥ 4

**Location** — location-specific:
- Location rate ≥ 50% above network average (≥ 3 responses)
- Location persistent top-3 contributor for ≥ 3 consecutive weeks

---

## Utility: Re-classify "Other" Responses

If some responses land in "Other" due to ambiguous feedback, re-run classification:

```bash
python purge_other.py            # dry run — shows what would be removed
python purge_other.py --confirm  # clears them from cache; re-run analysis in app
```

---

## Notes

- The app uses `claude-sonnet-4-6` by default. Model can be changed in `analyzer.py`.
- Analysis is batched (10 responses per API call) to minimize latency and cost.
- All visualizations use Plotly; the app is fully interactive with hover tooltips.
