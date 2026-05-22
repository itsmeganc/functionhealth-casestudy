# PreventativeScan Feedback Analysis Tool — Documentation Memo

**To:** Barbara Chirila, Martine White
**From:** Megan Carroll
**Date:** May 22, 2026
**Re:** Automated Feedback Analysis Tool — Submission

---

## What It Does

The tool takes the 500-response NPS CSV, runs every piece of feedback through Claude AI, and classifies it across 12 operational categories (Scheduling, Communication, Billing, Clinical, etc.) — assigning severity scores, sentiment, churn risk flags, root cause hypotheses, and recommended team owners. Everything surfaces in an interactive dashboard built for daily use: load new responses, run analysis, and within minutes you can see what's broken, where it's happening, and who should own the fix.

---

## How to Use It

A live version of the tool is hosted at the link included in this submission. All 500 responses have been pre-analyzed and the full dashboard is immediately accessible — no API key or setup required to explore it.

To run it locally:
1. `pip install -r requirements.txt` → add Anthropic API key to `.env` → `streamlit run app.py`
2. Click **Run Analysis** on first load (~3–5 min for 500 responses; cached after that)
3. Use the **Briefing tab** as the daily starting point — KPI strip with prior-period deltas, sentiment trends, issue velocity, and active alert callouts
4. **Top Issues tab** shows the 5 highest-severity categories with subcategory breakdowns and example quotes
5. **Alerts tab** routes Critical / Warning / Watchlist items to the responsible team
6. **Segments tab** flags locations deviating ≥50% above the network average
7. **Export tab** downloads the full enriched dataset, daily digest, or alerts as CSV/Excel for stakeholder distribution

---

## Technical Decisions

**Claude API over open-source models:** Accuracy and few-shot classification quality significantly outperforms smaller open models for nuanced healthcare feedback. The prompt includes category definitions, disambiguation rules, and worked examples to minimize ambiguous fallbacks and ensure consistent categorization across response types.

**Batch caching:** Each response is classified once and stored locally. Re-runs only process new responses, making daily incremental ingestion fast and cost-efficient (< $0.10 for 500 responses).

**p-chart statistical process control for anomaly detection:** Rather than static thresholds (e.g., ">5% of feedback"), the tool uses p-chart SPC — a methodology designed for monitoring proportions over time with variable sample sizes. Control limits are calculated from historical weekly baselines so normal seasonal variation doesn't trigger false alarms; a week only fires as anomalous when it's statistically unlikely (~3σ), not just numerically elevated. This matters especially for a scaling company: as scan volume grows from 1K to 20K per month, raw issue counts become meaningless without normalization, and a static 5% threshold set today will be miscalibrated in six months. SPC adapts — wider limits when weekly volume is low (more uncertainty), tighter limits as volume grows (more precision). The more weeks of data that accumulate, the more stable and trustworthy the baselines become. With 6–12 months of history, the charts will reliably distinguish genuine operational deterioration from normal noise, making them a durable monitoring tool rather than one that needs constant manual recalibration.

**Modular architecture:** `config.py` owns all thresholds and taxonomy constants — changing an alert threshold or adding a new issue category is a one-line edit, not a code change scattered across multiple files.

**Streamlit for UI:** Fastest path to a fully interactive, shareable prototype that a non-technical operator can use without any setup beyond running one command. The interface supports filtering, drill-downs, and exports without requiring any separate tooling.

## How AI Was Used to Build This

This tool was built end-to-end using Claude Code (Anthropic's AI coding assistant). AI was used at every stage:

**Planning:** I brought my own frameworks and experience to the architectural decisions before writing a line of code — choosing statistical process control for anomaly detection, defining the alert tier logic, and designing the issue taxonomy for a preventive imaging context. Claude helped translate those decisions into a concrete implementation plan: what modules to separate, how to wire the alert engine, and how to structure the data pipeline.

**Building:** Claude wrote the majority of the implementation across all modules — the CSV parser, the Claude API integration and prompt engineering, the metrics computations, the SPC p-chart logic, the alert engine, and the full Streamlit dashboard. I directed what to build and why, reviewed the output, and caught errors.

**Iterating and editing:** The interface went through significant iteration — refining how charts were displayed, reordering sections, adding tooltip definitions, fixing edge cases in the alert logic, and tuning the AI classification prompt to reduce miscategorization. Claude handled each change based on my direction.

**Tradeoffs:** The main tradeoff is that AI moves fast but needs careful review — generated code can be subtly wrong in ways that aren't immediately obvious. I caught several logic errors along the way (incorrect period comparisons in the KPI deltas, chart baselines computed on filtered instead of full data, gaps in how alerts were deduplicated) that required real diagnosis to fix. The upside is that the tool is significantly more complete than what I could have built solo in this timeframe — AI handled implementation, I focused on the product decisions and making sure the output was actually correct and useful.

---

## Production Roadmap

**Phase 1 — Automation:** Move analysis from on-demand to scheduled (daily cron job post-survey export). Add webhook/Slack integration for Critical alerts so the on-call team is notified same-day without opening the dashboard.

**Phase 2 — Integration:** Connect directly to the survey platform API (Medallia, Qualtrics, etc.) to eliminate manual CSV exports. Add member ID linkage to correlate NPS feedback with operational data (scan date, location, technician, wait time) for richer root cause analysis.

**Phase 3 — Closed Loop:** Build a case management layer — when an alert fires, auto-create a Jira/Linear ticket routed to the right team with the AI-generated root cause hypothesis pre-populated. Track resolution time and retest NPS at the location/category level to measure whether interventions are working.

**Phase 4 — AI Refinement & Predictive:** Before any of this is truly production-ready, the AI's categorization needs to be monitored and refined — regularly spot-checking tagged responses, identifying where the model gets it wrong, and tightening the prompt or adding examples to correct it. The tagging is only useful if it's trustworthy; that's an ongoing process, not a one-time setup. Once classification quality is validated, the accumulated labeled data becomes the foundation for a lightweight churn prediction model — flagging members likely to leave before they submit a detractor response and enabling proactive outreach.

---

## How I'd Use This Daily

Each morning, the Briefing tab is the first stop. The KPI strip immediately shows whether NPS, volume, and severity moved overnight vs. the prior equivalent period. If any Critical alerts are active, the red banner fires — that means a severity-5 response or a sustained high-severity category that needs same-day escalation to the relevant team owner.

The practical workflow:
- **Critical alerts → same-day Slack to team owner** with the example quotes and suggested action pre-filled from the AI output
- **Warning alerts → weekly ops review agenda item** — the category is trending but not yet acute; the team lead reviews root cause hypotheses and assigns an investigation
- **Location deviations → site operations call** — if Houston is running 60% above network average on Scheduling, that's a staffing or booking system problem specific to that location, not a systemic issue
- **Promoter drivers → product/marketing input** — what promoters praise (fast results, clinical reassurance, staff quality) informs what to double down on in member communications and service design

The goal is to compress the feedback-to-action loop from weeks (ad hoc survey reviews) to hours (automated daily triage with pre-routed ownership).

---

Built using Claude Code (Anthropic). All source code, README, and this memo are included in the submission package.
