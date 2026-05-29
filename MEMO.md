# PreventativeScan Feedback Analysis Tool — Documentation Memo

**To:** Barbara Chirila, Martine White  **·**  **From:** Megan Carroll  **·**  **Date:** May 22, 2026

---

## What It Does

The tool ingests the NPS feedback CSV, runs every response through Claude AI, and classifies it across 12 operational categories with severity scores, sentiment, churn risk flags, root cause hypotheses, and recommended team owners. Everything surfaces in an interactive dashboard built for daily monitoring — load new responses, run analysis, and within minutes you can see what's broken, where it's happening, and who should own the fix. Live demo (no setup required): https://functionhealth-casestudy-o3nfmsemkrfsbei8wyeauz.streamlit.app

## How to Use It

Start on the **Briefing tab** — KPI strip with prior-period deltas, sentiment trends, and active alerts. **Top Issues** shows the 5 highest-severity categories with subcategory breakdowns, root causes, and suggested actions. **Alerts** routes Critical / Warning / Watchlist items to the responsible team. **Segments** flags locations running ≥50% above the network average on any issue category. **Export** downloads the full enriched dataset or alerts as CSV/Excel. Full setup instructions are in the README at https://github.com/itsmeganc/functionhealth-casestudy.

## Output of the Analysis

Running the tool on the 500-response dataset surfaced several clear patterns. Scheduling and Communication issues drove the highest severity scores and generated the most Critical alerts — pointing to booking system failures and unresponsive support as the primary churn drivers. Multiple locations showed statistically significant deviations from the network average, with certain cities consistently over-indexing on specific issue categories week over week. The control charts flagged out-of-control signals in Staff/Clinical Experience at select locations, indicating localized problems rather than systemic ones. Promoter responses clustered around fast results, clinical reassurance, and smooth end-to-end experiences — a clear signal of what's working. Full classified output and alerts are included as attachments.

## Technical Decisions

**Claude API for classification:** Few-shot prompting with category definitions and worked examples produces consistent categorization that smaller open-source models can't reliably match for nuanced healthcare feedback.

**p-chart SPC for anomaly detection:** I chose statistical process control over static thresholds because static thresholds break at scale — a 5% cutoff set today is miscalibrated in six months as volume grows. SPC calculates limits from historical baselines and self-adjusts as data accumulates, making it a durable monitoring tool for a scaling company rather than one requiring constant manual recalibration.

**How AI was used:** I brought my own frameworks to the key decisions — choosing SPC, defining alert tier logic, designing the issue taxonomy. Claude handled implementation across all modules. The interface went through significant iteration; I caught several real logic errors along the way that required diagnosis. The result is a tool far more complete than what I could have built solo in this timeframe.

## Production Roadmap

**Automate:** Schedule daily analysis post-survey export; add Slack alerts so Critical issues reach the right team same-day.  
**Integrate:** Connect to the survey platform API directly; link member IDs to operational data (scan date, location, wait time) for richer root cause analysis.  
**Close the loop:** Auto-create tickets routed to the right team when alerts fire; track resolution time and retest NPS to measure whether interventions worked.  
**Refine:** Systematically spot-check AI categorization, correct where it gets it wrong, and build toward a churn prediction model once the labeled data is trustworthy.

## How I'd Use This to Improve the Member Experience

The daily workflow starts on the Briefing tab. Critical alerts go to the team owner the same morning with example quotes and suggested actions already surfaced — no manual triage required. But the real value is in using the output to make deliberate UX changes, not just react to individual complaints. If Scheduling issues are spiking at a specific location, that's a prompt to audit the booking flow there — is it a capacity problem, a system bug, or a communication gap at confirmation? If Communication issues trend upward network-wide for two consecutive weeks, that's a signal to review the post-booking and post-scan touchpoint experience. Promoter feedback tells you what to protect and amplify: if members consistently cite fast results and clinical reassurance as reasons they'd recommend PreventativeScan, those become non-negotiable quality bars as the company scales. The goal is compressing the feedback-to-action loop from weeks to hours — and making sure that every operational and experience decision is grounded in what members are actually saying.

---

Built using Claude Code (Anthropic).
