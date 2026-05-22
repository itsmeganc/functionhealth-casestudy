"""
Function Health-inspired color palette and CSS for the PreventativeScan
Feedback Analysis Tool.

Primary brand color extracted from Function Health homepage:
  burnt orange / terracotta #C4573A (banner, CTAs, chat bubble)
"""

# ── Brand palette ─────────────────────────────────────────────────────────────
PRIMARY = "#C4573A"
PRIMARY_DARK = "#A6432C"
PRIMARY_LIGHT = "#E8D5CE"

DARK = "#1A1A1A"
MEDIUM_GRAY = "#6B7280"
LIGHT_GRAY = "#E5E1DC"
CREAM = "#F8F4F0"
WHITE = "#FFFFFF"

# ── Alert tier colors ─────────────────────────────────────────────────────────
CRITICAL_BG = "#FEF2F2"
CRITICAL_BORDER = "#EF4444"
CRITICAL_TEXT = "#991B1B"

WARNING_BG = "#FFFBEB"
WARNING_BORDER = "#F59E0B"
WARNING_TEXT = "#92400E"

WATCHLIST_BG = "#FFFBEB"
WATCHLIST_BORDER = "#D97706"
WATCHLIST_TEXT = "#78350F"

LOCATION_BG = "#EFF6FF"
LOCATION_BORDER = "#3B82F6"
LOCATION_TEXT = "#1E3A8A"

POSITIVE_BG = "#F0FDF4"
POSITIVE_BORDER = "#22C55E"
POSITIVE_TEXT = "#14532D"

# ── Chart categorical palette ─────────────────────────────────────────────────
CHART_COLORS = [
    "#C4573A",  # brand orange
    "#1A1A1A",  # dark
    "#4A7C59",  # muted green
    "#D4870A",  # amber
    "#3B82F6",  # blue
    "#7C3AED",  # purple
    "#0891B2",  # teal
    "#6B7280",  # gray
    "#E879A0",  # pink
    "#84CC16",  # lime
]

# Severity score color scale (1 = green, 5 = red)
SEVERITY_COLORS = {
    1: "#22C55E",
    2: "#84CC16",
    3: "#F59E0B",
    4: "#F97316",
    5: "#EF4444",
}


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css() -> str:
    """Return a <style> block to inject into the Streamlit app."""
    return f"""
<style>
/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {WHITE};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {CREAM};
    border-right: 1px solid {LIGHT_GRAY};
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {DARK};
    font-weight: 600;
}}

/* ── Top-level headings ── */
h1 {{ color: {DARK}; font-weight: 700; letter-spacing: -0.5px; }}
h2 {{ color: {DARK}; font-weight: 600; }}
h3 {{ color: {DARK}; font-weight: 600; }}

/* ── Metric tiles ── */
[data-testid="stMetric"] {{
    background: {WHITE};
    border: 1px solid {LIGHT_GRAY};
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricLabel"] {{
    color: {MEDIUM_GRAY} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{
    color: {DARK} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}}

/* ── Tab bar ── */
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 2px solid {LIGHT_GRAY};
    gap: 0;
}}
[data-testid="stTabs"] [role="tab"] {{
    color: {MEDIUM_GRAY};
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.15s;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {PRIMARY} !important;
    font-weight: 600;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    color: {PRIMARY};
    background: {CREAM};
}}

/* ── Buttons ── */
[data-testid="stButton"] > button {{
    background-color: {PRIMARY};
    color: {WHITE};
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: background-color 0.15s;
}}
[data-testid="stButton"] > button:hover {{
    background-color: {PRIMARY_DARK};
}}
[data-testid="stDownloadButton"] > button {{
    background-color: {WHITE};
    color: {PRIMARY};
    border: 1.5px solid {PRIMARY};
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background-color: {PRIMARY};
    color: {WHITE};
}}

/* ── Expanders ── */
details summary {{
    font-weight: 500;
    color: {DARK};
}}
details summary:hover {{
    color: {PRIMARY};
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {LIGHT_GRAY};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Section divider ── */
hr {{
    border: none;
    border-top: 1px solid {LIGHT_GRAY};
    margin: 1.5rem 0;
}}
</style>
"""


# ── Alert box HTML builders ───────────────────────────────────────────────────

def alert_box(
    tier: str,
    title: str,
    body: str,
    meta: str = "",
) -> str:
    """Return an HTML alert box styled for the given tier."""
    config = {
        "Critical": (CRITICAL_BG, CRITICAL_BORDER, CRITICAL_TEXT, "🔴"),
        "Warning": (WARNING_BG, WARNING_BORDER, WARNING_TEXT, "🟡"),
        "Watchlist": (WATCHLIST_BG, WATCHLIST_BORDER, WATCHLIST_TEXT, "🟠"),
        "Location": (LOCATION_BG, LOCATION_BORDER, LOCATION_TEXT, "📍"),
    }
    bg, border, text, icon = config.get(
        tier, (CREAM, LIGHT_GRAY, DARK, "ℹ️")
    )
    meta_html = (
        f'<div style="font-size:0.75rem;color:{MEDIUM_GRAY};margin-top:6px">{meta}</div>'
        if meta
        else ""
    )
    return f"""
<div style="
    background:{bg};
    border-left:4px solid {border};
    border-radius:6px;
    padding:12px 16px;
    margin-bottom:10px;
">
  <div style="font-weight:700;color:{text};font-size:0.9rem">
    {icon}&nbsp;{title}
  </div>
  <div style="color:{DARK};font-size:0.85rem;margin-top:4px">{body}</div>
  {meta_html}
</div>"""


def kpi_card(label: str, value: str, sub: str = "", color: str = PRIMARY) -> str:
    """Inline KPI card for use in custom layouts."""
    return f"""
<div style="
    background:{WHITE};
    border:1px solid {LIGHT_GRAY};
    border-top:3px solid {color};
    border-radius:10px;
    padding:16px 20px;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
">
  <div style="font-size:0.72rem;font-weight:600;color:{MEDIUM_GRAY};
              text-transform:uppercase;letter-spacing:0.05em">{label}</div>
  <div style="font-size:2rem;font-weight:700;color:{DARK};margin-top:4px">{value}</div>
  {"" if not sub else f'<div style="font-size:0.78rem;color:{MEDIUM_GRAY};margin-top:2px">{sub}</div>'}
</div>"""


def section_header(title: str, subtitle: str = "") -> str:
    sub_html = (
        f'<p style="color:{MEDIUM_GRAY};font-size:0.875rem;margin-top:2px;'
        f'margin-bottom:0">{subtitle}</p>'
        if subtitle
        else ""
    )
    return f"""
<div style="margin-bottom:1.25rem">
  <h3 style="margin-bottom:0;color:{DARK}">{title}</h3>
  {sub_html}
</div>"""
