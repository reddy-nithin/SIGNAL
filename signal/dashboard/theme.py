"""
SIGNAL Dashboard — Shared Theme Constants
==========================================
Stage colors, method colors, Plotly layout, CSS injection, and
HTML component helpers for a production-grade dark dashboard.
"""
from __future__ import annotations

# ── Stage Colors (6-stage narrative arc) ─────────────────────────────────────
STAGE_COLORS: dict[str, str] = {
    "Curiosity": "#4ECDC4",       # teal
    "Experimentation": "#45B7D1", # cyan
    "Regular Use": "#FFA07A",     # salmon
    "Dependence": "#FF6B6B",      # coral-red
    "Crisis": "#E63946",          # bright red
    "Recovery": "#98D8C8",        # mint
}

# Ordered list for consistent chart rendering
STAGE_ORDER: list[str] = [
    "Curiosity", "Experimentation", "Regular Use",
    "Dependence", "Crisis", "Recovery",
]

# ── Method Colors ────────────────────────────────────────────────────────────
METHOD_COLORS: dict[str, str] = {
    "rule_based": "#7EB77F",   # sage green
    "embedding": "#B07CC6",    # lavender
    "fine_tuned": "#E8A838",   # amber
    "llm": "#5DA5DA",          # steel blue
    "ensemble": "#FAFAFA",     # white
}

# ── Plotly Dark Layout ────────────────────────────────────────────────────────
PLOTLY_LAYOUT: dict = {
    "paper_bgcolor": "#0E1117",
    "plot_bgcolor": "#0E1117",
    "font": {"color": "#FAFAFA", "family": "sans-serif"},
    "xaxis": {"gridcolor": "#1e2130", "zerolinecolor": "#2a2f3e"},
    "yaxis": {"gridcolor": "#1e2130", "zerolinecolor": "#2a2f3e"},
    "colorway": list(STAGE_COLORS.values()),
    "margin": {"l": 40, "r": 20, "t": 52, "b": 44},
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "bordercolor": "rgba(255,255,255,0.08)",
        "borderwidth": 1,
    },
    "hoverlabel": {
        "bgcolor": "#1e2130",
        "bordercolor": "#333344",
        "font": {"color": "#FAFAFA"},
    },
}


# ── Global CSS ───────────────────────────────────────────────────────────────

def get_css() -> str:
    """Return the full global CSS string for all dashboard pages."""
    return """
<style>

/* ── Hide Streamlit default top padding ───────── */
.block-container { padding-top: 1.5rem !important; }

/* ── Typography ───────────────────────────────── */
[data-testid="stMetricValue"] {
  font-weight: 800 !important;
  font-size: 1.6rem !important;
  line-height: 1.2 !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.76rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  opacity: 0.6 !important;
  font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b0c14 0%, #0E1117 40%, #0b0c14 100%);
  border-right: 1px solid rgba(255,107,107,0.15);
}

/* ── Primary button ───────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #FF6B6B 0%, #D63031 100%) !important;
  border: none !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  box-shadow: 0 4px 14px rgba(255,107,107,0.35) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(255,107,107,0.55) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Tab bar ──────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 4px !important; }
.stTabs [data-baseweb="tab"] {
  border-radius: 6px 6px 0 0 !important;
  padding: 9px 22px !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
  background: rgba(255,255,255,0.025) !important;
  transition: background 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(255,107,107,0.12) !important;
  border-bottom: 2px solid #FF6B6B !important;
  color: #FAFAFA !important;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255,107,107,0.07) !important;
}

/* ── Expander ─────────────────────────────────── */
.streamlit-expanderHeader {
  font-weight: 600 !important;
  background: rgba(255,255,255,0.02) !important;
  border-radius: 6px !important;
}

/* ── DataFrame header ─────────────────────────── */
.stDataFrame thead tr th {
  background-color: rgba(255,107,107,0.1) !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  font-size: 0.74rem !important;
}

/* ── Info / Warning / Error boxes ─────────────── */
.stAlert {
  border-radius: 8px !important;
  border-left-width: 3px !important;
}

/* ── Pulse animation for CRISIS banners ──────── */
@keyframes signal-pulse-glow {
  0%, 100% {
    box-shadow: 0 0 6px rgba(230,57,70,0.3),
                inset 0 0 40px rgba(230,57,70,0.02);
  }
  50% {
    box-shadow: 0 0 22px rgba(230,57,70,0.7),
                inset 0 0 40px rgba(230,57,70,0.07);
  }
}
.signal-crisis-pulse {
  animation: signal-pulse-glow 2.2s ease-in-out infinite;
}

/* ── Gradient divider ─────────────────────────── */
.signal-divider {
  height: 1px;
  background: linear-gradient(
    90deg, transparent 0%,
    rgba(255,107,107,0.45) 50%,
    transparent 100%
  );
  margin: 28px 0;
  border: none;
}

/* ── Section header strip ─────────────────────── */
.signal-section-header {
  border-left: 3px solid #FF6B6B;
  padding: 3px 0 3px 14px;
  margin: 28px 0 14px 0;
}
.signal-section-header .sh-text {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: #FAFAFA;
}
.signal-section-header .sh-sub {
  margin: 2px 0 0 0;
  font-size: 0.8rem;
  opacity: 0.5;
  font-weight: 400;
}

/* ── Hero banner ──────────────────────────────── */
.signal-hero {
  background: linear-gradient(135deg, #0d0e1a 0%, #16213e 45%, #0d1321 100%);
  padding: 52px 36px;
  border-radius: 16px;
  border: 1px solid rgba(255,107,107,0.2);
  text-align: center;
  margin-bottom: 28px;
}
.signal-hero-title {
  font-size: 3.2rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  background: linear-gradient(135deg, #FF6B6B 0%, #FFA07A 60%, #E8A838 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin: 0 0 12px 0;
}
.signal-hero-subtitle {
  font-size: 1.05rem;
  color: rgba(250,250,250,0.65);
  max-width: 560px;
  margin: 0 auto 10px auto;
  line-height: 1.5;
}
.signal-hero-context {
  font-size: 0.78rem;
  color: rgba(250,250,250,0.35);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin: 0;
}

/* ── Metric card grid ─────────────────────────── */
.signal-metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 20px 0;
}
.signal-metric-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 20px 16px;
  text-align: center;
  transition: border-color 0.2s, background 0.2s, transform 0.2s;
}
.signal-metric-card:hover {
  border-color: rgba(255,107,107,0.3);
  background: rgba(255,107,107,0.04);
  transform: translateY(-2px);
}
.signal-metric-card .val {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.1;
  margin-bottom: 6px;
}
.signal-metric-card .lbl {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  opacity: 0.5;
}

/* ── Pipeline chips ───────────────────────────── */
.signal-pipeline {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin: 20px 0;
  justify-content: center;
}
.signal-pipe-step {
  background: rgba(255,107,107,0.08);
  border: 1px solid rgba(255,107,107,0.22);
  border-radius: 22px;
  padding: 7px 16px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: #FFA07A;
  white-space: nowrap;
}
.signal-pipe-arrow {
  color: rgba(250,250,250,0.25);
  font-size: 1.1em;
}

/* ── Stage arc ────────────────────────────────── */
.signal-stage-arc {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
  justify-content: center;
  margin: 18px 0 24px 0;
}
.signal-arc-pill {
  border-radius: 20px;
  padding: 5px 13px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  border-width: 1px;
  border-style: solid;
}
.signal-arc-arrow {
  color: rgba(250,250,250,0.2);
  font-size: 0.85em;
}

/* ── Nav cards ────────────────────────────────── */
.signal-nav-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin: 20px 0;
}
.signal-nav-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 24px 20px;
  transition: border-color 0.2s, background 0.2s, transform 0.2s;
}
.signal-nav-card:hover {
  border-color: rgba(255,107,107,0.28);
  background: rgba(255,107,107,0.04);
  transform: translateY(-3px);
}
.signal-nav-card .card-num {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #FF6B6B;
  opacity: 0.7;
  margin-bottom: 8px;
}
.signal-nav-card .card-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 10px;
  color: #FAFAFA;
}
.signal-nav-card .card-body {
  font-size: 0.86rem;
  opacity: 0.6;
  line-height: 1.55;
}

/* ── Risk banner ──────────────────────────────── */
.signal-risk-banner {
  border-left-width: 4px;
  border-left-style: solid;
  border-radius: 8px;
  padding: 16px 20px;
  margin: 14px 0;
}
.signal-risk-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  opacity: 0.55;
  margin-bottom: 4px;
}
.signal-risk-stage {
  font-size: 1.2rem;
  font-weight: 800;
  margin-bottom: 6px;
  letter-spacing: 0.02em;
}
.signal-risk-msg {
  font-size: 0.9rem;
  opacity: 0.85;
  line-height: 1.55;
}

/* ── Text input styling ───────────────────────── */
.stTextArea textarea {
  border-radius: 8px !important;
  border-color: rgba(255,255,255,0.1) !important;
  background: rgba(255,255,255,0.03) !important;
  font-size: 0.92rem !important;
  line-height: 1.55 !important;
  transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: rgba(255,107,107,0.4) !important;
  box-shadow: 0 0 0 1px rgba(255,107,107,0.2) !important;
}

/* ── Select box styling ───────────────────────── */
.stSelectbox > div > div {
  border-radius: 8px !important;
  border-color: rgba(255,255,255,0.1) !important;
  background: rgba(255,255,255,0.03) !important;
  transition: border-color 0.2s !important;
}
.stSelectbox > div > div:hover {
  border-color: rgba(255,107,107,0.3) !important;
}

/* ── DistilBERT card ──────────────────────────── */
.signal-bert-card {
  background: rgba(232,168,56,0.05);
  border: 1px solid rgba(232,168,56,0.25);
  border-radius: 12px;
  padding: 20px;
  margin: 10px 0;
}
.signal-bert-header {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: #E8A838;
  opacity: 0.8;
  margin-bottom: 14px;
}

/* ── Community risk callout ───────────────────── */
.signal-risk-callout {
  border-radius: 12px;
  padding: 0;
  margin: 14px 0;
  overflow: hidden;
}
.signal-risk-callout-header {
  padding: 10px 20px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.signal-risk-callout-body {
  padding: 16px 20px;
}
.signal-risk-callout .community-name {
  font-size: 1.35rem;
  font-weight: 800;
  margin-bottom: 10px;
}
.signal-risk-callout .stat-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.signal-stat-pill {
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  border: 1px solid;
}
.signal-risk-callout .rec-text {
  font-size: 0.85rem;
  opacity: 0.7;
  line-height: 1.5;
}

</style>
"""


def inject_css() -> None:
    """Inject the global SIGNAL CSS into the current Streamlit page."""
    import streamlit as st
    st.markdown(get_css(), unsafe_allow_html=True)


# ── HTML Component Helpers ────────────────────────────────────────────────────

def stage_arc_html() -> str:
    """Render the 6-stage addiction narrative arc as colored pills with arrows."""
    parts: list[str] = []
    for i, stage in enumerate(STAGE_ORDER):
        color = STAGE_COLORS[stage]
        parts.append(
            f'<span class="signal-arc-pill" '
            f'style="background-color:{color}18; color:{color}; border-color:{color}33;">'
            f'{stage}</span>'
        )
        if i < len(STAGE_ORDER) - 1:
            parts.append('<span class="signal-arc-arrow">&#8594;</span>')

    inner = "\n".join(parts)
    return f'<div class="signal-stage-arc">{inner}</div>'


def pipeline_html(layers: list[str]) -> str:
    """Render a horizontal 4-layer pipeline flow with chip-style steps."""
    parts: list[str] = []
    for i, layer in enumerate(layers):
        parts.append(f'<span class="signal-pipe-step">{layer}</span>')
        if i < len(layers) - 1:
            parts.append('<span class="signal-pipe-arrow">&#8594;</span>')
    inner = "\n".join(parts)
    return f'<div class="signal-pipeline">{inner}</div>'


def metric_grid_html(
    metrics: list[tuple[str, str, str]],
    columns: int = 4,
) -> str:
    """
    Render a grid of styled metric cards.

    Args:
        metrics: list of (value, label, accent_color) tuples
        columns: grid column count (default 4)
    """
    cards: list[str] = []
    for value, label, color in metrics:
        cards.append(
            f'<div class="signal-metric-card">'
            f'<div class="val" style="color:{color};">{value}</div>'
            f'<div class="lbl">{label}</div>'
            f'</div>'
        )
    grid_style = f"grid-template-columns: repeat({columns}, 1fr);"
    return (
        f'<div class="signal-metric-grid" style="{grid_style}">'
        + "".join(cards)
        + "</div>"
    )


def nav_cards_html(cards: list[tuple[str, str, str]]) -> str:
    """
    Render 3-column page navigation cards.

    Args:
        cards: list of (number_label, title, description) tuples
    """
    html_cards: list[str] = []
    for num, title, desc in cards:
        html_cards.append(
            f'<div class="signal-nav-card">'
            f'<div class="card-num">{num}</div>'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-body">{desc}</div>'
            f'</div>'
        )
    return (
        '<div class="signal-nav-cards">'
        + "".join(html_cards)
        + "</div>"
    )


def section_header_html(title: str, subtitle: str = "") -> str:
    """Render a left-bordered section header with optional subtitle."""
    sub = (
        f'<p class="sh-sub">{subtitle}</p>' if subtitle else ""
    )
    return (
        f'<div class="signal-section-header">'
        f'<p class="sh-text">{title}</p>'
        f'{sub}'
        f'</div>'
    )


def gradient_divider_html() -> str:
    """Return an HR tag with the gradient divider class."""
    return '<hr class="signal-divider">'


def risk_banner_html(stage: str, message: str) -> str:
    """
    Render a styled risk level banner.
    Crisis banners include the pulsing glow animation.
    """
    color = STAGE_COLORS.get(stage, "#888888")
    pulse_class = " signal-crisis-pulse" if stage == "Crisis" else ""
    return (
        f'<div class="signal-risk-banner{pulse_class}" '
        f'style="border-left-color:{color}; background-color:{color}14;">'
        f'<div class="signal-risk-label">Signal Risk Level</div>'
        f'<div class="signal-risk-stage" style="color:{color};">{stage}</div>'
        f'<div class="signal-risk-msg">{message}</div>'
        f'</div>'
    )


def distilbert_card_html(metrics: list[tuple[str, str, str]]) -> str:
    """Render a styled DistilBERT performance card with amber accent."""
    cards: list[str] = []
    for value, label, delta in metrics:
        delta_html = (
            f'<div style="font-size:0.72rem;color:#888;margin-top:3px;">{delta}</div>'
            if delta else ""
        )
        cards.append(
            f'<div style="text-align:center; padding: 12px 8px;">'
            f'<div style="font-size:1.6rem;font-weight:800;color:#E8A838;line-height:1.1;">{value}</div>'
            f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;opacity:0.55;margin-top:5px;">{label}</div>'
            f'{delta_html}'
            f'</div>'
        )

    grid = (
        f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);gap:8px;">'
        + "".join(cards)
        + "</div>"
    )
    return (
        '<div class="signal-bert-card">'
        '<div class="signal-bert-header">Fine-Tuned DistilBERT &mdash; 5-Fold Cross-Validation</div>'
        + grid
        + "</div>"
    )


def community_risk_callout_html(
    label: str,
    tier: str,
    crisis_pct: float,
    dep_pct: float,
    recommendation: str,
) -> str:
    """Render a prominent community risk callout card."""
    tier_colors = {
        "CRITICAL": "#E63946",
        "HIGH": "#FFA07A",
        "MODERATE": "#E8A838",
        "LOW": "#4ECDC4",
    }
    color = tier_colors.get(tier, "#888888")
    combined = crisis_pct + dep_pct

    crisis_pill = (
        f'<span class="signal-stat-pill" '
        f'style="background:{color}15;color:{color};border-color:{color}33;">'
        f'Crisis {crisis_pct:.0%}</span>'
    )
    dep_pill = (
        f'<span class="signal-stat-pill" '
        f'style="background:{color}15;color:{color};border-color:{color}33;">'
        f'Dependence {dep_pct:.0%}</span>'
    )
    combined_pill = (
        f'<span class="signal-stat-pill" '
        f'style="background:{color}22;color:{color};border-color:{color}44;">'
        f'Combined {combined:.0%}</span>'
    )

    return (
        f'<div class="signal-risk-callout" style="border:1px solid {color}30;">'
        f'<div class="signal-risk-callout-header" '
        f'style="background:{color}20; color:{color};">'
        f'Highest-Risk Community &mdash; {tier}'
        f'</div>'
        f'<div class="signal-risk-callout-body">'
        f'<div class="community-name" style="color:#FAFAFA;">{label}</div>'
        f'<div class="stat-pills">{crisis_pill}{dep_pill}{combined_pill}</div>'
        f'<div class="rec-text">{recommendation}</div>'
        f'</div>'
        f'</div>'
    )


# ── Legacy badge helpers (unchanged) ─────────────────────────────────────────

def agreement_badge(count: int, total: int) -> str:
    """Return an HTML badge string colored by agreement level."""
    if total == 0:
        return '<span style="color: #888;">N/A</span>'
    ratio = count / total
    if ratio >= 0.9:
        color = "#4ECDC4"
        label = "Strong"
    elif ratio >= 0.6:
        color = "#FFA07A"
        label = "Moderate"
    else:
        color = "#E63946"
        label = "Low"
    return (
        f'<span style="background-color: {color}20; color: {color}; '
        f'padding: 2px 8px; border-radius: 4px; font-weight: 600;">'
        f'{label} ({count}/{total})</span>'
    )


def stage_badge(stage: str) -> str:
    """Return an HTML badge with stage color."""
    color = STAGE_COLORS.get(stage, "#888888")
    return (
        f'<span style="background-color: {color}20; color: {color}; '
        f'padding: 2px 8px; border-radius: 4px; font-weight: 600;">'
        f'{stage}</span>'
    )


def risk_badge(tier: str) -> str:
    """Return an HTML badge for a community risk tier (CRITICAL/HIGH/MODERATE/LOW)."""
    colors = {
        "CRITICAL": "#E63946",
        "HIGH": "#FFA07A",
        "MODERATE": "#E8A838",
        "LOW": "#4ECDC4",
    }
    color = colors.get(tier, "#888888")
    return (
        f'<span style="background-color: {color}20; color: {color}; '
        f'padding: 2px 8px; border-radius: 4px; font-weight: 600;">'
        f'{tier}</span>'
    )


def community_risk_tier(crisis_pct: float, dependence_pct: float) -> str:
    """Classify a community's risk tier from Crisis + Dependence proportion."""
    combined = crisis_pct + dependence_pct
    if combined >= 0.50:
        return "CRITICAL"
    elif combined >= 0.30:
        return "HIGH"
    elif combined >= 0.15:
        return "MODERATE"
    else:
        return "LOW"
