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

/* ── Substance badge ─────────────────────────── */
.signal-substance-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 16px;
  padding: 6px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  margin: 4px;
  transition: transform 0.15s, box-shadow 0.15s;
}
.signal-substance-badge:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.signal-substance-badge .conf-bar {
  height: 4px;
  border-radius: 2px;
  min-width: 30px;
  max-width: 50px;
  opacity: 0.7;
}

/* ── Evidence card ───────────────────────────── */
.signal-evidence-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 14px 16px;
  margin: 8px 0;
  transition: transform 0.15s, border-color 0.2s;
}
.signal-evidence-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255,107,107,0.2);
}
.signal-evidence-card .ev-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.signal-evidence-card .ev-name {
  font-size: 0.82rem;
  font-weight: 700;
  color: #FAFAFA;
}
.signal-evidence-card .ev-type-badge {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-radius: 10px;
  padding: 2px 8px;
}
.signal-evidence-card .ev-relevance {
  height: 3px;
  border-radius: 2px;
  background: rgba(255,255,255,0.06);
  margin: 8px 0;
  overflow: hidden;
}
.signal-evidence-card .ev-relevance-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
}
.signal-evidence-card .ev-snippet {
  font-size: 0.76rem;
  opacity: 0.55;
  line-height: 1.5;
  font-family: 'SF Mono', 'Fira Code', monospace;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

/* ── Brief section (visible, no expander) ────── */
.signal-brief-section {
  border-radius: 10px;
  padding: 16px 20px;
  margin: 10px 0;
  border-left: 3px solid;
}
.signal-brief-section .brief-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.signal-brief-section .brief-body {
  font-size: 0.86rem;
  line-height: 1.65;
  opacity: 0.85;
}

/* ── Intervention card (compact) ─────────────── */
.signal-intervention-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 14px 16px;
  transition: border-color 0.2s, transform 0.15s;
}
.signal-intervention-card:hover {
  border-color: rgba(255,107,107,0.2);
  transform: translateY(-1px);
}
.signal-intervention-card .iv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.signal-intervention-card .iv-community {
  font-size: 0.92rem;
  font-weight: 700;
}
.signal-intervention-card .iv-tier {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  border-radius: 10px;
  padding: 2px 10px;
}
.signal-intervention-card .iv-stats {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 0.76rem;
  opacity: 0.6;
}
.signal-intervention-card .iv-rec {
  font-size: 0.8rem;
  opacity: 0.75;
  line-height: 1.5;
}

/* ── Fade-in animation ───────────────────────── */
@keyframes signal-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.signal-fade-in {
  animation: signal-fade-in 0.4s ease both;
}

/* ── Confidence matrix ───────────────────────── */
.signal-conf-matrix {
  border-collapse: separate;
  border-spacing: 3px;
  width: 100%;
  margin: 12px 0;
}
.signal-conf-matrix th {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 6px 8px;
  opacity: 0.6;
}
.signal-conf-matrix td {
  text-align: center;
  padding: 8px 6px;
  border-radius: 6px;
  font-size: 0.78rem;
  font-weight: 600;
  transition: transform 0.15s;
}
.signal-conf-matrix td:hover {
  transform: scale(1.05);
}

/* ── Responsive ──────────────────────────────── */
@media (max-width: 1024px) {
  .signal-metric-grid { grid-template-columns: repeat(2, 1fr); }
  .signal-nav-cards { grid-template-columns: 1fr; }
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


def highlighted_text_html(text: str, matches: list) -> str:
    """Render original text with colored inline highlights over detected substances.

    Args:
        text: The original post text.
        matches: List of substance match objects with char_start, char_end,
                 clinical_name, and drug_class attributes.
    """
    import html as _html

    DRUG_CLASS_COLORS: dict[str, str] = {
        "opioid": "#FF6B6B",
        "benzodiazepine": "#E8A838",
        "stimulant": "#45B7D1",
        "alcohol": "#B07CC6",
        "cannabis": "#7EB77F",
        "other": "#FFA07A",
    }

    # Deduplicate and sort spans descending so we can insert right-to-left
    spans: list[tuple[int, int, str, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for m in matches:
        start = getattr(m, "char_start", None)
        end = getattr(m, "char_end", None)
        if start is None or end is None or start < 0 or end <= start:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        clinical = getattr(m, "clinical_name", "unknown")
        drug_class = getattr(m, "drug_class", "other").lower()
        color = DRUG_CLASS_COLORS.get(drug_class, "#FFA07A")
        spans.append((start, end, clinical, drug_class, color))

    if not spans:
        safe = _html.escape(text)
        return (
            f'<div style="border:1px solid rgba(255,255,255,0.08); '
            f'border-left:3px solid rgba(255,255,255,0.25); '
            f'background:rgba(255,255,255,0.02); border-radius:8px; '
            f'padding:14px 18px; font-size:0.9rem; line-height:1.7; '
            f'margin:8px 0 14px 0;">{safe}</div>'
        )

    # Sort descending by start so replacements don't shift offsets
    spans.sort(key=lambda s: s[0], reverse=True)
    result = _html.escape(text)

    # We need to work on the *original* text offsets, then escape.
    # Rebuild from raw text with spans injected.
    parts: list[str] = []
    prev_end = len(text)
    spans.sort(key=lambda s: s[0], reverse=True)

    for start, end, clinical, drug_class, color in spans:
        # Text AFTER this span (up to previous span start)
        after = _html.escape(text[end:prev_end])
        matched = _html.escape(text[start:end])
        tooltip = f"{clinical} ({drug_class})"
        span_html = (
            f'<span style="background:{color}25; color:{color}; '
            f'border-bottom:2px solid {color}; padding:1px 4px; '
            f'border-radius:3px; font-weight:600; cursor:help;" '
            f'title="{_html.escape(tooltip)}">{matched}</span>'
        )
        parts.append(after)
        parts.append(span_html)
        prev_end = start

    # Text before first span
    parts.append(_html.escape(text[:prev_end]))
    parts.reverse()
    body = "".join(parts)

    # Legend pills
    used_classes = sorted(set(s[3] for s in spans))
    legend_pills = " ".join(
        f'<span style="display:inline-block; background:{DRUG_CLASS_COLORS.get(c, "#FFA07A")}20; '
        f'color:{DRUG_CLASS_COLORS.get(c, "#FFA07A")}; border:1px solid {DRUG_CLASS_COLORS.get(c, "#FFA07A")}40; '
        f'border-radius:12px; padding:2px 10px; font-size:0.72rem; '
        f'font-weight:600; letter-spacing:0.03em;">{c.title()}</span>'
        for c in used_classes
    )

    return (
        f'<div style="border:1px solid rgba(255,255,255,0.1); '
        f'border-radius:10px; overflow:hidden; margin:8px 0 14px 0;">'
        f'<div style="padding:14px 18px; font-size:0.9rem; line-height:1.8;">'
        f'{body}</div>'
        f'<div style="border-top:1px solid rgba(255,255,255,0.06); '
        f'padding:8px 18px; display:flex; gap:8px; align-items:center;">'
        f'<span style="font-size:0.68rem; opacity:0.4; font-weight:600; '
        f'letter-spacing:0.05em; text-transform:uppercase;">Detected:</span>'
        f'{legend_pills}</div>'
        f'</div>'
    )


def narrative_arc_indicator_html(top_stage: str, confidences: dict[str, float]) -> str:
    """Render a visual 6-stage arc with the detected stage highlighted.

    Args:
        top_stage: The detected narrative stage name.
        confidences: Dict mapping stage name → confidence (0-1).
    """
    nodes: list[str] = []
    for i, stage in enumerate(STAGE_ORDER):
        color = STAGE_COLORS[stage]
        conf = confidences.get(stage, 0)
        is_active = stage == top_stage

        if is_active:
            node = (
                f'<div style="display:flex; flex-direction:column; align-items:center; flex:1;">'
                f'<div style="width:48px; height:48px; border-radius:50%; '
                f'background:{color}; box-shadow:0 0 20px {color}80, 0 0 40px {color}40; '
                f'display:flex; align-items:center; justify-content:center; '
                f'transition:all 0.3s;"></div>'
                f'<div style="margin-top:8px; font-size:0.78rem; font-weight:800; '
                f'color:{color}; letter-spacing:0.02em;">{stage}</div>'
                f'<div style="font-size:0.72rem; font-weight:700; color:{color}; '
                f'opacity:0.9;">{conf:.0%}</div>'
                f'</div>'
            )
        else:
            node = (
                f'<div style="display:flex; flex-direction:column; align-items:center; flex:1;">'
                f'<div style="width:20px; height:20px; border-radius:50%; '
                f'background:{color}40; border:2px solid {color}30; '
                f'transition:all 0.3s;"></div>'
                f'<div style="margin-top:8px; font-size:0.68rem; font-weight:600; '
                f'color:{color}; opacity:0.5;">{stage}</div>'
                f'<div style="font-size:0.62rem; opacity:0.35;">{conf:.0%}</div>'
                f'</div>'
            )

        nodes.append(node)

        # Connector line between nodes
        if i < len(STAGE_ORDER) - 1:
            next_stage = STAGE_ORDER[i + 1]
            # Highlight connector if between active and adjacent
            line_active = (stage == top_stage or next_stage == top_stage)
            line_opacity = "0.4" if line_active else "0.12"
            line_color = color if line_active else "#FAFAFA"
            nodes.append(
                f'<div style="flex:0.5; height:2px; background:{line_color}; '
                f'opacity:{line_opacity}; align-self:center; margin-top:-16px;"></div>'
            )

    return (
        f'<div style="display:flex; align-items:flex-start; justify-content:center; '
        f'padding:20px 10px; margin:10px 0;">'
        + "".join(nodes)
        + '</div>'
    )


def architecture_diagram_html() -> str:
    """Render a visual 4-layer architecture diagram with icons and descriptions."""
    layers = [
        ("1", "Substance Resolution", "#4ECDC4",
         "Slang → Clinical Entity",
         "Rule-based lexicon · SBERT embeddings · Gemini zero-shot"),
        ("2", "Narrative Stage", "#45B7D1",
         "Post → Addiction Arc Stage",
         "Keyword rules · Fine-tuned DistilBERT · Gemini few-shot"),
        ("3", "Clinical Grounding", "#FFA07A",
         "Evidence Retrieval + Safety Signals",
         "FAISS/BM25 hybrid · 81 knowledge chunks · 265 FAERS signals"),
        ("4", "Analyst Brief", "#FF6B6B",
         "Evidence-Cited Risk Assessment",
         "Gemini synthesis · Citation linking · Intervention mapping"),
    ]

    cards: list[str] = []
    for i, (num, name, color, tagline, tech) in enumerate(layers):
        cards.append(
            f'<div style="flex:1; background:rgba(255,255,255,0.025); '
            f'border:1px solid {color}30; border-radius:12px; '
            f'padding:18px 14px; text-align:center; position:relative; '
            f'transition:border-color 0.2s, transform 0.2s;">'
            f'<div style="font-size:0.62rem; font-weight:800; letter-spacing:0.12em; '
            f'color:{color}; opacity:0.7; text-transform:uppercase; margin-bottom:6px;">'
            f'Layer {num}</div>'
            f'<div style="font-size:0.95rem; font-weight:700; color:#FAFAFA; '
            f'margin-bottom:4px;">{name}</div>'
            f'<div style="font-size:0.78rem; color:{color}; font-weight:600; '
            f'margin-bottom:8px;">{tagline}</div>'
            f'<div style="font-size:0.68rem; opacity:0.4; line-height:1.4;">{tech}</div>'
            f'</div>'
        )
        if i < len(layers) - 1:
            cards.append(
                f'<div style="display:flex; align-items:center; padding:0 2px; '
                f'color:rgba(255,255,255,0.2); font-size:1.2rem;">&#9654;</div>'
            )

    return (
        f'<div style="display:flex; align-items:stretch; gap:6px; margin:16px 0; '
        f'overflow-x:auto;">'
        + "".join(cards)
        + '</div>'
    )


def comparison_grid_html() -> str:
    """Render a SIGNAL vs Typical Approaches visual comparison grid."""
    rows = [
        ("Narrative Stages", "6-stage addiction arc (novel)", "Binary risk classification", True),
        ("Detection Methods", "3 methods × 2 tasks = 6 evaluations", "1–2 methods", True),
        ("Model Training", "Fine-tuned DistilBERT (5-fold CV)", "API calls only", True),
        ("Clinical Grounding", "81 KB chunks + 265 FAERS signals", "None or generic", True),
        ("Evaluation", "Inter-method agreement + per-class F1", "Basic accuracy", True),
        ("Slang Resolution", "200+ slang → clinical mappings", "Dictionary lookup", True),
    ]

    header = (
        '<div style="display:grid; grid-template-columns:1.2fr 2fr 2fr; gap:0; '
        'border-radius:10px; overflow:hidden; border:1px solid rgba(255,255,255,0.08); '
        'margin:14px 0;">'
        '<div style="padding:10px 14px; background:rgba(255,255,255,0.05); '
        'font-size:0.68rem; font-weight:700; letter-spacing:0.08em; '
        'text-transform:uppercase; opacity:0.5;"></div>'
        '<div style="padding:10px 14px; background:rgba(255,107,107,0.1); '
        'font-size:0.72rem; font-weight:800; letter-spacing:0.06em; '
        'text-transform:uppercase; color:#FF6B6B; text-align:center;">SIGNAL</div>'
        '<div style="padding:10px 14px; background:rgba(255,255,255,0.03); '
        'font-size:0.72rem; font-weight:700; letter-spacing:0.06em; '
        'text-transform:uppercase; opacity:0.4; text-align:center;">Typical</div>'
    )

    row_html: list[str] = []
    for feature, signal_val, typical_val, signal_wins in rows:
        check = "&#10003;" if signal_wins else ""
        row_html.append(
            f'<div style="padding:10px 14px; border-top:1px solid rgba(255,255,255,0.05); '
            f'font-size:0.8rem; font-weight:600; color:#FAFAFA;">{feature}</div>'
            f'<div style="padding:10px 14px; border-top:1px solid rgba(255,255,255,0.05); '
            f'font-size:0.8rem; color:#4ECDC4; text-align:center;">'
            f'<span style="color:#4ECDC4; margin-right:6px;">{check}</span>{signal_val}</div>'
            f'<div style="padding:10px 14px; border-top:1px solid rgba(255,255,255,0.05); '
            f'font-size:0.8rem; opacity:0.4; text-align:center;">{typical_val}</div>'
        )

    return header + "".join(row_html) + '</div>'


def evidence_meter_html(n_chunks: int, n_signals: int, avg_relevance: float) -> str:
    """Render a visual evidence strength meter for clinical grounding."""
    rel_pct = min(avg_relevance * 100, 100)
    return (
        f'<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:10px 0;">'
        # Knowledge chunks
        f'<div style="background:rgba(78,205,196,0.06); border:1px solid rgba(78,205,196,0.2); '
        f'border-radius:8px; padding:12px; text-align:center;">'
        f'<div style="font-size:1.4rem; font-weight:800; color:#4ECDC4;">{n_chunks}</div>'
        f'<div style="font-size:0.65rem; font-weight:600; letter-spacing:0.06em; '
        f'text-transform:uppercase; opacity:0.5; margin-top:2px;">Evidence Chunks</div></div>'
        # FAERS signals
        f'<div style="background:rgba(255,160,122,0.06); border:1px solid rgba(255,160,122,0.2); '
        f'border-radius:8px; padding:12px; text-align:center;">'
        f'<div style="font-size:1.4rem; font-weight:800; color:#FFA07A;">{n_signals}</div>'
        f'<div style="font-size:0.65rem; font-weight:600; letter-spacing:0.06em; '
        f'text-transform:uppercase; opacity:0.5; margin-top:2px;">FAERS Signals</div></div>'
        # Relevance bar
        f'<div style="background:rgba(232,168,56,0.06); border:1px solid rgba(232,168,56,0.2); '
        f'border-radius:8px; padding:12px; text-align:center;">'
        f'<div style="font-size:1.4rem; font-weight:800; color:#E8A838;">{avg_relevance:.0%}</div>'
        f'<div style="font-size:0.65rem; font-weight:600; letter-spacing:0.06em; '
        f'text-transform:uppercase; opacity:0.5; margin-top:2px;">Avg Relevance</div>'
        f'<div style="height:4px; background:rgba(232,168,56,0.15); border-radius:2px; '
        f'margin-top:6px; overflow:hidden;">'
        f'<div style="height:100%; width:{rel_pct:.0f}%; background:#E8A838; '
        f'border-radius:2px;"></div></div></div>'
        f'</div>'
    )


def training_story_html() -> str:
    """Render a visual training pipeline for the DistilBERT model."""
    steps = [
        ("01", "Data Collection", "600 posts across 6 stages", "#4ECDC4"),
        ("02", "Augmentation", "Gemini-generated variations", "#45B7D1"),
        ("03", "Fine-Tuning", "DistilBERT + class-weighted loss", "#E8A838"),
        ("04", "Validation", "5-fold cross-validation", "#FFA07A"),
        ("05", "Deployment", "Real-time stage classification", "#98D8C8"),
    ]

    items: list[str] = []
    for num, title, desc, color in steps:
        items.append(
            f'<div style="display:flex; align-items:center; gap:12px; '
            f'padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<div style="min-width:32px; height:32px; border-radius:50%; '
            f'background:{color}20; border:2px solid {color}50; display:flex; '
            f'align-items:center; justify-content:center; font-size:0.68rem; '
            f'font-weight:800; color:{color};">{num}</div>'
            f'<div>'
            f'<div style="font-size:0.82rem; font-weight:700; color:#FAFAFA;">{title}</div>'
            f'<div style="font-size:0.7rem; opacity:0.5;">{desc}</div>'
            f'</div></div>'
        )

    return (
        '<div style="background:rgba(232,168,56,0.04); border:1px solid rgba(232,168,56,0.15); '
        'border-radius:10px; padding:14px 16px; margin:10px 0;">'
        '<div style="font-size:0.68rem; font-weight:700; letter-spacing:0.08em; '
        'text-transform:uppercase; color:#E8A838; opacity:0.7; margin-bottom:10px;">'
        'Model Training Pipeline</div>'
        + "".join(items)
        + '</div>'
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


# ── v2 HTML Helpers ──────────────────────────────────────────────────────────

DRUG_CLASS_COLORS: dict[str, str] = {
    "opioid": "#FF6B6B",
    "benzodiazepine": "#E8A838",
    "stimulant": "#45B7D1",
    "alcohol": "#B07CC6",
    "cannabis": "#7EB77F",
    "other": "#FFA07A",
}


def substance_badge_html(
    slang: str, clinical: str, drug_class: str, confidence: float,
) -> str:
    """Render a colored pill badge for a detected substance."""
    color = DRUG_CLASS_COLORS.get(drug_class.lower(), "#FFA07A")
    conf_w = max(10, int(confidence * 50))
    return (
        f'<div class="signal-substance-badge" '
        f'style="background:{color}12; border:1px solid {color}30; color:{color};">'
        f'<span style="font-weight:700;">{slang}</span>'
        f'<span style="opacity:0.5; font-size:0.68rem;">&rarr;</span>'
        f'<span style="opacity:0.8; font-size:0.72rem;">{clinical}</span>'
        f'<div class="conf-bar" style="background:{color}; width:{conf_w}px;"></div>'
        f'<span style="font-size:0.65rem; opacity:0.5;">{confidence:.0%}</span>'
        f'</div>'
    )


def evidence_card_html(
    chunk_name: str, chunk_type: str, relevance: float,
    snippet: str, color: str = "#4ECDC4",
) -> str:
    """Render a visual evidence card replacing dataframe row."""
    type_colors = {
        "ingredient": "#4ECDC4",
        "faers_signals": "#FF6B6B",
        "safety": "#E8A838",
        "classification": "#45B7D1",
    }
    tc = type_colors.get(chunk_type.lower(), "#FFA07A")
    pct = max(0, min(100, int(relevance * 100)))
    # Escape HTML in snippet
    safe_snippet = snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="signal-evidence-card" style="border-left:3px solid {color};">'
        f'<div class="ev-header">'
        f'<span class="ev-name">{chunk_name}</span>'
        f'<span class="ev-type-badge" style="background:{tc}18; color:{tc}; '
        f'border:1px solid {tc}30;">{chunk_type}</span>'
        f'</div>'
        f'<div class="ev-relevance">'
        f'<div class="ev-relevance-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<div style="font-size:0.65rem; opacity:0.4; margin-bottom:4px;">'
        f'Relevance: {relevance:.1%}</div>'
        f'<div class="ev-snippet">{safe_snippet}</div>'
        f'</div>'
    )


def brief_summary_card_html(
    substances: list[str], stage: str, risk_level: str, recommendation: str,
) -> str:
    """Render visual summary card for the analyst brief overview."""
    stage_color = STAGE_COLORS.get(stage, "#FFA07A")
    risk_colors = {
        "Crisis": "#E63946", "Dependence": "#FF6B6B",
        "Regular Use": "#FFA07A", "Experimentation": "#45B7D1",
        "Curiosity": "#4ECDC4", "Recovery": "#98D8C8",
    }
    rc = risk_colors.get(risk_level, stage_color)
    sub_pills = " ".join(
        f'<span style="background:rgba(255,255,255,0.06); border-radius:12px; '
        f'padding:3px 10px; font-size:0.72rem; font-weight:600;">{s}</span>'
        for s in substances[:4]
    )
    # Truncate recommendation
    rec_short = recommendation[:180] + "..." if len(recommendation) > 180 else recommendation
    return (
        f'<div style="background:linear-gradient(135deg, {rc}08, {rc}03); '
        f'border:1px solid {rc}25; border-radius:12px; padding:20px; margin:10px 0;">'
        f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; '
        f'margin-bottom:14px;">'
        # Substances
        f'<div>'
        f'<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em; '
        f'text-transform:uppercase; opacity:0.45; margin-bottom:6px;">Substances</div>'
        f'<div style="display:flex; flex-wrap:wrap; gap:4px;">{sub_pills}</div>'
        f'</div>'
        # Stage
        f'<div>'
        f'<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em; '
        f'text-transform:uppercase; opacity:0.45; margin-bottom:6px;">Narrative Stage</div>'
        f'<div style="font-size:1.1rem; font-weight:800; color:{stage_color};">{stage}</div>'
        f'</div>'
        # Risk
        f'<div>'
        f'<div style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em; '
        f'text-transform:uppercase; opacity:0.45; margin-bottom:6px;">Risk Level</div>'
        f'<div style="font-size:1.1rem; font-weight:800; color:{rc};">{risk_level}</div>'
        f'</div>'
        f'</div>'
        # Recommendation
        f'<div style="font-size:0.82rem; opacity:0.7; line-height:1.55; '
        f'border-top:1px solid rgba(255,255,255,0.06); padding-top:12px;">'
        f'{rec_short}</div>'
        f'</div>'
    )


def brief_section_html(
    header: str, body: str, color: str, icon: str = "",
) -> str:
    """Render a visible brief section (NOT in an expander)."""
    import re as _re
    # Convert markdown bold to HTML before newline replacement
    safe_body = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    # Convert markdown bullets to styled list items
    safe_body = _re.sub(
        r"(?m)^\s*\*\s+(.+)$",
        r'<div style="padding-left:12px; margin:2px 0;">• \1</div>',
        safe_body,
    )
    safe_body = safe_body.replace("\n", "<br>")
    return (
        f'<div class="signal-brief-section" '
        f'style="border-left-color:{color}; background:{color}06;">'
        f'<div class="brief-header" style="color:{color};">'
        f'{icon} {header}</div>'
        f'<div class="brief-body">{safe_body}</div>'
        f'</div>'
    )


def intervention_card_html(
    community: str, tier: str, crisis_pct: float, dep_pct: float,
    dominant_stage: str, recommendation: str,
) -> str:
    """Compact intervention card for a community."""
    tier_colors = {
        "CRITICAL": "#E63946", "HIGH": "#FFA07A",
        "MODERATE": "#E8A838", "LOW": "#4ECDC4",
    }
    tc = tier_colors.get(tier, "#888")
    combined = crisis_pct + dep_pct
    return (
        f'<div class="signal-intervention-card">'
        f'<div class="iv-header">'
        f'<span class="iv-community">{community}</span>'
        f'<span class="iv-tier" style="background:{tc}18; color:{tc}; '
        f'border:1px solid {tc}35;">{tier}</span>'
        f'</div>'
        f'<div class="iv-stats">'
        f'<span>Crisis {crisis_pct:.0%}</span>'
        f'<span style="opacity:0.3;">|</span>'
        f'<span>Dependence {dep_pct:.0%}</span>'
        f'<span style="opacity:0.3;">|</span>'
        f'<span style="color:{tc}; font-weight:600;">Combined {combined:.0%}</span>'
        f'</div>'
        f'<div style="font-size:0.72rem; font-weight:700; color:{tc}; '
        f'letter-spacing:0.04em; margin-bottom:4px;">'
        f'Dominant: {dominant_stage}</div>'
        f'<div class="iv-rec">{recommendation}</div>'
        f'</div>'
    )


def how_it_works_html() -> str:
    """Visual 4-layer pipeline example showing data flow."""
    steps = [
        (
            "Input",
            '"I overdosed on fentanyl last night. My roommate had to call 911. '
            'I was mixing lean with bars and I almost died."',
            "#FAFAFA", "rgba(255,255,255,0.04)",
        ),
        (
            "L1: Substance Resolution",
            '<span style="color:#FF6B6B;">fentanyl</span> (opioid) &middot; '
            '<span style="color:#FF6B6B;">lean</span> &rarr; codeine (opioid) &middot; '
            '<span style="color:#E8A838;">bars</span> &rarr; alprazolam (benzo)',
            "#4ECDC4", "rgba(78,205,196,0.04)",
        ),
        (
            "L2: Narrative Stage",
            '<span style="color:#E63946; font-weight:700;">Crisis</span> &mdash; '
            '79% confidence &middot; 3/3 method agreement',
            "#E63946", "rgba(230,57,70,0.04)",
        ),
        (
            "L3: Clinical Grounding",
            '5 knowledge chunks &middot; 19 FAERS adverse event signals &middot; '
            'FDA black box warning: benzo + opioid interaction',
            "#E8A838", "rgba(232,168,56,0.04)",
        ),
        (
            "L4: Analyst Brief",
            '"Acute poly-drug crisis involving fentanyl, codeine, and alprazolam. '
            'Immediate naloxone access and crisis intervention indicated..."',
            "#45B7D1", "rgba(69,183,209,0.04)",
        ),
    ]
    items: list[str] = []
    for i, (title, content, color, bg) in enumerate(steps):
        delay = f"animation-delay:{i * 0.08}s;"
        items.append(
            f'<div class="signal-fade-in" style="{delay}">'
            f'<div style="background:{bg}; border:1px solid {color}18; '
            f'border-left:3px solid {color}; border-radius:8px; padding:14px 18px;">'
            f'<div style="font-size:0.68rem; font-weight:700; letter-spacing:0.07em; '
            f'text-transform:uppercase; color:{color}; opacity:0.8; margin-bottom:6px;">'
            f'{title}</div>'
            f'<div style="font-size:0.84rem; opacity:0.8; line-height:1.5;">'
            f'{content}</div>'
            f'</div></div>'
        )
        if i < len(steps) - 1:
            items.append(
                f'<div style="text-align:center; padding:4px 0; opacity:0.2; '
                f'font-size:1rem;">&#9660;</div>'
            )
    return '<div style="margin:16px 0;">' + "".join(items) + '</div>'


def confidence_matrix_html(
    stages: list[str], methods: list[str],
    values: list[list[float]],
) -> str:
    """Render a 6-stage x 3-method confidence matrix as an HTML table."""
    # values[method_idx][stage_idx]
    header = '<tr><th></th>' + "".join(
        f'<th style="color:{STAGE_COLORS.get(s, "#888")};">{s}</th>'
        for s in stages
    ) + '</tr>'
    rows: list[str] = []
    for mi, method in enumerate(methods):
        mc = METHOD_COLORS.get(method.lower().replace(" ", "_"), "#888")
        cells = ""
        for si in range(len(stages)):
            v = values[mi][si] if mi < len(values) and si < len(values[mi]) else 0
            alpha = max(0.05, min(0.5, v * 0.6))
            sc = STAGE_COLORS.get(stages[si], "#888")
            txt_w = "700" if v > 0.3 else "400"
            cells += (
                f'<td style="background:{sc}{int(alpha*255):02x}; '
                f'font-weight:{txt_w};">{v:.0%}</td>'
            )
        mname = method.replace("_", " ").title()
        rows.append(
            f'<tr><td style="text-align:right; padding-right:10px; '
            f'font-size:0.72rem; font-weight:600; color:{mc}; opacity:0.8;">'
            f'{mname}</td>{cells}</tr>'
        )
    return (
        f'<table class="signal-conf-matrix">{header}{"".join(rows)}</table>'
    )
