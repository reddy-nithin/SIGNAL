"""
SIGNAL Dashboard — Main Entry Point
=====================================
Streamlit multi-page app. Run with:
    streamlit run signal/dashboard/signal_app.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parent.parent.parent
# Streamlit imports stdlib 'signal' before any page runs; pop it from the cache
# so PathFinder can find our local signal/ package instead.
if not getattr(_sys.modules.get("signal"), "__path__", None):
    _sys.modules.pop("signal", None)
    for _k in [k for k in _sys.modules if k.startswith("signal.")]:
        _sys.modules.pop(_k, None)
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import streamlit as st

from signal.dashboard.theme import (
    inject_css,
    stage_arc_html,
    pipeline_html,
    metric_grid_html,
    nav_cards_html,
    gradient_divider_html,
    section_header_html,
    architecture_diagram_html,
    comparison_grid_html,
    how_it_works_html,
    STAGE_COLORS,
)

st.set_page_config(
    page_title="SIGNAL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<div style="padding: 4px 0 12px 0;">'
    '<span style="font-size:1.4rem; font-weight:800; letter-spacing:0.1em; '
    'background:linear-gradient(135deg, #FF6B6B, #FFA07A); '
    '-webkit-background-clip:text; -webkit-text-fill-color:transparent; '
    'background-clip:text;">SIGNAL</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Substance Intelligence through Grounded\n"
    "Narrative Analysis of Language"
)
st.sidebar.divider()
st.sidebar.markdown(
    "**NSF NRT Challenge 1**  \n"
    "AI for Substance Abuse Risk Detection  \n"
    "from Social Signals"
)

# ── Hero Section ──────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="signal-hero">
  <div class="signal-hero-title">SIGNAL</div>
  <div class="signal-hero-subtitle">
    Substance Intelligence through Grounded Narrative Analysis of Language
  </div>
  <div class="signal-hero-context">
    NSF NRT Challenge 1 &nbsp;&middot;&nbsp; UMKC 2026 Spring Research-A-Thon
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Key Stats Grid ────────────────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.1s;">'
    + metric_grid_html([
        ("84",    "Knowledge Chunks",      "#4ECDC4"),
        ("310",   "Adverse Event Signals", "#45B7D1"),
        ("6",     "Narrative Stages",      "#FFA07A"),
        ("3 × 2", "Detection Methods",     "#FF6B6B"),
    ])
    + '</div>',
    unsafe_allow_html=True,
)

# ── How It Works ─────────────────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.2s;">'
    + section_header_html(
        "How It Works",
        "A real post flowing through all 4 layers of analysis",
    )
    + '</div>',
    unsafe_allow_html=True,
)
st.markdown(how_it_works_html(), unsafe_allow_html=True)

st.markdown(gradient_divider_html(), unsafe_allow_html=True)

# ── Addiction Narrative Arc ───────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.3s;">'
    + section_header_html(
        "Addiction Narrative Arc",
        "6 stages classified from unstructured social media text — a novel NLP task",
    )
    + '</div>',
    unsafe_allow_html=True,
)
st.markdown(stage_arc_html(), unsafe_allow_html=True)

st.markdown(gradient_divider_html(), unsafe_allow_html=True)

# ── Innovation Comparison ─────────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.4s;">'
    + section_header_html(
        "What Makes SIGNAL Different",
        "Novel contributions vs. standard approaches in substance abuse NLP",
    )
    + '</div>',
    unsafe_allow_html=True,
)
st.markdown(comparison_grid_html(), unsafe_allow_html=True)

st.markdown(gradient_divider_html(), unsafe_allow_html=True)

# ── Architecture Pipeline ─────────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.5s;">'
    + section_header_html(
        "4-Layer Detection Pipeline",
        "Each layer feeds into the next — substance → stage → grounding → brief",
    )
    + '</div>',
    unsafe_allow_html=True,
)
st.markdown(architecture_diagram_html(), unsafe_allow_html=True)

st.markdown(gradient_divider_html(), unsafe_allow_html=True)

# ── Page Navigation Cards ─────────────────────────────────────────────────────

st.markdown(
    '<div class="signal-fade-in" style="animation-delay:0.6s;">'
    + section_header_html("Explore the Dashboard")
    + nav_cards_html([
        (
            "Page 01",
            "Deep Analysis",
            "Paste any post for full 4-layer analysis with inline substance highlighting and evidence-cited briefs.",
        ),
        (
            "Page 02",
            "Narrative Pulse",
            "Cross-community stage distributions, risk profiles, and CDC mortality context.",
        ),
        (
            "Page 03",
            "Method Comparison",
            "3-method evaluation on both tasks: DistilBERT CV, agreement stats, and Sankey vote flow.",
        ),
    ])
    + '</div>',
    unsafe_allow_html=True,
)
