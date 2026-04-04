"""
SIGNAL Dashboard — Page 2: Narrative Pulse
============================================
Cross-community narrative stage distributions.
Shows how different online communities map to the 6-stage addiction arc.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_root = _Path(__file__).resolve().parent.parent.parent.parent
if not getattr(_sys.modules.get("signal"), "__path__", None):
    _sys.modules.pop("signal", None)
    for _k in [k for k in _sys.modules if k.startswith("signal.")]:
        _sys.modules.pop(_k, None)
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from signal.config import MORTALITY_PATH
from signal.dashboard.theme import (
    STAGE_COLORS, STAGE_ORDER, PLOTLY_LAYOUT,
    inject_css,
    community_risk_tier,
    section_header_html, gradient_divider_html,
    community_risk_callout_html,
    intervention_card_html,
)

inject_css()

st.markdown(
    section_header_html(
        "Narrative Pulse",
        "Cross-community stage distributions — where populations sit in the addiction arc",
    ),
    unsafe_allow_html=True,
)

# ── Stage intervention mapping ─────────────────────────────────────────────────

STAGE_INTERVENTIONS: dict[str, str] = {
    "Curiosity":       "Prevention messaging, peer education, myth-busting about substance risks.",
    "Experimentation": "Brief intervention, harm-reduction information, low-barrier counseling access.",
    "Regular Use":     "Screening and brief intervention (SBI), motivational interviewing resources.",
    "Dependence":      "MAT referral (buprenorphine, naltrexone), treatment access programs.",
    "Crisis":          "Crisis line integration, naloxone distribution, emergency harm reduction.",
    "Recovery":        "Peer support networks, MAT maintenance, relapse prevention programming.",
}


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_distributions() -> list[dict] | None:
    from signal.temporal.narrative_tracker import load_cached_distributions
    return load_cached_distributions()


@st.cache_data
def _load_mortality() -> dict | None:
    if MORTALITY_PATH.exists():
        try:
            return json.loads(MORTALITY_PATH.read_text())
        except Exception:
            return None
    return None


# ── Main page ──────────────────────────────────────────────────────────────────

distributions = _load_distributions()

if distributions is None:
    st.warning(
        "Stage distribution data not cached yet. "
        "Run the demo cache script first:\n\n"
        "```bash\npython -m signal.dashboard.demo_cache\n```"
    )

    if st.button("Compute now (may take a few minutes)"):
        with st.spinner("Computing stage distributions across communities…"):
            from signal.temporal.narrative_tracker import compute_and_cache
            distributions = compute_and_cache()
            st.rerun()
    else:
        st.stop()

# Sidebar controls
st.sidebar.markdown(
    '<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.07em;'
    'text-transform:uppercase;opacity:0.55;margin-bottom:8px;">Filters</div>',
    unsafe_allow_html=True,
)
min_group = st.sidebar.slider("Min group size", 50, 500, 100, step=50)
bar_mode = st.sidebar.radio("Chart mode", ["Stacked", "Grouped"], index=0)

# Filter by min group size
filtered = [d for d in distributions if d["group_size"] >= min_group]

if not filtered:
    st.info("No communities meet the minimum group size threshold. Try lowering the filter.")
    st.stop()

# Build DataFrame for plotting
rows = []
for d in filtered:
    for stage in STAGE_ORDER:
        rows.append({
            "Community": d["label"],
            "Stage": stage,
            "Proportion": d["stage_proportions"].get(stage, 0),
            "Count": d["stage_counts"].get(stage, 0),
        })
df = pd.DataFrame(rows)

# Main chart
fig = px.bar(
    df,
    x="Community",
    y="Proportion",
    color="Stage",
    barmode="stack" if bar_mode == "Stacked" else "group",
    color_discrete_map=STAGE_COLORS,
    category_orders={"Stage": STAGE_ORDER},
    text_auto=".0%" if bar_mode == "Grouped" else False,
)
fig.update_layout(
    title="Narrative Stage Distribution by Community",
    yaxis_title="Proportion of Posts",
    xaxis_title="",
    height=520,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
    ),
    margin={"l": 40, "r": 20, "t": 52, "b": 100},
    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin", "legend")},
)
fig.update_traces(marker_line_width=0)
fig.update_xaxes(tickangle=-40, tickfont=dict(size=11))
st.plotly_chart(fig, use_container_width=True)

# ── Radar chart: risk profile comparison (N1) ─────────────────────────────────

if len(filtered) >= 2:
    # Find highest-risk and lowest-risk communities for radar comparison
    ranked = sorted(
        filtered,
        key=lambda d: d["stage_proportions"].get("Crisis", 0) + d["stage_proportions"].get("Dependence", 0),
        reverse=True,
    )
    high_risk_comm = ranked[0]
    low_risk_comm = ranked[-1]

    radar_col1, radar_col2 = st.columns([2, 1])
    with radar_col1:
        fig_radar = go.Figure()

        for comm, dash_style, width in [
            (high_risk_comm, "solid", 2.5),
            (low_risk_comm, "dash", 1.5),
        ]:
            r_vals = [comm["stage_proportions"].get(s, 0) for s in STAGE_ORDER]
            r_vals.append(r_vals[0])  # close the polygon
            theta = list(STAGE_ORDER) + [STAGE_ORDER[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=theta,
                fill="toself",
                name=comm["label"],
                line=dict(dash=dash_style, width=width),
                opacity=0.7,
            ))

        fig_radar.update_layout(
            title="Community Risk Profile Comparison",
            polar=dict(
                bgcolor="#0E1117",
                radialaxis=dict(
                    visible=True, range=[0, max(0.5, max(
                        high_risk_comm["stage_proportions"].get(s, 0) for s in STAGE_ORDER
                    ) + 0.05)],
                    gridcolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=9, color="rgba(255,255,255,0.4)"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.08)",
                    tickfont=dict(size=10, color="#FAFAFA"),
                ),
            ),
            legend=dict(
                orientation="h", yanchor="top", y=-0.1,
                xanchor="center", x=0.5, font=dict(size=11),
            ),
            height=380,
            paper_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            margin=dict(l=60, r=60, t=50, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with radar_col2:
        st.markdown(
            '<div style="padding:20px 0;">'
            '<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.08em; '
            'text-transform:uppercase; opacity:0.45; margin-bottom:12px;">Reading the Radar</div>'
            '<div style="font-size:0.82rem; opacity:0.7; line-height:1.7;">'
            'Each axis represents a narrative stage. '
            '<strong style="color:#E63946;">Bulges toward Crisis/Dependence</strong> indicate '
            'at-risk populations. '
            '<strong style="color:#98D8C8;">Bulges toward Recovery</strong> signal active peer '
            'support communities. Compare shapes to identify intervention priorities.'
            '</div></div>',
            unsafe_allow_html=True,
        )

# ── Highest-risk community callout ─────────────────────────────────────────────

crisis_dep = [
    (d["label"],
     d["stage_proportions"].get("Crisis", 0),
     d["stage_proportions"].get("Dependence", 0))
    for d in filtered
]
crisis_dep.sort(key=lambda x: x[1] + x[2], reverse=True)

if crisis_dep:
    top_label, top_crisis, top_dep = crisis_dep[0]
    tier = community_risk_tier(top_crisis, top_dep)
    dominant = "Crisis" if top_crisis >= top_dep else "Dependence"
    rec = STAGE_INTERVENTIONS.get(dominant, "")
    st.markdown(
        community_risk_callout_html(top_label, tier, top_crisis, top_dep, rec),
        unsafe_allow_html=True,
    )

# ── Summary table with risk tier badges ───────────────────────────────────────

st.markdown(gradient_divider_html(), unsafe_allow_html=True)
st.markdown(
    section_header_html(
        "Community Risk Intelligence",
        "Risk tier = Crisis + Dependence proportion combined",
    ),
    unsafe_allow_html=True,
)

summary_rows = []
for d in filtered:
    crisis_pct = d["stage_proportions"].get("Crisis", 0)
    dep_pct = d["stage_proportions"].get("Dependence", 0)
    tier = community_risk_tier(crisis_pct, dep_pct)
    row = {
        "Community": d["label"],
        "Risk Tier": tier,
        "Posts Sampled": d["total_classified"],
        "Group Size": d["group_size"],
    }
    for stage in STAGE_ORDER:
        row[stage] = f"{d['stage_proportions'].get(stage, 0):.1%}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)


def _color_tier(val: str) -> str:
    colors = {
        "CRITICAL": "background-color: #E6394620; color: #E63946; font-weight: 600",
        "HIGH":     "background-color: #FFA07A20; color: #FFA07A; font-weight: 600",
        "MODERATE": "background-color: #E8A83820; color: #E8A838; font-weight: 600",
        "LOW":      "background-color: #4ECDC420; color: #4ECDC4; font-weight: 600",
    }
    return colors.get(val, "")


styled_df = summary_df.style.applymap(_color_tier, subset=["Risk Tier"])
st.dataframe(styled_df, width='stretch', hide_index=True)

# ── Intervention Recommendations ───────────────────────────────────────────────

st.markdown(gradient_divider_html(), unsafe_allow_html=True)
st.markdown(
    section_header_html(
        "Recommended Interventions by Community",
        "Stage-appropriate public health responses based on observed narrative distribution",
    ),
    unsafe_allow_html=True,
)

HIGH_RISK_STAGES = {"Crisis", "Dependence"}
THRESHOLD = 0.25

at_risk_communities = [
    d for d in filtered
    if sum(d["stage_proportions"].get(s, 0) for s in HIGH_RISK_STAGES) > THRESHOLD
]

if not at_risk_communities:
    st.info("No communities exceed the 25% Crisis+Dependence threshold with current filters.")
else:
    sorted_communities = sorted(
        at_risk_communities,
        key=lambda x: x["stage_proportions"].get("Crisis", 0) + x["stage_proportions"].get("Dependence", 0),
        reverse=True,
    )
    # Render as 2-column grid of compact cards
    cols = st.columns(2)
    for i, d in enumerate(sorted_communities):
        crisis_pct = d["stage_proportions"].get("Crisis", 0)
        dep_pct = d["stage_proportions"].get("Dependence", 0)
        tier = community_risk_tier(crisis_pct, dep_pct)
        dominant = max(
            ["Crisis", "Dependence"],
            key=lambda s: d["stage_proportions"].get(s, 0),
        )
        rec = STAGE_INTERVENTIONS.get(dominant, "")
        with cols[i % 2]:
            st.markdown(
                intervention_card_html(
                    d["label"], tier, crisis_pct, dep_pct, dominant, rec,
                ),
                unsafe_allow_html=True,
            )

# ── CDC mortality context (promoted — N2) ─────────────────────────────────────

st.markdown(gradient_divider_html(), unsafe_allow_html=True)
st.markdown(
    section_header_html(
        "Why This Matters",
        "CDC overdose mortality data provides real-world context for narrative stage analysis",
    ),
    unsafe_allow_html=True,
)

mortality = _load_mortality()
if mortality and "annual_national" in mortality:
    data = mortality["annual_national"]
    years = [d.get("year") for d in data if d.get("year")]
    totals = [d.get("total_overdose_deaths", 0) for d in data if d.get("year")]

    if years and totals:
        fig_mort = go.Figure()
        fig_mort.add_trace(go.Scatter(
            x=years, y=totals,
            mode="lines+markers",
            name="Total Overdose Deaths",
            line={"color": "#E63946", "width": 2.5},
            marker={"size": 5},
            fill="tozeroy",
            fillcolor="rgba(230,57,70,0.06)",
        ))
        # Annotate fentanyl wave
        fig_mort.add_vrect(
            x0=2013, x1=2021,
            fillcolor="rgba(230,57,70,0.05)", line_width=0,
            annotation_text="Fentanyl Wave", annotation_position="top left",
            annotation_font_color="rgba(230,57,70,0.6)",
            annotation_font_size=10,
        )
        fig_mort.update_layout(
            title="Annual Drug Overdose Deaths (CDC)",
            xaxis_title="Year",
            yaxis_title="Deaths",
            height=340,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_mort, use_container_width=True)
    else:
        st.caption("Mortality data present but format not visualizable.")
else:
    st.caption("CDC mortality data not available in expected format.")
