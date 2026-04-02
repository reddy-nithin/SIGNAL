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
st.plotly_chart(fig, width='stretch')

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
    for d in sorted(
        at_risk_communities,
        key=lambda x: x["stage_proportions"].get("Crisis", 0) + x["stage_proportions"].get("Dependence", 0),
        reverse=True,
    ):
        crisis_pct = d["stage_proportions"].get("Crisis", 0)
        dep_pct = d["stage_proportions"].get("Dependence", 0)
        rec_pct = d["stage_proportions"].get("Recovery", 0)
        tier = community_risk_tier(crisis_pct, dep_pct)
        color = {"CRITICAL": "#E63946", "HIGH": "#FFA07A", "MODERATE": "#E8A838", "LOW": "#4ECDC4"}[tier]

        dominant = max(
            ["Crisis", "Dependence"],
            key=lambda s: d["stage_proportions"].get(s, 0),
        )

        with st.expander(
            f"{d['label']}  ·  {tier}  ·  {crisis_pct + dep_pct:.0%} high-risk",
        ):
            cols = st.columns(3)
            cols[0].metric("Crisis", f"{crisis_pct:.1%}")
            cols[1].metric("Dependence", f"{dep_pct:.1%}")
            cols[2].metric("Recovery Signal", f"{rec_pct:.1%}")

            st.markdown(
                f'<div style="border-left:3px solid {color}; padding:8px 12px; '
                f'background:{color}10; border-radius:4px; margin:8px 0;">'
                f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.07em;'
                f'text-transform:uppercase;color:{color};opacity:0.8;margin-bottom:4px;">'
                f'Dominant concern</div>'
                f'<div style="font-weight:700;color:{color};">{dominant}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Recommended response:** {STAGE_INTERVENTIONS[dominant]}")

            if crisis_pct > 0.2 and dep_pct > 0.2:
                st.markdown(f"**Also indicated:** {STAGE_INTERVENTIONS['Dependence']}")

            if rec_pct > 0.2:
                st.success(
                    f"Recovery activity detected ({rec_pct:.0%} of posts) — "
                    "reinforce peer support and MAT continuation resources."
                )

# ── CDC mortality context (optional) ──────────────────────────────────────────

st.markdown(gradient_divider_html(), unsafe_allow_html=True)

with st.expander("CDC Overdose Mortality Trends (contextual reference)"):
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
            fig_mort.update_layout(
                title="Annual Drug Overdose Deaths (CDC)",
                xaxis_title="Year",
                yaxis_title="Deaths",
                height=340,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_mort, width='stretch')
        else:
            st.caption("Mortality data present but format not visualizable.")
    else:
        st.caption("CDC mortality data not available in expected format.")
