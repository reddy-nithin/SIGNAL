"""
SIGNAL Dashboard — Page 1: Deep Analysis
==========================================
Paste any social media post → full 4-layer SIGNAL analysis.
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

import html
import json
import re
import time
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from signal.config import CACHE_DIR
from signal.dashboard.theme import (
    STAGE_COLORS,
    STAGE_ORDER,
    METHOD_COLORS,
    PLOTLY_LAYOUT,
    inject_css,
    risk_banner_html,
    section_header_html,
    gradient_divider_html,
    agreement_badge,
    highlighted_text_html,
    narrative_arc_indicator_html,
    evidence_meter_html,
    substance_badge_html,
    evidence_card_html,
    brief_summary_card_html,
    brief_section_html,
    confidence_matrix_html,
)

inject_css()

# ── Risk Level Map ─────────────────────────────────────────────────────────────

RISK_MAP: dict[str, tuple[str, str]] = {
    "Curiosity":       ("info",    "Low risk — Prevention and education window is open. Monitoring recommended."),
    "Experimentation": ("info",    "Low-moderate — Early-stage use. Education and brief intervention appropriate."),
    "Regular Use":     ("warning", "Moderate — Patterned use established. Screening and brief intervention recommended."),
    "Dependence":      ("warning", "High — Dependence markers present. MAT referral and treatment access resources indicated."),
    "Crisis":          ("error",   "Acute risk — Immediate crisis intervention and harm reduction resources indicated."),
    "Recovery":        ("success", "Recovery — Peer support, MAT maintenance, and relapse prevention resources recommended."),
}

# ── Demo examples ──────────────────────────────────────────────────────────────

DEMO_EXAMPLES: dict[str, str] = {
    "Curiosity — opioids": (
        "Has anyone tried oxy for back pain? What does it feel like? "
        "Is it safe to take occasionally or is it too risky?"
    ),
    "Experimentation — benzo + alcohol": (
        "Tried mixing xans with a few drinks at a party last weekend. "
        "Wild experience, not something I'd do regularly though."
    ),
    "Dependence — opioids": (
        "I literally cannot get through a day without my percs anymore. "
        "When I try to stop I get so sick I can't move. I need help."
    ),
    "Crisis — poly-drug": (
        "I overdosed on fentanyl last night. My roommate had to call 911. "
        "I was mixing lean with bars and I almost died."
    ),
    "Recovery — MAT": (
        "90 days clean off heroin today. Suboxone has been a lifesaver. "
        "My sponsor says the first year is the hardest but I'm making it."
    ),
}

DEMO_CACHE_PATH = CACHE_DIR / "demo_reports.json"


# ── Pipeline access ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SIGNAL pipeline...")
def _get_pipeline():
    from signal.synthesis.pipeline import SIGNALPipeline
    from signal.config import VERTEX_PROJECT_ID
    from signal.substance.embedding_detector import load_or_build_substance_embeddings
    from signal.narrative.fine_tuned_classifier import _load_model as _load_distilbert

    if not VERTEX_PROJECT_ID:
        # No Vertex AI credentials — pre-warm SBERT so the 5-40s model load
        # happens here (with spinner) instead of on the first user click.
        from signal.grounding.indexer import get_sbert_model
        model = get_sbert_model()
        model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)

    # Pre-load substance prototype embeddings and DistilBERT at startup.
    # Both are module-level singletons: this ensures they are resident in
    # memory before any user request arrives, eliminating first-click lag.
    load_or_build_substance_embeddings()
    _load_distilbert()
    return SIGNALPipeline()


def _load_cached_reports() -> dict | None:
    """Load pre-cached demo reports if available."""
    if DEMO_CACHE_PATH.exists():
        try:
            return json.loads(DEMO_CACHE_PATH.read_text())
        except Exception:
            return None
    return None


def _dict_to_report(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict/list to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_report(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_report(item) for item in d]
    return d


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _render_risk_banner(stage: str) -> None:
    """Render the styled risk level banner using the theme helper."""
    if stage not in RISK_MAP:
        return
    _, message = RISK_MAP[stage]
    st.markdown(risk_banner_html(stage, message), unsafe_allow_html=True)


def _render_substances(report) -> None:
    """Render Layer 1: Substance Resolution tab."""
    matches = report.substance_results.matches
    if not matches:
        st.info("No substances detected in this text.")
        return

    n_methods = len(report.substance_results.method_results)
    st.markdown(
        section_header_html(
            f"{len(matches)} substance(s) detected",
            f"Agreement: {agreement_badge(report.substance_results.agreement_count, n_methods)}",
        ),
        unsafe_allow_html=True,
    )

    # Substance badge grid (replaces plain dataframe)
    badges = "".join(
        substance_badge_html(m.substance_name, m.clinical_name, m.drug_class, m.confidence)
        for m in matches if not m.is_negated
    )
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:6px; margin:12px 0;">{badges}</div>',
        unsafe_allow_html=True,
    )

    # Inline method comparison strip (replaces hidden expander)
    st.markdown(
        '<div style="font-size:0.68rem; font-weight:700; letter-spacing:0.08em; '
        'text-transform:uppercase; opacity:0.4; margin:16px 0 8px 0;">Per-Method Detection</div>',
        unsafe_allow_html=True,
    )
    for mr in report.substance_results.method_results:
        method_label = mr.method.replace("_", " ").title()
        det_names = [m.clinical_name for m in mr.matches if not m.is_negated]
        color = METHOD_COLORS.get(mr.method, "#888")
        if det_names:
            sub_pills = " ".join(
                f'<span style="background:{color}15; color:{color}; border:1px solid {color}30; '
                f'border-radius:12px; padding:2px 10px; font-size:0.72rem; font-weight:600;">'
                f'{n}</span>'
                for n in det_names
            )
        else:
            sub_pills = '<span style="opacity:0.35; font-size:0.76rem; font-style:italic;">none detected</span>'
        elapsed = getattr(mr, "elapsed_ms", 0)
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; padding:8px 0; '
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="min-width:90px; color:{color}; font-weight:700; font-size:0.82rem;">'
            f'{method_label}</span>'
            f'<span style="opacity:0.3; font-size:0.72rem;">{elapsed:.0f}ms</span>'
            f'<div style="display:flex; flex-wrap:wrap; gap:4px;">{sub_pills}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_narrative(report) -> None:
    """Render Layer 2: Narrative Stage tab."""
    top = report.narrative_results.top_stage
    n_methods = len(report.narrative_results.method_results)
    color = STAGE_COLORS.get(top.stage, "#888")

    # ── Visual Arc Position Indicator (D2: core novelty visualization) ─────
    conf_dict = {sc.stage: sc.confidence for sc in report.narrative_results.all_stages}
    st.markdown(
        narrative_arc_indicator_html(top.stage, conf_dict),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="display:flex; align-items:baseline; gap:14px; margin-bottom:16px; '
        f'justify-content:center;">'
        f'<span style="font-size:1.3rem; font-weight:800; color:{color};">{top.stage}</span>'
        f'<span style="font-size:0.9rem; opacity:0.7; font-weight:600;">{top.confidence:.0%} confidence</span>'
        f'<span style="font-size:0.8rem; opacity:0.5;">Agreement: {agreement_badge(report.narrative_results.agreement_count, n_methods)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Full confidence matrix (6 stages × 3 methods) ────────────────────────
    method_names: list[str] = []
    matrix_values: list[list[float]] = []
    for mr in report.narrative_results.method_results:
        method_names.append(mr.method)
        stage_map = {sc.stage: sc.confidence for sc in mr.all_stages}
        matrix_values.append([stage_map.get(s, 0) for s in STAGE_ORDER])

    if method_names:
        st.markdown(
            '<div style="font-size:0.68rem; font-weight:700; letter-spacing:0.08em; '
            'text-transform:uppercase; opacity:0.4; margin:8px 0 4px 0;">'
            'Method × Stage Confidence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            confidence_matrix_html(STAGE_ORDER, method_names, matrix_values),
            unsafe_allow_html=True,
        )

    # Ensemble confidence bar chart
    stages = [sc.stage for sc in report.narrative_results.all_stages]
    confs = [sc.confidence for sc in report.narrative_results.all_stages]

    fig = go.Figure(go.Bar(
        x=confs, y=stages, orientation="h",
        marker_color=[STAGE_COLORS.get(s, "#888") for s in stages],
        marker_line_width=0,
        text=[f"{c:.0%}" for c in confs],
        textposition="auto",
    ))
    fig.update_layout(
        title="Ensemble Stage Confidence",
        xaxis_title="Confidence",
        height=280,
        **{
            **PLOTLY_LAYOUT,
            "yaxis": {
                **PLOTLY_LAYOUT.get("yaxis", {}),
                "categoryorder": "array",
                "categoryarray": list(reversed(STAGE_ORDER)),
            },
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_grounding(report) -> None:
    """Render Layer 3: Clinical Grounding tab."""
    contexts = report.clinical_contexts
    if not contexts:
        st.info("No clinical context available (no substances detected).")
        return

    for ctx in contexts:
        with st.expander(
            f"{ctx.substance}  ·  {ctx.drug_class}",
            expanded=len(contexts) <= 2,
        ):
            # Evidence strength meter (D5)
            n_chunks = len(ctx.evidence) if ctx.evidence else 0
            n_signals = len(ctx.faers_signals) if ctx.faers_signals else 0
            avg_rel = (
                sum(e.relevance_score for e in ctx.evidence) / n_chunks
                if ctx.evidence and n_chunks > 0 else 0
            )
            st.markdown(
                evidence_meter_html(n_chunks, n_signals, avg_rel),
                unsafe_allow_html=True,
            )

            # Evidence chunks as visual cards
            if ctx.evidence:
                st.markdown(
                    section_header_html("Retrieved Knowledge Chunks"),
                    unsafe_allow_html=True,
                )
                drug_color = {
                    "opioid": "#FF6B6B", "benzodiazepine": "#E8A838",
                    "stimulant": "#45B7D1", "alcohol": "#B07CC6",
                }.get(ctx.drug_class.lower(), "#4ECDC4")
                cards_html = "".join(
                    evidence_card_html(
                        ev.chunk_filename, ev.chunk_type, ev.relevance_score,
                        ev.text_snippet[:200] + "..." if len(ev.text_snippet) > 200 else ev.text_snippet,
                        drug_color,
                    )
                    for ev in ctx.evidence
                )
                st.markdown(cards_html, unsafe_allow_html=True)

            # FAERS signals — bubble chart + table
            if ctx.faers_signals:
                st.markdown(
                    section_header_html("Adverse Event Signals"),
                    unsafe_allow_html=True,
                )
                # Build data for bubble chart
                sig_rows = []
                for sig in ctx.faers_signals:
                    prr_val = sig.prr if sig.prr is not None else 0
                    ror_val = sig.ror if sig.ror is not None else 0
                    sig_rows.append({
                        "Reaction": sig.reaction,
                        "PRR": prr_val,
                        "ROR": ror_val,
                        "Source": sig.source,
                    })
                df_sig = pd.DataFrame(sig_rows)

                # Bubble chart (D3: visual proof of clinical grounding)
                valid_sigs = df_sig[(df_sig["PRR"] > 0) & (df_sig["ROR"] > 0)]
                if len(valid_sigs) >= 3:
                    fig_bubble = go.Figure()
                    fig_bubble.add_trace(go.Scatter(
                        x=valid_sigs["PRR"],
                        y=valid_sigs["ROR"],
                        mode="markers",
                        marker=dict(
                            size=[max(8, min(v * 3, 40)) for v in valid_sigs["PRR"]],
                            color=valid_sigs["PRR"],
                            colorscale=[[0, "#45B7D1"], [0.5, "#FFA07A"], [1, "#E63946"]],
                            showscale=True,
                            colorbar=dict(title="PRR", thickness=12),
                            opacity=0.75,
                            line=dict(width=1, color="rgba(255,255,255,0.2)"),
                        ),
                        text=valid_sigs["Reaction"],
                        hovertemplate="<b>%{text}</b><br>PRR: %{x:.1f}<br>ROR: %{y:.1f}<extra></extra>",
                    ))
                    fig_bubble.add_hline(y=1, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                                         annotation_text="ROR=1 (baseline)", annotation_position="bottom right",
                                         annotation_font_color="rgba(255,255,255,0.3)")
                    fig_bubble.add_vline(x=1, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                    fig_bubble.update_layout(
                        title="Adverse Event Signal Strength",
                        xaxis_title="PRR (Proportional Reporting Ratio)",
                        yaxis_title="ROR (Reporting Odds Ratio)",
                        height=320,
                        showlegend=False,
                        **PLOTLY_LAYOUT,
                    )
                    st.plotly_chart(fig_bubble, use_container_width=True)

                with st.expander("Signal details table"):
                    display_df = df_sig.copy()
                    display_df["PRR"] = display_df["PRR"].apply(lambda v: f"{v:.1f}" if v > 0 else "—")
                    display_df["ROR"] = display_df["ROR"].apply(lambda v: f"{v:.1f}" if v > 0 else "—")
                    st.dataframe(display_df, width='stretch', hide_index=True)

            # Interaction warnings
            if ctx.interactions:
                for iw in ctx.interactions:
                    st.warning(
                        f"**Interaction Risk:** {' + '.join(iw.substances)}\n\n"
                        f"{iw.risk_description}\n\n"
                        f"*Source: {iw.source_chunk}*"
                    )


def _render_brief(report, brief_override: str | None = None) -> None:
    """Render Layer 4: Analyst Brief tab — summary card + visible sections."""
    brief_text = brief_override if brief_override is not None else (
        report.analyst_brief if hasattr(report, "analyst_brief") else None
    )
    if not brief_text:
        st.info("No analyst brief generated (no substances detected or brief was skipped).")
        return

    # Summary card from report data (not brief text)
    top_stage = report.narrative_results.top_stage.stage
    substances = [m.clinical_name for m in report.substance_results.matches if not m.is_negated]
    # Extract first actionable sentence from brief for recommendation preview
    rec_match = re.search(
        r"(?:recommend|action|intervention|response)[:\s]*(.+?)(?:\n|$)",
        brief_text, re.IGNORECASE,
    )
    rec_preview = rec_match.group(1).strip() if rec_match else ""
    st.markdown(
        brief_summary_card_html(substances, top_stage, top_stage, rec_preview),
        unsafe_allow_html=True,
    )

    # Parse into sections
    section_pattern = re.compile(
        r"(?:^|\n)(?:\d+\.\s+)?([A-Z][A-Z\s&/\-]{4,}:?)\s*\n",
        re.MULTILINE,
    )
    parts = section_pattern.split(brief_text)

    if len(parts) < 3:
        st.markdown(brief_text)
        return

    preamble = parts[0].strip()
    if preamble:
        st.markdown(
            f'<div style="font-size:0.86rem; opacity:0.7; line-height:1.6; '
            f'margin:10px 0 16px 0;">{preamble}</div>',
            unsafe_allow_html=True,
        )

    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if header and body:
            sections.append((header, body))

    if not sections:
        st.markdown(brief_text)
        return

    # Section color + icon mapping
    section_style: dict[str, tuple[str, str]] = {
        "SUBSTANCE": ("#4ECDC4", "\U0001f48a"),
        "DRUG": ("#4ECDC4", "\U0001f48a"),
        "NARRATIVE": ("#45B7D1", "\U0001f4ca"),
        "STAGE": ("#45B7D1", "\U0001f4ca"),
        "RISK": ("#FFA07A", "\u26a0\ufe0f"),
        "CLINICAL": ("#FFA07A", "\u26a0\ufe0f"),
        "INTERACTION": ("#E63946", "\u26a1"),
        "POLY": ("#E63946", "\u26a1"),
        "RECOMMEND": ("#98D8C8", "\u2705"),
        "ACTION": ("#98D8C8", "\u2705"),
        "INTERVENTION": ("#98D8C8", "\u2705"),
    }

    for header, body in sections:
        color, icon = "#FAFAFA", ""
        for keyword, (c, ic) in section_style.items():
            if keyword in header.upper():
                color, icon = c, ic
                break
        st.markdown(
            brief_section_html(header, body, color, icon),
            unsafe_allow_html=True,
        )


# ── Main page ──────────────────────────────────────────────────────────────────

st.markdown(
    section_header_html(
        "Deep Analysis",
        "Full 4-layer SIGNAL analysis on any social media post",
    ),
    unsafe_allow_html=True,
)

# Input area
col_input, col_example = st.columns([3, 1])

with col_example:
    example_choice = st.selectbox(
        "Load example",
        ["Custom input"] + list(DEMO_EXAMPLES.keys()),
    )

with col_input:
    default_text = DEMO_EXAMPLES.get(example_choice, "") if example_choice != "Custom input" else ""
    user_text = st.text_area(
        "Post text",
        value=default_text,
        height=150,
        placeholder="Paste a social media post here…",
        max_chars=3000,
    )

# Show styled post preview when a demo is loaded
if example_choice != "Custom input" and user_text.strip():
    safe_preview = html.escape(user_text.strip())
    st.markdown(
        f'<div style="border:1px solid rgba(255,255,255,0.08); border-left:3px solid rgba(255,255,255,0.25); '
        f'background:rgba(255,255,255,0.02); border-radius:6px; padding:12px 16px; '
        f'font-size:0.88rem; opacity:0.75; line-height:1.6; margin:4px 0 12px 0; '
        f'font-style:italic;">"{safe_preview}"</div>',
        unsafe_allow_html=True,
    )

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

# Pre-warm pipeline at page load (not on first click) so the spinner shows
# immediately when the user arrives, not as a hidden hang on first analysis.
_get_pipeline()

# ── Rate limiting ──────────────────────────────────────────────────────────────

_RATE_LIMIT = 10       # max analyses per session
_COOLDOWN_SEC = 5      # seconds between consecutive analyses

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0.0

# ── Result rendering (from cache or live pipeline) ─────────────────────────────

cached_reports = _load_cached_reports()
report = None

_live_brief: str | None = None  # brief generated in Phase 2 of two-phase rendering

if analyze_clicked and user_text.strip():
    now = time.time()
    seconds_since_last = now - st.session_state.last_analysis_time
    if st.session_state.analysis_count >= _RATE_LIMIT:
        st.warning(f"Session limit reached ({_RATE_LIMIT} analyses). Refresh the page to continue.")
        st.stop()
    elif seconds_since_last < _COOLDOWN_SEC:
        remaining = int(_COOLDOWN_SEC - seconds_since_last) + 1
        st.info(f"Please wait {remaining}s before analyzing again.")
        st.stop()
    else:
        pipeline = _get_pipeline()

        # Phase 1: Run L1+L2 (parallel) + L3 — show core results fast
        with st.status("Running SIGNAL analysis…", expanded=True) as _status:
            _status.update(label="Detecting substances & classifying narrative stage (parallel)…")
            try:
                report = pipeline.analyze_core(user_text.strip())
            except ValueError as e:
                st.warning(f"Invalid input: {e}")
                st.stop()
            except Exception:
                st.error("Analysis failed. Please try again or select a demo example.")
                st.stop()

            _status.update(label="Retrieving clinical evidence…")
            # Layer 3 already ran inside analyze_core(); just update the label.

            _status.update(
                label=f"Core analysis complete — {report.elapsed_ms:.0f} ms  ·  Generating analyst brief…",
                state="running",
            )

        st.session_state.analysis_count += 1
        st.session_state.last_analysis_time = time.time()

        # Phase 2: Generate brief separately so core results render first
        if report.clinical_contexts:
            from signal.synthesis.brief_generator import generate_brief as _gen_brief
            with st.spinner("Generating analyst brief…"):
                try:
                    _live_brief = _gen_brief(
                        original_text=user_text.strip(),
                        narrative_stage=report.narrative_results.top_stage.stage,
                        narrative_confidence=report.narrative_results.top_stage.confidence,
                        contexts=report.clinical_contexts,
                    )
                except Exception as exc:
                    _live_brief = f"[Brief generation failed: {exc!r}]"

elif not analyze_clicked and example_choice != "Custom input" and cached_reports:
    cached_dict = cached_reports.get(example_choice)
    if cached_dict:
        report = _dict_to_report(cached_dict)

elif analyze_clicked:
    st.warning("Please enter some text to analyze.")

# ── Render results ─────────────────────────────────────────────────────────────

if report is not None:
    substance_count = len(report.substance_results.matches)
    top_stage = report.narrative_results.top_stage.stage
    stage_conf = report.narrative_results.top_stage.confidence
    elapsed = getattr(report, "elapsed_ms", 0)

    # ── Inline highlighted text (D1: biggest visual impact) ───────────────
    _source_text = user_text.strip() if user_text.strip() else ""
    if _source_text and report.substance_results.matches:
        st.markdown(
            highlighted_text_html(_source_text, report.substance_results.matches),
            unsafe_allow_html=True,
        )

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Substances", substance_count)
    c2.metric("Narrative Stage", top_stage)
    c3.metric("Stage Confidence", f"{stage_conf:.0%}")
    if elapsed:
        c4.metric("Elapsed", f"{elapsed:.0f} ms")

    # Risk Level synthesis banner
    _render_risk_banner(top_stage)

    st.markdown(gradient_divider_html(), unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Substances",
        "Narrative Stage",
        "Clinical Grounding",
        "Analyst Brief",
    ])

    with tab1:
        _render_substances(report)
    with tab2:
        _render_narrative(report)
    with tab3:
        _render_grounding(report)
    with tab4:
        # For live analyses: use the brief generated in Phase 2 (_live_brief).
        # For cached demo reports: brief is already on the report object.
        _render_brief(report, brief_override=_live_brief)

elif example_choice != "Custom input" and cached_reports is None:
    st.info(
        "Pre-cached demo results not found. Click **Analyze** to run live, "
        "or pre-cache with: `python -m signal.dashboard.demo_cache`"
    )
