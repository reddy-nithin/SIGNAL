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

    rows = []
    for m in matches:
        rows.append({
            "Slang Term": m.substance_name,
            "Clinical Name": m.clinical_name,
            "Drug Class": m.drug_class.title(),
            "Confidence": f"{m.confidence:.0%}",
            "Negated": "Yes" if m.is_negated else "",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # Per-method breakdown
    with st.expander("Per-method detection details"):
        for mr in report.substance_results.method_results:
            method_label = mr.method.replace("_", " ").title()
            det_names = [m.clinical_name for m in mr.matches if not m.is_negated]
            color = METHOD_COLORS.get(mr.method, "#888")
            st.markdown(
                f'<div style="padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="color:{color}; font-weight:700;">{method_label}</span> '
                f'<span style="opacity:0.5; font-size:0.85em;">({mr.elapsed_ms:.0f}ms)</span>'
                f'<span style="margin-left:12px;">'
                f'{", ".join(det_names) if det_names else "<em>none detected</em>"}'
                f'</span></div>',
                unsafe_allow_html=True,
            )


def _render_narrative(report) -> None:
    """Render Layer 2: Narrative Stage tab."""
    top = report.narrative_results.top_stage
    n_methods = len(report.narrative_results.method_results)
    color = STAGE_COLORS.get(top.stage, "#888")

    st.markdown(
        f'<div style="display:flex; align-items:baseline; gap:14px; margin-bottom:16px;">'
        f'<span style="font-size:1.5rem; font-weight:800; color:{color};">{top.stage}</span>'
        f'<span style="font-size:1rem; opacity:0.7; font-weight:600;">{top.confidence:.0%} confidence</span>'
        f'<span style="font-size:0.85rem; opacity:0.5;">— Agreement: {agreement_badge(report.narrative_results.agreement_count, n_methods)}</span>'
        f'</div>',
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
        height=300,
        **{
            **PLOTLY_LAYOUT,
            "yaxis": {
                **PLOTLY_LAYOUT.get("yaxis", {}),
                "categoryorder": "array",
                "categoryarray": list(reversed(STAGE_ORDER)),
            },
        },
    )
    st.plotly_chart(fig, width='stretch')

    # Per-method comparison
    from signal.narrative.ensemble import build_comparison_table
    comp_rows = build_comparison_table(report.narrative_results)
    df = pd.DataFrame(comp_rows)
    pivot = df.pivot_table(index="stage", columns="method", values="confidence")
    stage_order_present = [s for s in STAGE_ORDER if s in pivot.index]
    pivot = pivot.reindex(stage_order_present)

    with st.expander("Per-method stage scores"):
        st.dataframe(pivot.style.format("{:.0%}").background_gradient(
            cmap="YlOrRd", axis=None
        ), width='stretch')


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
            # Evidence chunks
            if ctx.evidence:
                st.markdown(
                    section_header_html("Retrieved Knowledge Chunks"),
                    unsafe_allow_html=True,
                )
                ev_rows = []
                for ev in ctx.evidence:
                    ev_rows.append({
                        "Chunk": ev.chunk_filename,
                        "Type": ev.chunk_type,
                        "Relevance": f"{ev.relevance_score:.3f}",
                        "Snippet": ev.text_snippet[:200] + "…" if len(ev.text_snippet) > 200 else ev.text_snippet,
                    })
                st.dataframe(pd.DataFrame(ev_rows), width='stretch', hide_index=True)

            # FAERS signals
            if ctx.faers_signals:
                st.markdown(
                    section_header_html("Adverse Event Signals"),
                    unsafe_allow_html=True,
                )
                sig_rows = []
                for sig in ctx.faers_signals:
                    prr_str = f"{sig.prr:.1f}" if sig.prr is not None else "—"
                    ror_str = f"{sig.ror:.1f}" if sig.ror is not None else "—"
                    sig_rows.append({
                        "Reaction": sig.reaction,
                        "PRR": prr_str,
                        "ROR": ror_str,
                        "Source": sig.source,
                    })
                df_sig = pd.DataFrame(sig_rows)
                st.dataframe(df_sig, width='stretch', hide_index=True)

            # Interaction warnings
            if ctx.interactions:
                for iw in ctx.interactions:
                    st.warning(
                        f"**Interaction Risk:** {' + '.join(iw.substances)}\n\n"
                        f"{iw.risk_description}\n\n"
                        f"*Source: {iw.source_chunk}*"
                    )


def _render_brief(report, brief_override: str | None = None) -> None:
    """Render Layer 4: Analyst Brief tab — structured section renderer with fallback.

    Args:
        report: SignalReport (or SimpleNamespace for cached reports).
        brief_override: If provided, use this string instead of report.analyst_brief.
            Used in two-phase rendering where the brief is generated after core results.
    """
    brief_text = brief_override if brief_override is not None else (
        report.analyst_brief if hasattr(report, "analyst_brief") else None
    )
    if not brief_text:
        st.info("No analyst brief generated (no substances detected or brief was skipped).")
        return

    # Try to split into sections on patterns like "1. SUBSTANCE IDENTIFICATION" or "## Header"
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
        st.markdown(preamble)

    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if header and body:
            sections.append((header, body))

    if not sections:
        st.markdown(brief_text)
        return

    action_keywords = {"RECOMMEND", "ACTION", "INTERVENTION", "RESPONSE"}

    for j, (header, body) in enumerate(sections):
        is_action = any(kw in header.upper() for kw in action_keywords)
        is_last = j == len(sections) - 1

        color = "#FAFAFA"
        if "SUBSTANCE" in header.upper() or "DRUG" in header.upper():
            color = "#4ECDC4"
        elif "NARRATIVE" in header.upper() or "STAGE" in header.upper():
            color = "#45B7D1"
        elif "RISK" in header.upper() or "CLINICAL" in header.upper():
            color = "#FFA07A"
        elif "INTERACTION" in header.upper() or "POLY" in header.upper():
            color = "#E63946"
        elif "RECOMMEND" in header.upper() or "ACTION" in header.upper():
            color = "#98D8C8"

        with st.expander(header, expanded=(is_action or is_last)):
            st.markdown(
                f'<div style="border-left:3px solid {color}; padding:2px 0 2px 12px; '
                f'margin-bottom:10px;">'
                f'<span style="color:{color}; font-weight:700; font-size:0.85rem; '
                f'letter-spacing:0.05em; text-transform:uppercase;">{header}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(body)


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
