"""
Phase 4 test suite — Clinical Grounding (Layer 3) + Brief Generator (Layer 4).

Tests clinical_contextualizer, FAERS lookup, interaction detection,
and brief generation. Uses mocked/synthetic data where possible to
avoid requiring Vertex AI credentials for basic tests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from signal.grounding.types import (
    ClinicalContext,
    FAERSSignal,
    InteractionWarning,
    RetrievedEvidence,
    SignalReport,
)
from signal.substance.types import SubstanceMatch, DetectionResult, EnsembleResult
from signal.narrative.types import StageClassification, NarrativeEnsembleResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_vertex_credentials() -> bool:
    try:
        import google.auth
        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        return False


HAS_VERTEX = _has_vertex_credentials()


def _make_substance_match(
    clinical_name: str = "fentanyl",
    drug_class: str = "opioid",
    confidence: float = 0.85,
    is_negated: bool = False,
) -> SubstanceMatch:
    return SubstanceMatch(
        substance_name=clinical_name,
        clinical_name=clinical_name,
        drug_class=drug_class,
        confidence=confidence,
        method="rule_based",
        context_snippet=f"test context for {clinical_name}",
        is_negated=is_negated,
        char_start=0,
        char_end=10,
    )


def _make_ensemble_result(
    matches: tuple[SubstanceMatch, ...] | None = None,
) -> EnsembleResult:
    if matches is None:
        matches = (_make_substance_match(),)
    return EnsembleResult(
        post_id="test_post",
        matches=matches,
        method_results=(),
        agreement_count=1,
    )


# ── Test FAERS Signal Lookup ─────────────────────────────────────────────────

class TestFAERSLookup:
    """Test FAERS and supplementary signal loading and lookup."""

    def test_lookup_morphine_returns_signals(self):
        """morphine is in faers_signal_results.json, should have signals."""
        import signal.grounding.clinical_contextualizer as cc
        cc._faers_cache = None
        cc._supp_cache = None

        signals = cc.lookup_faers_signals("morphine")
        assert len(signals) > 0
        assert all(isinstance(s, FAERSSignal) for s in signals)
        assert any(s.reaction == "Respiratory depression" for s in signals)

    def test_lookup_alcohol_returns_supplementary(self):
        """alcohol is in supplementary_signals.json, should find curated signals."""
        import signal.grounding.clinical_contextualizer as cc
        cc._faers_cache = None
        cc._supp_cache = None

        signals = cc.lookup_faers_signals("alcohol")
        assert len(signals) > 0
        assert any(s.source == "literature_curated" for s in signals)

    def test_lookup_nonexistent_returns_empty(self):
        """Unknown drug should return empty tuple."""
        import signal.grounding.clinical_contextualizer as cc
        cc._faers_cache = None
        cc._supp_cache = None

        signals = cc.lookup_faers_signals("nonexistent_drug_xyz")
        assert signals == ()

    def test_faers_signal_has_prr_and_ror(self):
        """Known opioid signals should have PRR and ROR values."""
        import signal.grounding.clinical_contextualizer as cc
        cc._faers_cache = None
        cc._supp_cache = None

        signals = cc.lookup_faers_signals("morphine")
        resp_dep = [s for s in signals if s.reaction == "Respiratory depression"]
        assert len(resp_dep) > 0
        assert resp_dep[0].prr is not None
        assert resp_dep[0].prr > 1.0

    def test_alprazolam_in_supplementary(self):
        """alprazolam should have signals from supplementary file."""
        import signal.grounding.clinical_contextualizer as cc
        cc._faers_cache = None
        cc._supp_cache = None

        signals = cc.lookup_faers_signals("alprazolam")
        assert len(signals) > 0
        assert any("dependence" in s.reaction.lower() for s in signals)


# ── Test Interaction Detection ───────────────────────────────────────────────

class TestInteractionDetection:
    """Test poly-drug interaction detection logic."""

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_opioid_benzo_interaction(self):
        """opioid + benzo should trigger interaction warning."""
        from signal.grounding.clinical_contextualizer import detect_interactions
        from signal.grounding.indexer import HybridRetriever

        retriever = HybridRetriever()
        warnings = detect_interactions(["opioid", "benzo"], retriever)
        assert len(warnings) > 0
        assert any(
            "opioid" in w.substances and "benzo" in w.substances
            for w in warnings
        )

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_single_substance_no_interaction(self):
        """Single substance class should not trigger interactions."""
        from signal.grounding.clinical_contextualizer import detect_interactions
        from signal.grounding.indexer import HybridRetriever

        retriever = HybridRetriever()
        warnings = detect_interactions(["opioid"], retriever)
        assert len(warnings) == 0


# ── Test Clinical Context Building ───────────────────────────────────────────

class TestClinicalContext:
    """Test build_clinical_context and contextualize_all."""

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_build_context_returns_evidence(self):
        """Clinical context for fentanyl should include evidence chunks."""
        from signal.grounding.clinical_contextualizer import build_clinical_context
        from signal.grounding.indexer import HybridRetriever

        retriever = HybridRetriever()
        ctx = build_clinical_context("fentanyl", "opioid", "Crisis", retriever)
        assert isinstance(ctx, ClinicalContext)
        assert ctx.substance == "fentanyl"
        assert len(ctx.evidence) > 0
        assert len(ctx.faers_signals) > 0

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_contextualize_all_with_ensemble(self):
        """contextualize_all should process EnsembleResult matches."""
        from signal.grounding.clinical_contextualizer import contextualize_all
        from signal.grounding.indexer import HybridRetriever

        retriever = HybridRetriever()
        ensemble = _make_ensemble_result(matches=(
            _make_substance_match("fentanyl", "opioid"),
            _make_substance_match("alprazolam", "benzo"),
        ))
        contexts = contextualize_all(ensemble, "Crisis", retriever)
        assert len(contexts) == 2
        substances = {ctx.substance for ctx in contexts}
        assert "fentanyl" in substances
        assert "alprazolam" in substances

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_negated_substance_excluded(self):
        """Negated substances should not get clinical context."""
        from signal.grounding.clinical_contextualizer import contextualize_all
        from signal.grounding.indexer import HybridRetriever

        retriever = HybridRetriever()
        ensemble = _make_ensemble_result(matches=(
            _make_substance_match("fentanyl", "opioid", is_negated=True),
        ))
        contexts = contextualize_all(ensemble, "Curiosity", retriever)
        assert len(contexts) == 0

    def test_empty_ensemble_returns_empty(self):
        """Empty matches should return empty tuple."""
        from signal.grounding.clinical_contextualizer import contextualize_all

        retriever = MagicMock()
        ensemble = _make_ensemble_result(matches=())
        contexts = contextualize_all(ensemble, "Curiosity", retriever)
        assert contexts == ()


# ── Test Brief Generator ─────────────────────────────────────────────────────

class TestBriefGenerator:
    """Test brief generation with mocked Gemini."""

    def test_empty_contexts_returns_fallback(self):
        """No contexts → fallback brief."""
        from signal.synthesis.brief_generator import generate_brief

        brief = generate_brief("test text", "Crisis", 0.9, ())
        assert "No substances detected" in brief

    def test_brief_with_mocked_gemini(self):
        """Mocked Gemini should produce a brief with citations."""
        from signal.synthesis.brief_generator import generate_brief

        mock_response = """SIGNAL INTELLIGENCE BRIEF

1. SUBSTANCE IDENTIFICATION
Fentanyl (opioid) detected. [KB:ingredient_fentanyl.txt]

2. NARRATIVE STAGE ASSESSMENT
Crisis stage (confidence: 0.90). [KB:ingredient_fentanyl.txt]

3. CLINICAL RISK PROFILE
Fentanyl is a potent synthetic opioid. [KB:ingredient_fentanyl.txt] [FAERS:fentanyl+Respiratory depression]

4. POLY-DRUG INTERACTION RISKS
None detected.

5. STAGE-SPECIFIC RISK ANNOTATION
Crisis stage with opioid use indicates acute overdose risk. [KB:ingredient_fentanyl.txt]

6. RECOMMENDED ACTIONS
Assess naloxone availability. [KB:ingredient_naloxone.txt]"""

        ctx = ClinicalContext(
            substance="fentanyl",
            drug_class="opioid",
            evidence=(
                RetrievedEvidence(
                    chunk_filename="ingredient_fentanyl.txt",
                    chunk_type="pharmacology",
                    drug_name="fentanyl",
                    relevance_score=0.92,
                    text_snippet="Fentanyl is a synthetic opioid...",
                ),
            ),
            faers_signals=(
                FAERSSignal(
                    drug_name="fentanyl",
                    reaction="Respiratory depression",
                    prr=17.26,
                    ror=17.41,
                    source="faers",
                ),
            ),
            interactions=(),
            narrative_stage="Crisis",
        )

        with patch("signal.synthesis.brief_generator._call_gemini", return_value=mock_response):
            brief = generate_brief("I overdosed on fent", "Crisis", 0.90, (ctx,))
            assert "SIGNAL INTELLIGENCE BRIEF" in brief
            assert "[KB:" in brief
            assert "[FAERS:" in brief


# ── Test Type Definitions ────────────────────────────────────────────────────

class TestGroundingTypes:
    """Test that grounding type dataclasses are frozen and well-formed."""

    def test_clinical_context_is_frozen(self):
        ctx = ClinicalContext(
            substance="test",
            drug_class="opioid",
            evidence=(),
            faers_signals=(),
            interactions=(),
            narrative_stage="Curiosity",
        )
        with pytest.raises(AttributeError):
            ctx.substance = "modified"  # type: ignore[misc]

    def test_signal_report_is_frozen(self):
        report = SignalReport(
            post_id="test",
            original_text="test",
            substance_results=None,
            narrative_results=None,
            clinical_contexts=(),
            analyst_brief="",
            elapsed_ms=0.0,
        )
        with pytest.raises(AttributeError):
            report.analyst_brief = "modified"  # type: ignore[misc]

    def test_retrieved_evidence_fields(self):
        ev = RetrievedEvidence(
            chunk_filename="test.txt",
            chunk_type="pharmacology",
            drug_name="morphine",
            relevance_score=0.85,
            text_snippet="test snippet",
        )
        assert ev.chunk_filename == "test.txt"
        assert ev.relevance_score == 0.85
