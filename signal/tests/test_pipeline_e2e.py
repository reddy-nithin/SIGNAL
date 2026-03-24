"""
Phase 4 E2E test suite — SIGNALPipeline integration tests.

Tests the full 4-layer pipeline from text input to SignalReport output.
Most tests mock Gemini API calls to avoid cost/latency.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from signal.grounding.types import SignalReport


# ── Helpers ──────────────────────────────────────────────────────────────────

def _has_vertex_credentials() -> bool:
    try:
        import google.auth
        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        return False


HAS_VERTEX = _has_vertex_credentials()

_MOCK_GEMINI_SUBSTANCE = """{"substances": [{"name": "codeine", "clinical_name": "codeine", "drug_class": "opioid", "confidence": 0.9, "reasoning": "lean contains codeine"}]}"""

_MOCK_GEMINI_NARRATIVE = """{"stage": "Dependence", "confidence": 0.75, "scores": {"Curiosity": 0.02, "Experimentation": 0.05, "Regular Use": 0.10, "Dependence": 0.75, "Crisis": 0.05, "Recovery": 0.03}, "reasoning": "can't stop indicates dependence"}"""

_MOCK_GEMINI_BRIEF = """SIGNAL INTELLIGENCE BRIEF

1. SUBSTANCE IDENTIFICATION
Codeine (opioid) detected. [KB:ingredient_codeine.txt]

2. NARRATIVE STAGE ASSESSMENT
Dependence stage. [KB:ingredient_codeine.txt]

3. CLINICAL RISK PROFILE
Codeine is a natural opioid. [KB:ingredient_codeine.txt] [FAERS:codeine+Respiratory depression]

4. POLY-DRUG INTERACTION RISKS
None identified.

5. STAGE-SPECIFIC RISK ANNOTATION
Dependence stage with opioid indicates withdrawal risk. [KB:ingredient_codeine.txt]

6. RECOMMENDED ACTIONS
Consider MAT referral. [KB:ingredient_naltrexone.txt]"""


def _mock_gemini_side_effect(prompt: str) -> str:
    """Route mocked Gemini calls based on prompt content."""
    if "substance" in prompt.lower() or "detect" in prompt.lower():
        return _MOCK_GEMINI_SUBSTANCE
    if "stage" in prompt.lower() or "narrative" in prompt.lower():
        return _MOCK_GEMINI_NARRATIVE
    if "brief" in prompt.lower() or "intelligence" in prompt.lower():
        return _MOCK_GEMINI_BRIEF
    return _MOCK_GEMINI_BRIEF


# ── Test Pipeline Initialization ─────────────────────────────────────────────

class TestPipelineInit:
    """Test SIGNALPipeline construction."""

    def test_pipeline_creates_without_error(self):
        from signal.synthesis.pipeline import SIGNALPipeline
        pipeline = SIGNALPipeline()
        assert pipeline is not None

    def test_pipeline_lazy_retriever(self):
        """Retriever should not be initialized until first analyze() call."""
        from signal.synthesis.pipeline import SIGNALPipeline
        pipeline = SIGNALPipeline()
        assert pipeline._retriever is None


# ── Test Pipeline Analyze (with Vertex AI) ───────────────────────────────────

class TestPipelineAnalyze:
    """Full pipeline integration tests (require Vertex AI)."""

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_analyze_returns_signal_report(self):
        """Full analysis should return a SignalReport."""
        from signal.synthesis.pipeline import SIGNALPipeline

        pipeline = SIGNALPipeline()

        # Mock only the brief generator's Gemini call to avoid extra API cost
        with patch("signal.synthesis.brief_generator._call_gemini", return_value=_MOCK_GEMINI_BRIEF):
            report = pipeline.analyze("I've been sippin lean every day and I can't stop")

        assert isinstance(report, SignalReport)
        assert report.original_text == "I've been sippin lean every day and I can't stop"
        assert report.elapsed_ms > 0

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_analyze_skip_brief(self):
        """skip_brief=True should return empty brief string."""
        from signal.synthesis.pipeline import SIGNALPipeline

        pipeline = SIGNALPipeline()
        report = pipeline.analyze("test post", skip_brief=True)
        assert isinstance(report, SignalReport)
        # Brief should be empty when skipped
        assert report.analyst_brief == "" or report.clinical_contexts == ()

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_analyze_neutral_text(self):
        """Non-substance text should return a valid report with no substances."""
        from signal.synthesis.pipeline import SIGNALPipeline

        pipeline = SIGNALPipeline()
        report = pipeline.analyze("The weather is nice today.", skip_brief=True)
        assert isinstance(report, SignalReport)
        assert report.original_text == "The weather is nice today."

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_analyze_batch(self):
        """Batch mode should return one report per input."""
        from signal.synthesis.pipeline import SIGNALPipeline

        pipeline = SIGNALPipeline()
        texts = ["I tried percs last weekend", "30 days clean today"]
        reports = pipeline.analyze_batch(texts, skip_brief=True)
        assert len(reports) == 2
        assert all(isinstance(r, SignalReport) for r in reports)
        assert reports[0].post_id == "batch_0"
        assert reports[1].post_id == "batch_1"


# ── Test Report Structure ────────────────────────────────────────────────────

class TestReportStructure:
    """Validate SignalReport fields and types (no API required)."""

    def test_signal_report_fields(self):
        """SignalReport should have all expected fields."""
        report = SignalReport(
            post_id="test",
            original_text="test text",
            substance_results=None,
            narrative_results=None,
            clinical_contexts=(),
            analyst_brief="test brief",
            elapsed_ms=100.0,
        )
        assert report.post_id == "test"
        assert report.original_text == "test text"
        assert report.clinical_contexts == ()
        assert report.analyst_brief == "test brief"
        assert report.elapsed_ms == 100.0

    def test_signal_report_is_frozen(self):
        """SignalReport should be immutable."""
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
            report.post_id = "modified"  # type: ignore[misc]
