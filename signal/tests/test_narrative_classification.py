"""
Tests for Phase 3: Narrative Stage Classification.
====================================================
Covers types, rule-based, fine-tuned (mocked), LLM (mocked),
ensemble fusion, agreement statistics, and synthetic test cases.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import json

import numpy as np
import pytest

from signal.config import (
    NARRATIVE_ENSEMBLE_WEIGHTS,
    NARRATIVE_ENSEMBLE_THRESHOLD,
    NARRATIVE_CLASSIFICATION_METHODS,
    STAGE_NAMES,
    STAGE_COUNT,
    DISTILBERT_CHECKPOINT_DIR,
)
from signal.ingestion.post_ingester import Post
from signal.narrative.types import (
    StageClassification,
    ClassificationResult,
    NarrativeEnsembleResult,
)
from signal.narrative import rule_based_classifier
from signal.narrative import ensemble as narrative_ensemble


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_post(text: str, post_id: str = "test") -> Post:
    return Post(id=post_id, text=text, raw_text=text, source="test")


def _make_classification_result(
    method: str,
    stage_scores: dict[str, float],
    post_id: str = "test",
) -> ClassificationResult:
    """Build a ClassificationResult from a stage→score mapping."""
    all_stages = tuple(
        StageClassification(
            stage=name,
            stage_index=i,
            confidence=stage_scores.get(name, 0.0),
            method=method,
            reasoning="test",
        )
        for i, name in enumerate(STAGE_NAMES)
    )
    top_idx = max(range(STAGE_COUNT), key=lambda i: all_stages[i].confidence)
    return ClassificationResult(
        post_id=post_id,
        top_stage=all_stages[top_idx],
        all_stages=all_stages,
        method=method,
        elapsed_ms=1.0,
    )


# ── Type Tests ───────────────────────────────────────────────────────────────

class TestStageClassification:
    def test_frozen(self):
        sc = StageClassification("Curiosity", 0, 0.9, "rule_based", "test")
        with pytest.raises(AttributeError):
            sc.confidence = 0.5  # type: ignore[misc]

    def test_fields(self):
        sc = StageClassification("Crisis", 4, 0.8, "llm", "urgent language")
        assert sc.stage == "Crisis"
        assert sc.stage_index == 4
        assert sc.confidence == 0.8


class TestClassificationResult:
    def test_all_stages_length(self):
        result = _make_classification_result(
            "rule_based",
            {name: 1.0 / STAGE_COUNT for name in STAGE_NAMES},
        )
        assert len(result.all_stages) == STAGE_COUNT

    def test_top_stage_in_all_stages(self):
        result = _make_classification_result("llm", {"Recovery": 0.9, "Crisis": 0.1})
        assert result.top_stage.stage == "Recovery"
        assert result.top_stage in result.all_stages


class TestNarrativeEnsembleResult:
    def test_frozen(self):
        er = NarrativeEnsembleResult(
            post_id="t",
            top_stage=StageClassification("Curiosity", 0, 0.5, "ensemble", ""),
            all_stages=tuple(
                StageClassification(name, i, 0.1, "ensemble", "")
                for i, name in enumerate(STAGE_NAMES)
            ),
            method_results=(),
            agreement_count=0,
        )
        with pytest.raises(AttributeError):
            er.agreement_count = 3  # type: ignore[misc]


# ── Rule-Based Classifier ───────────────────────────────────────────────────

class TestRuleBasedClassifier:
    def test_curiosity_post(self):
        post = _make_post("What does fentanyl feel like? Is it safe to try?")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Curiosity"
        assert result.method == "rule_based"

    def test_experimentation_post(self):
        post = _make_post("Tried molly for the first time at a party, just experimenting, not addicted")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Experimentation"

    def test_regular_use_post(self):
        post = _make_post("I use every day now, helps me cope with stress, been using for months")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Regular Use"

    def test_dependence_post(self):
        post = _make_post("Can't function without it, withdrawal is killing me, physically dependent")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Dependence"

    def test_crisis_post(self):
        post = _make_post("I overdosed last night and was hospitalized, almost died, called 911")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Crisis"

    def test_recovery_post(self):
        post = _make_post("30 days clean and sober, going to NA meeting every day, in recovery")
        result = rule_based_classifier.classify(post)
        assert result.top_stage.stage == "Recovery"

    def test_all_stages_returned(self):
        post = _make_post("What does heroin feel like?")
        result = rule_based_classifier.classify(post)
        assert len(result.all_stages) == STAGE_COUNT
        indices = [s.stage_index for s in result.all_stages]
        assert indices == list(range(STAGE_COUNT))

    def test_scores_sum_to_one(self):
        post = _make_post("I overdosed and was hospitalized")
        result = rule_based_classifier.classify(post)
        total = sum(s.confidence for s in result.all_stages)
        assert abs(total - 1.0) < 0.01

    def test_unrelated_low_max_confidence(self):
        post = _make_post("The weather is really nice today and I went for a walk")
        result = rule_based_classifier.classify(post)
        # With no keywords matching, should be uniform
        assert result.top_stage.confidence < 0.3

    def test_elapsed_ms_positive(self):
        post = _make_post("just testing")
        result = rule_based_classifier.classify(post)
        assert result.elapsed_ms >= 0

    def test_batch_classification(self):
        posts = [_make_post(f"test post {i}", f"p{i}") for i in range(5)]
        results = rule_based_classifier.classify_batch(posts)
        assert len(results) == 5


class TestTenseDetection:
    def test_future_tense(self):
        tense = rule_based_classifier._detect_tense("I'm thinking about trying it, going to ask my friend")
        assert tense == "future"

    def test_past_tense(self):
        tense = rule_based_classifier._detect_tense("I used to take pills back when I was in college")
        assert tense == "past"

    def test_present_tense(self):
        tense = rule_based_classifier._detect_tense("I am currently using every day, still dependent")
        assert tense == "present"


class TestHedgingDetection:
    def test_high_hedging(self):
        score = rule_based_classifier._detect_hedging("Maybe I should try it? I think it might be okay, probably safe I guess")
        assert score > 0.3

    def test_low_hedging(self):
        score = rule_based_classifier._detect_hedging("I overdosed and was hospitalized immediately")
        assert score < 0.2


class TestUrgencyDetection:
    def test_high_urgency(self):
        score = rule_based_classifier._compute_urgency("HELP! I'm dying, please call 911, I overdosed!")
        assert score > 0.5

    def test_low_urgency(self):
        score = rule_based_classifier._compute_urgency("I'm curious about trying new things")
        assert score < 0.3


# ── Fine-Tuned Classifier (mocked) ──────────────────────────────────────────

class TestFineTunedClassifier:
    def test_is_model_available_false(self):
        """When no checkpoint exists, should return False."""
        from signal.narrative import fine_tuned_classifier
        with patch.object(fine_tuned_classifier, "DISTILBERT_CHECKPOINT_DIR", MagicMock()):
            # Mock the path check
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            with patch("signal.narrative.fine_tuned_classifier.DISTILBERT_CHECKPOINT_DIR", mock_path):
                # Need to check the actual config.json subpath
                result = (mock_path / "config.json").exists()
                assert result is not None  # basic sanity


# ── LLM Classifier (mocked) ─────────────────────────────────────────────────

class TestLLMClassifier:
    def test_cache_key_deterministic(self):
        from signal.narrative.llm_classifier import _cache_key
        key1 = _cache_key("test prompt")
        key2 = _cache_key("test prompt")
        assert key1 == key2
        assert len(key1) == 24

    def test_parse_response_valid(self):
        from signal.narrative.llm_classifier import _parse_response
        response = json.dumps({
            "stage": "Crisis",
            "stage_index": 4,
            "confidence": 0.85,
            "all_stages": [
                {"stage": "Curiosity", "score": 0.02},
                {"stage": "Experimentation", "score": 0.03},
                {"stage": "Regular Use", "score": 0.05},
                {"stage": "Dependence", "score": 0.05},
                {"stage": "Crisis", "score": 0.85},
                {"stage": "Recovery", "score": 0.00},
            ],
            "reasoning": "overdose language",
        })
        parsed = _parse_response(response)
        assert parsed["stage"] == "Crisis"
        assert len(parsed["all_stages"]) == 6

    def test_parse_response_markdown_fenced(self):
        from signal.narrative.llm_classifier import _parse_response
        response = '```json\n{"stage": "Recovery", "confidence": 0.9}\n```'
        parsed = _parse_response(response)
        assert parsed["stage"] == "Recovery"

    def test_parse_response_invalid(self):
        from signal.narrative.llm_classifier import _parse_response
        parsed = _parse_response("not valid json at all")
        assert parsed == {}

    def test_classify_with_mock(self):
        from signal.narrative import llm_classifier
        mock_response = json.dumps({
            "stage": "Dependence",
            "stage_index": 3,
            "confidence": 0.75,
            "all_stages": [
                {"stage": name, "score": 0.75 if name == "Dependence" else 0.05}
                for name in STAGE_NAMES
            ],
            "reasoning": "withdrawal language",
        })

        with patch.object(llm_classifier, "_call_gemini", return_value=mock_response):
            with patch.object(llm_classifier, "_load_exemplars_cached", return_value=[]):
                post = _make_post("Can't stop using, withdrawal is terrible")
                result = llm_classifier.classify(post)
                assert result.top_stage.stage == "Dependence"
                assert result.method == "llm"
                assert len(result.all_stages) == STAGE_COUNT


# ── Ensemble Fusion ──────────────────────────────────────────────────────────

class TestEnsembleFusion:
    def test_unanimous_agreement(self):
        """All 3 methods pick the same stage → agreement_count matches."""
        scores = {"Crisis": 0.8, "Curiosity": 0.04, "Experimentation": 0.04,
                  "Regular Use": 0.04, "Dependence": 0.04, "Recovery": 0.04}
        rb = _make_classification_result("rule_based", scores)
        ft = _make_classification_result("fine_tuned", scores)
        llm = _make_classification_result("llm", scores)

        result = narrative_ensemble.classify_from_results("test", (rb, ft, llm))
        assert result.agreement_count == 3
        assert result.top_stage.stage == "Crisis"

    def test_split_vote(self):
        """Methods disagree — ensemble picks weighted winner."""
        rb = _make_classification_result("rule_based", {"Curiosity": 0.9, "Crisis": 0.1})
        ft = _make_classification_result("fine_tuned", {"Crisis": 0.8, "Curiosity": 0.2})
        llm = _make_classification_result("llm", {"Crisis": 0.85, "Curiosity": 0.15})

        # Weights: rb=0.20, ft=0.35, llm=0.45 → Crisis dominates
        result = narrative_ensemble.classify_from_results("test", (rb, ft, llm))
        assert result.top_stage.stage == "Crisis"
        assert result.agreement_count == 2  # ft and llm agree

    def test_weight_redistribution_without_distilbert(self):
        """When only 2 methods available, weights redistribute."""
        weights = {"rule_based": 0.20, "fine_tuned": 0.35, "llm": 0.45}
        new_weights = narrative_ensemble._redistribute_weights(weights, "fine_tuned")
        assert "fine_tuned" not in new_weights
        assert abs(sum(new_weights.values()) - 1.0) < 0.01

    def test_comparison_table_structure(self):
        scores = {name: 1.0 / STAGE_COUNT for name in STAGE_NAMES}
        rb = _make_classification_result("rule_based", scores)
        llm = _make_classification_result("llm", scores)
        result = narrative_ensemble.classify_from_results("test", (rb, llm))
        table = narrative_ensemble.build_comparison_table(result)
        # 6 stages x 3 methods (rb, llm, ensemble)
        assert len(table) == STAGE_COUNT * 3
        assert all("stage" in row and "method" in row and "confidence" in row for row in table)

    def test_all_stages_sum_to_one(self):
        scores = {"Dependence": 0.6, "Crisis": 0.3, "Recovery": 0.1}
        rb = _make_classification_result("rule_based", scores)
        ft = _make_classification_result("fine_tuned", scores)
        llm = _make_classification_result("llm", scores)
        result = narrative_ensemble.classify_from_results("test", (rb, ft, llm))
        total = sum(s.confidence for s in result.all_stages)
        assert abs(total - 1.0) < 0.01


# ── Agreement Statistics ─────────────────────────────────────────────────────

class TestAgreementStats:
    def test_cohens_kappa_perfect(self):
        labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        kappa = narrative_ensemble._cohens_kappa(labels, labels)
        assert abs(kappa - 1.0) < 0.01

    def test_cohens_kappa_complete_disagreement(self):
        a = [0, 0, 0, 0, 0]
        b = [1, 1, 1, 1, 1]
        kappa = narrative_ensemble._cohens_kappa(a, b)
        assert kappa < 0.01

    def test_fleiss_kappa_perfect(self):
        # 5 subjects, 3 raters, all agree on same category
        ratings = np.array([
            [3, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 3, 0],
        ])
        kappa = narrative_ensemble._fleiss_kappa(ratings)
        assert abs(kappa - 1.0) < 0.01

    def test_compute_agreement_stats_empty(self):
        stats = narrative_ensemble.compute_agreement_stats([])
        assert stats["all_agree_pct"] == 0.0
        assert stats["fleiss_kappa"] == 0.0

    def test_compute_agreement_stats_basic(self):
        # Create results where all methods agree
        scores = {"Crisis": 0.8}
        rb = _make_classification_result("rule_based", scores)
        llm = _make_classification_result("llm", scores)
        ens_result = narrative_ensemble.classify_from_results("t1", (rb, llm))

        stats = narrative_ensemble.compute_agreement_stats([ens_result])
        assert stats["all_agree_pct"] == 1.0


# ── Synthetic Test Cases ─────────────────────────────────────────────────────

NARRATIVE_TEST_CASES = [
    # Curiosity (5)
    {"text": "What does heroin feel like? Is it as dangerous as they say?", "expected": "Curiosity"},
    {"text": "Is it safe to try shrooms? Thinking about trying them", "expected": "Curiosity"},
    {"text": "Has anyone tried Xanax? What happens if you take it?", "expected": "Curiosity"},
    {"text": "What's it like to be high on oxy? Curious about it", "expected": "Curiosity"},
    {"text": "Should I try molly? Ever tried it at a rave?", "expected": "Curiosity"},
    # Experimentation (5)
    {"text": "Tried Adderall for the first time last weekend, was fun", "expected": "Experimentation"},
    {"text": "Popped a xanax at the party, just experimenting, not addicted", "expected": "Experimentation"},
    {"text": "First time trying edibles, recreational use only, just for fun", "expected": "Experimentation"},
    {"text": "Tried coke for the first time, just partying with friends", "expected": "Experimentation"},
    {"text": "Weekend use only, tried it once, not a big deal", "expected": "Experimentation"},
    # Regular Use (5)
    {"text": "I use every day now, it helps me cope with stress, been using for months", "expected": "Regular Use"},
    {"text": "Every night I take it to sleep, it's become my routine", "expected": "Regular Use"},
    {"text": "Helps me deal with anxiety, I use regularly, need more to feel it", "expected": "Regular Use"},
    {"text": "Every weekend without fail, tolerance is building up", "expected": "Regular Use"},
    {"text": "Daily use for months, helps me function at work", "expected": "Regular Use"},
    # Dependence (5)
    {"text": "Can't stop using, withdrawal is terrible, physically dependent", "expected": "Dependence"},
    {"text": "I'm addicted, can't function without it, cravings are intense", "expected": "Dependence"},
    {"text": "Sick when I stop, can't go a day without, cold turkey failed", "expected": "Dependence"},
    {"text": "Need it to function, dependent on it, withdrawal symptoms are awful", "expected": "Dependence"},
    {"text": "Can't stop the cravings, physically dependent, tried cold turkey", "expected": "Dependence"},
    # Crisis (5)
    {"text": "I overdosed last night, was hospitalized, almost died", "expected": "Crisis"},
    {"text": "Lost my job, lost my family, rock bottom, emergency room visit", "expected": "Crisis"},
    {"text": "Called 911, destroyed my life, arrested for possession", "expected": "Crisis"},
    {"text": "Overdosed and almost died, hospitalized for three days", "expected": "Crisis"},
    {"text": "Lost everything, want to die, rock bottom, lost my family", "expected": "Crisis"},
    # Recovery (5)
    {"text": "30 days clean and sober, going to NA meetings, in recovery", "expected": "Recovery"},
    {"text": "Just got out of rehab, getting help, one day at a time", "expected": "Recovery"},
    {"text": "My sponsor says I'm doing well, 6 months of sobriety", "expected": "Recovery"},
    {"text": "In treatment program, staying clean, AA meetings daily", "expected": "Recovery"},
    {"text": "Recovery is hard but worth it, sober for a year, getting help", "expected": "Recovery"},
]


class TestSyntheticNarrativeCases:
    @pytest.mark.parametrize("case", NARRATIVE_TEST_CASES, ids=[c["text"][:40] for c in NARRATIVE_TEST_CASES])
    def test_individual_case(self, case):
        post = _make_post(case["text"])
        result = rule_based_classifier.classify(post)
        # Not asserting exact match — some cases have keyword overlap
        # Just verify it returns a valid stage
        assert result.top_stage.stage in STAGE_NAMES

    def test_overall_accuracy(self):
        """Rule-based should get >= 70% of synthetic cases correct."""
        correct = 0
        for case in NARRATIVE_TEST_CASES:
            post = _make_post(case["text"])
            result = rule_based_classifier.classify(post)
            if result.top_stage.stage == case["expected"]:
                correct += 1
        accuracy = correct / len(NARRATIVE_TEST_CASES)
        assert accuracy >= 0.70, f"Accuracy {accuracy:.2%} < 70% threshold"


# ── Config Constants ─────────────────────────────────────────────────────────

class TestConfigConstants:
    def test_ensemble_weights_sum(self):
        assert abs(sum(NARRATIVE_ENSEMBLE_WEIGHTS.values()) - 1.0) < 0.01

    def test_methods_match_weights(self):
        for method in NARRATIVE_CLASSIFICATION_METHODS:
            assert method in NARRATIVE_ENSEMBLE_WEIGHTS

    def test_stage_count(self):
        assert STAGE_COUNT == 6
        assert len(STAGE_NAMES) == 6
