"""
Ensemble narrative stage classifier.
======================================
Weighted voting across rule-based, fine-tuned DistilBERT, and LLM classifiers.
Produces fused results + per-method comparison + agreement statistics.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from signal.config import (
    NARRATIVE_ENSEMBLE_WEIGHTS,
    NARRATIVE_ENSEMBLE_THRESHOLD,
    STAGE_NAMES,
    STAGE_COUNT,
)
from signal.ingestion.post_ingester import Post
from signal.narrative.types import (
    StageClassification,
    ClassificationResult,
    NarrativeEnsembleResult,
)
from signal.narrative import rule_based_classifier
from signal.narrative import fine_tuned_classifier
from signal.narrative import llm_classifier

logger = logging.getLogger(__name__)


# ── Fusion ───────────────────────────────────────────────────────────────────

def _fuse_classifications(
    method_results: tuple[ClassificationResult, ...],
    weights: dict[str, float],
) -> tuple[StageClassification, tuple[StageClassification, ...]]:
    """Fuse per-stage scores from multiple classifiers via weighted sum.

    Returns (top_stage, all_stages) where all_stages is sorted by stage_index.
    """
    # Accumulate weighted scores per stage
    stage_scores = np.zeros(STAGE_COUNT)
    for result in method_results:
        w = weights.get(result.method, 0.0)
        for sc in result.all_stages:
            stage_scores[sc.stage_index] += w * sc.confidence

    # Normalize to sum to 1
    total = stage_scores.sum()
    if total > 0:
        stage_scores = stage_scores / total
    else:
        stage_scores = np.ones(STAGE_COUNT) / STAGE_COUNT

    # Build reasoning from method top picks
    method_picks = [f"{r.method}={r.top_stage.stage}" for r in method_results]
    reasoning = "Ensemble: " + ", ".join(method_picks)

    all_stages = tuple(
        StageClassification(
            stage=STAGE_NAMES[i],
            stage_index=i,
            confidence=round(float(stage_scores[i]), 4),
            method="ensemble",
            reasoning=reasoning,
        )
        for i in range(STAGE_COUNT)
    )

    top_idx = int(np.argmax(stage_scores))
    return all_stages[top_idx], all_stages


def _count_agreement(method_results: tuple[ClassificationResult, ...]) -> int:
    """Count how many methods agree on the top stage."""
    if not method_results:
        return 0
    top_stages = [r.top_stage.stage for r in method_results]
    most_common = max(set(top_stages), key=top_stages.count)
    return top_stages.count(most_common)


def _redistribute_weights(
    weights: dict[str, float],
    exclude: str,
) -> dict[str, float]:
    """Redistribute an excluded method's weight proportionally to remaining methods."""
    excluded_w = weights.get(exclude, 0.0)
    remaining = {k: v for k, v in weights.items() if k != exclude}
    total_remaining = sum(remaining.values())
    if total_remaining == 0:
        return remaining
    factor = (total_remaining + excluded_w) / total_remaining
    return {k: round(v * factor, 4) for k, v in remaining.items()}


# ── Classification ───────────────────────────────────────────────────────────

def classify(
    post: Post,
    weights: dict[str, float] | None = None,
) -> NarrativeEnsembleResult:
    """Run all classifiers and fuse results.

    Skips DistilBERT if no trained checkpoint is available.
    """
    if weights is None:
        weights = dict(NARRATIVE_ENSEMBLE_WEIGHTS)

    results: list[ClassificationResult] = []

    # Rule-based (always available)
    results.append(rule_based_classifier.classify(post))

    # Fine-tuned DistilBERT (skip if unavailable)
    if fine_tuned_classifier.is_model_available():
        results.append(fine_tuned_classifier.classify(post))
    else:
        logger.info("DistilBERT not available — redistributing weight")
        weights = _redistribute_weights(weights, "fine_tuned")

    # LLM (Gemini few-shot)
    results.append(llm_classifier.classify(post))

    method_results = tuple(results)
    top_stage, all_stages = _fuse_classifications(method_results, weights)
    agreement = _count_agreement(method_results)

    return NarrativeEnsembleResult(
        post_id=post.id,
        top_stage=top_stage,
        all_stages=all_stages,
        method_results=method_results,
        agreement_count=agreement,
    )


def classify_from_results(
    post_id: str,
    method_results: tuple[ClassificationResult, ...],
    weights: dict[str, float] | None = None,
) -> NarrativeEnsembleResult:
    """Fuse pre-computed classifier results (for testing without API calls)."""
    if weights is None:
        weights = dict(NARRATIVE_ENSEMBLE_WEIGHTS)
    top_stage, all_stages = _fuse_classifications(method_results, weights)
    agreement = _count_agreement(method_results)
    return NarrativeEnsembleResult(
        post_id=post_id,
        top_stage=top_stage,
        all_stages=all_stages,
        method_results=method_results,
        agreement_count=agreement,
    )


def classify_batch(
    posts: list[Post],
    weights: dict[str, float] | None = None,
) -> list[NarrativeEnsembleResult]:
    """Run ensemble classification on a batch of posts."""
    return [classify(p, weights=weights) for p in posts]


# ── Comparison Table ─────────────────────────────────────────────────────────

def build_comparison_table(result: NarrativeEnsembleResult) -> list[dict]:
    """Build a stage-by-method table for the dashboard.

    Returns list of dicts: {"stage", "method", "confidence", "is_top"}
    """
    rows: list[dict] = []
    for mr in result.method_results:
        for sc in mr.all_stages:
            rows.append({
                "stage": sc.stage,
                "method": mr.method,
                "confidence": sc.confidence,
                "is_top": sc.stage == mr.top_stage.stage,
            })
    # Ensemble row
    for sc in result.all_stages:
        rows.append({
            "stage": sc.stage,
            "method": "ensemble",
            "confidence": sc.confidence,
            "is_top": sc.stage == result.top_stage.stage,
        })
    return rows


# ── Agreement Statistics ─────────────────────────────────────────────────────

def _cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's kappa for two annotators on categorical labels."""
    n = len(labels_a)
    if n == 0:
        return 0.0

    k = max(max(labels_a, default=0), max(labels_b, default=0)) + 1
    confusion = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        confusion[a][b] += 1

    po = np.trace(confusion) / n  # observed agreement
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums)) / (n * n)  # expected agreement

    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def _fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Fleiss' kappa for multiple raters.

    Args:
        ratings_matrix: shape (n_subjects, n_categories) — count of raters per category per subject.
    """
    n_subjects, n_cats = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())

    if n_subjects == 0 or n_raters <= 1:
        return 0.0

    # P_i for each subject
    p_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))

    # P_j for each category (proportion of all assignments to category j)
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    pe_bar = float(np.sum(p_j ** 2))

    if pe_bar >= 1.0:
        return 1.0
    return (p_bar - pe_bar) / (1.0 - pe_bar)


def compute_agreement_stats(results: list[NarrativeEnsembleResult]) -> dict:
    """Compute inter-method agreement statistics across a corpus.

    Returns dict with:
        - pairwise_kappa: Cohen's kappa between each method pair
        - fleiss_kappa: Fleiss' kappa across all methods
        - pairwise_agreement: fraction of posts where each method pair agrees
        - all_agree_pct: fraction of posts where all methods agree
    """
    n = len(results)
    if n == 0:
        return {
            "pairwise_kappa": {},
            "fleiss_kappa": 0.0,
            "pairwise_agreement": {},
            "all_agree_pct": 0.0,
        }

    # Extract per-method top stage labels
    method_labels: dict[str, list[int]] = {}
    for result in results:
        for mr in result.method_results:
            method_labels.setdefault(mr.method, []).append(mr.top_stage.stage_index)

    methods = sorted(method_labels.keys())

    # Pairwise Cohen's kappa and agreement
    pairwise_kappa: dict[str, float] = {}
    pairwise_agreement: dict[str, float] = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            key = f"{a}_vs_{b}"
            labels_a = method_labels[a]
            labels_b = method_labels[b]
            min_len = min(len(labels_a), len(labels_b))
            pairwise_kappa[key] = round(_cohens_kappa(labels_a[:min_len], labels_b[:min_len]), 4)
            agree = sum(la == lb for la, lb in zip(labels_a[:min_len], labels_b[:min_len]))
            pairwise_agreement[key] = round(agree / max(min_len, 1), 4)

    # All-agree
    all_agree = 0
    for result in results:
        stages = [mr.top_stage.stage for mr in result.method_results]
        if len(set(stages)) == 1:
            all_agree += 1

    # Fleiss' kappa — build ratings matrix
    n_methods = len(methods)
    if n_methods >= 2:
        ratings = np.zeros((n, STAGE_COUNT), dtype=int)
        for idx, result in enumerate(results):
            for mr in result.method_results:
                ratings[idx, mr.top_stage.stage_index] += 1
        fleiss = round(_fleiss_kappa(ratings), 4)
    else:
        fleiss = 0.0

    return {
        "pairwise_kappa": pairwise_kappa,
        "fleiss_kappa": fleiss,
        "pairwise_agreement": pairwise_agreement,
        "all_agree_pct": round(all_agree / n, 4),
    }
