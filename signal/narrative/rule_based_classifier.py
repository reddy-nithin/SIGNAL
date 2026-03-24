"""
Rule-based narrative stage classifier.
========================================
Keywords + tense analysis + hedging detection + urgency scoring.
Deterministic, no API calls required.
"""
from __future__ import annotations

import re
import time
import logging
from pathlib import Path

import numpy as np

from signal.config import NARRATIVE_STAGES, STAGE_NAMES, STAGE_COUNT, MODELS_DIR
from signal.ingestion.post_ingester import Post
from signal.narrative.types import StageClassification, ClassificationResult
from signal.narrative.stage_exemplars import STAGE_KEYWORDS

logger = logging.getLogger(__name__)

# ── Tense Detection ──────────────────────────────────────────────────────────

_PAST_MARKERS = re.compile(
    r"\b(was|were|did|used to|had|went|took|back when|years ago|months ago|"
    r"last year|last month|last week|ended up|wound up|got caught)\b",
    re.IGNORECASE,
)
_PRESENT_MARKERS = re.compile(
    r"\b(am|is|are|currently|right now|every day|every night|nowadays|"
    r"at the moment|these days|still|keep|keeps|always)\b",
    re.IGNORECASE,
)
_FUTURE_MARKERS = re.compile(
    r"\b(will|going to|gonna|planning to|thinking about|want to try|"
    r"considering|might try|should i|what if i)\b",
    re.IGNORECASE,
)

# Tense → stage bonus matrix: tense_bonuses[tense][stage_index]
_TENSE_BONUSES: dict[str, tuple[float, ...]] = {
    #             Cur   Exp   Reg   Dep   Cri   Rec
    "future":  ( 0.15, 0.05, 0.00, 0.00, 0.00, 0.00),
    "past":    ( 0.05, 0.10, 0.00, 0.00,-0.05, 0.15),
    "present": ( 0.00, 0.00, 0.10, 0.10, 0.10, 0.05),
    "mixed":   ( 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
}


def _detect_tense(text: str) -> str:
    """Detect dominant tense: 'past', 'present', 'future', or 'mixed'."""
    past = len(_PAST_MARKERS.findall(text))
    present = len(_PRESENT_MARKERS.findall(text))
    future = len(_FUTURE_MARKERS.findall(text))

    total = past + present + future
    if total == 0:
        return "mixed"

    scores = {"past": past, "present": present, "future": future}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] / total >= 0.5:
        return best
    return "mixed"


# ── Hedging Detection ────────────────────────────────────────────────────────

_HEDGING_PHRASES = re.compile(
    r"\b(maybe|i think|not sure|i guess|might|probably|perhaps|"
    r"i wonder|could be|kind of|sort of|i don'?t know|supposedly)\b",
    re.IGNORECASE,
)


def _detect_hedging(text: str) -> float:
    """Return hedging score 0.0-1.0. High hedging → Curiosity/Experimentation."""
    matches = len(_HEDGING_PHRASES.findall(text))
    word_count = max(len(text.split()), 1)
    # Normalize: 3+ hedging phrases in a short post = high hedging
    return min(matches / max(word_count / 20, 1), 1.0)


# ── Urgency Detection ────────────────────────────────────────────────────────

_URGENCY_WORDS = re.compile(
    r"\b(help|please|dying|emergency|desperate|terrified|can'?t breathe|"
    r"call 911|save me|i'?m going to die|need help|hospitalized|ambulance|"
    r"od'?d|overdose|seizure|someone help)\b",
    re.IGNORECASE,
)


def _compute_urgency(text: str) -> float:
    """Return urgency score 0.0-1.0. High urgency → Crisis/Dependence."""
    word_matches = len(_URGENCY_WORDS.findall(text))
    exclamations = text.count("!")
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    raw = word_matches * 2 + exclamations + caps_words * 0.5
    return min(raw / 8.0, 1.0)


# ── Keyword Scoring ──────────────────────────────────────────────────────────

def _keyword_scores(text: str) -> dict[str, float]:
    """Compute per-stage keyword match scores (0.0-1.0)."""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for stage, keywords in STAGE_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[stage] = min(matches / len(keywords) * 3, 1.0)
        else:
            scores[stage] = 0.0
    return scores


# ── Centroid Scoring (optional) ──────────────────────────────────────────────

_centroids_cache: np.ndarray | None = None
_centroids_loaded: bool = False

CENTROIDS_PATH = MODELS_DIR / "stage_centroids.npy"


def _load_centroids() -> np.ndarray | None:
    """Load stage centroids if available. Returns None on failure."""
    global _centroids_cache, _centroids_loaded
    if _centroids_loaded:
        return _centroids_cache
    _centroids_loaded = True
    if CENTROIDS_PATH.exists():
        try:
            _centroids_cache = np.load(str(CENTROIDS_PATH))
            logger.info("Loaded stage centroids from %s", CENTROIDS_PATH)
        except Exception as e:
            logger.warning("Failed to load centroids: %s", e)
    return _centroids_cache


def _centroid_scores(text: str) -> dict[str, float] | None:
    """Compute cosine similarity to each stage centroid. Returns None if unavailable."""
    centroids = _load_centroids()
    if centroids is None:
        return None

    try:
        from signal.grounding.indexer import embed_query
        vec, _ = embed_query(text)
        # vec shape (1, dim), centroids shape (6, dim)
        # L2-normalize for cosine similarity
        vec_norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
        cent_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        sims = (vec_norm @ cent_norm.T).flatten()  # shape (6,)
        # Convert to 0-1 range (cosine sim can be negative)
        sims = np.clip((sims + 1) / 2, 0, 1)
        return {STAGE_NAMES[i]: float(sims[i]) for i in range(STAGE_COUNT)}
    except Exception as e:
        logger.warning("Centroid scoring failed: %s", e)
        return None


# ── Main Classifier ──────────────────────────────────────────────────────────

def classify(post: Post) -> ClassificationResult:
    """Classify a post into a narrative stage using rule-based signals.

    Combines: keywords (0.50) + tense (0.15) + hedging (0.15) + urgency (0.10) + centroids (0.10).
    If centroids unavailable, keyword weight increases to 0.60.
    """
    t0 = time.perf_counter()

    kw = _keyword_scores(post.text)
    tense = _detect_tense(post.text)
    hedging = _detect_hedging(post.text)
    urgency = _compute_urgency(post.text)
    centroids = _centroid_scores(post.text)

    # Weight allocation
    if centroids is not None:
        w_kw, w_tense, w_hedge, w_urg, w_cent = 0.50, 0.15, 0.15, 0.10, 0.10
    else:
        w_kw, w_tense, w_hedge, w_urg, w_cent = 0.60, 0.15, 0.15, 0.10, 0.00

    tense_bonus = _TENSE_BONUSES.get(tense, _TENSE_BONUSES["mixed"])

    # Hedging bonuses
    hedge_bonus = [0.0] * STAGE_COUNT
    if hedging > 0.5:
        hedge_bonus[0] = 0.10  # Curiosity
        hedge_bonus[1] = 0.05  # Experimentation
    elif hedging < 0.2:
        hedge_bonus[3] = 0.05  # Dependence
        hedge_bonus[4] = 0.05  # Crisis

    # Urgency bonuses
    urg_bonus = [0.0] * STAGE_COUNT
    if urgency > 0.7:
        urg_bonus[4] = 0.15  # Crisis
        urg_bonus[3] = 0.05  # Dependence
    elif urgency < 0.3:
        urg_bonus[0] = 0.05  # Curiosity
        urg_bonus[2] = 0.05  # Regular Use

    # Combine scores for each stage
    stage_scores: list[float] = []
    for i, name in enumerate(STAGE_NAMES):
        score = (
            w_kw * kw.get(name, 0.0)
            + w_tense * tense_bonus[i]
            + w_hedge * hedge_bonus[i]
            + w_urg * urg_bonus[i]
        )
        if centroids is not None:
            score += w_cent * centroids.get(name, 0.0)
        stage_scores.append(max(score, 0.0))

    # Normalize to sum to 1 (if any positive)
    total = sum(stage_scores)
    if total > 0:
        stage_scores = [s / total for s in stage_scores]
    else:
        stage_scores = [1.0 / STAGE_COUNT] * STAGE_COUNT

    # Build reasoning
    parts = [f"tense={tense}"]
    if hedging > 0.3:
        parts.append(f"hedging={hedging:.2f}")
    if urgency > 0.3:
        parts.append(f"urgency={urgency:.2f}")
    top_kw = max(kw, key=kw.get) if any(v > 0 for v in kw.values()) else "none"  # type: ignore[arg-type]
    parts.append(f"top_keyword_stage={top_kw}")

    elapsed = (time.perf_counter() - t0) * 1000

    all_stages = tuple(
        StageClassification(
            stage=STAGE_NAMES[i],
            stage_index=i,
            confidence=round(stage_scores[i], 4),
            method="rule_based",
            reasoning="; ".join(parts),
        )
        for i in range(STAGE_COUNT)
    )

    top_idx = int(np.argmax(stage_scores))
    return ClassificationResult(
        post_id=post.id,
        top_stage=all_stages[top_idx],
        all_stages=all_stages,
        method="rule_based",
        elapsed_ms=round(elapsed, 2),
    )


def classify_batch(posts: list[Post]) -> list[ClassificationResult]:
    """Classify a batch of posts."""
    return [classify(p) for p in posts]
