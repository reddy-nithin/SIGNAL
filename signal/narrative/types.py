"""
Narrative stage classification type definitions.
=================================================
Frozen dataclasses shared by all classifier modules.
Single-label multi-class: one stage per post, scores for all 6.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageClassification:
    """A single stage score within a classification result."""
    stage: str          # "Curiosity", "Experimentation", etc.
    stage_index: int    # 0-5
    confidence: float   # 0.0-1.0
    method: str         # "rule_based" | "fine_tuned" | "llm" | "ensemble"
    reasoning: str      # brief explanation


@dataclass(frozen=True)
class ClassificationResult:
    """Output of a single classifier on a single post."""
    post_id: str
    top_stage: StageClassification
    all_stages: tuple[StageClassification, ...]  # always 6, sorted by stage_index
    method: str
    elapsed_ms: float


@dataclass(frozen=True)
class NarrativeEnsembleResult:
    """Output of the ensemble voter — fused classification + per-method breakdown."""
    post_id: str
    top_stage: StageClassification
    all_stages: tuple[StageClassification, ...]  # ensemble scores for all 6
    method_results: tuple[ClassificationResult, ...]  # one per method
    agreement_count: int  # how many methods agree on top stage
