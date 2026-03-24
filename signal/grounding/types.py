"""
Clinical grounding type definitions.
======================================
Frozen dataclasses shared by clinical_contextualizer and brief_generator.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedEvidence:
    """A single knowledge chunk retrieved for a substance."""
    chunk_filename: str
    chunk_type: str
    drug_name: str | None
    relevance_score: float
    text_snippet: str  # first ~500 chars of the chunk


@dataclass(frozen=True)
class FAERSSignal:
    """A single adverse event signal for a substance."""
    drug_name: str
    reaction: str
    prr: float | None
    ror: float | None
    source: str  # "faers" | "literature_curated"


@dataclass(frozen=True)
class InteractionWarning:
    """A poly-drug interaction risk identified from knowledge chunks."""
    substances: tuple[str, ...]
    risk_description: str
    source_chunk: str


@dataclass(frozen=True)
class ClinicalContext:
    """Full clinical grounding for a single substance in a post."""
    substance: str
    drug_class: str
    evidence: tuple[RetrievedEvidence, ...]
    faers_signals: tuple[FAERSSignal, ...]
    interactions: tuple[InteractionWarning, ...]
    narrative_stage: str  # for stage-specific risk annotation


@dataclass(frozen=True)
class SignalReport:
    """Complete 4-layer SIGNAL analysis output for a single post."""
    post_id: str
    original_text: str
    substance_results: object  # EnsembleResult from signal.substance.types
    narrative_results: object  # NarrativeEnsembleResult from signal.narrative.types
    clinical_contexts: tuple[ClinicalContext, ...]
    analyst_brief: str  # Gemini-generated intelligence brief
    elapsed_ms: float
