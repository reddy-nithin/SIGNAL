"""
SIGNAL — Clinical Contextualizer (Layer 3)
============================================
For each resolved substance from Layer 1, retrieves relevant knowledge chunks
via FAISS/BM25 hybrid search, looks up FAERS safety signals, checks poly-drug
interactions, and annotates stage-specific risks.

Public API:
    build_clinical_context()  — single substance
    contextualize_all()       — all substances from an EnsembleResult
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from signal.config import (
    FAERS_SIGNAL_PATH,
    SUPPLEMENTARY_SIGNALS_PATH,
)
from signal.grounding.indexer import HybridRetriever, RetrievalResult
from signal.grounding.types import (
    ClinicalContext,
    FAERSSignal,
    InteractionWarning,
    RetrievedEvidence,
)
from signal.substance.types import EnsembleResult

logger = logging.getLogger(__name__)

# ── FAERS / Supplementary Signal Loading ─────────────────────────────────────

_faers_cache: list[dict] | None = None
_supp_cache: list[dict] | None = None


def _load_faers_signals(path: Path = FAERS_SIGNAL_PATH) -> list[dict]:
    """Load FAERS signals from disk (cached after first call)."""
    global _faers_cache
    if _faers_cache is not None:
        return _faers_cache
    if not path.exists():
        logger.warning("FAERS signal file not found: %s", path)
        _faers_cache = []
        return _faers_cache
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _faers_cache = data.get("signals", [])
    return _faers_cache


def _load_supplementary_signals(path: Path = SUPPLEMENTARY_SIGNALS_PATH) -> list[dict]:
    """Load literature-curated supplementary signals (cached)."""
    global _supp_cache
    if _supp_cache is not None:
        return _supp_cache
    if not path.exists():
        logger.warning("Supplementary signals file not found: %s", path)
        _supp_cache = []
        return _supp_cache
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _supp_cache = data.get("signals", [])
    return _supp_cache


def lookup_faers_signals(drug_name: str) -> tuple[FAERSSignal, ...]:
    """Find all FAERS + supplementary signals matching a drug name."""
    drug_lower = drug_name.lower()
    results: list[FAERSSignal] = []

    for sig in _load_faers_signals():
        if sig.get("drug_name", "").lower() == drug_lower:
            prr_val = sig.get("prr", {})
            ror_val = sig.get("ror", {})
            results.append(FAERSSignal(
                drug_name=sig["drug_name"],
                reaction=sig["reaction"],
                prr=prr_val.get("value") if isinstance(prr_val, dict) else None,
                ror=ror_val.get("value") if isinstance(ror_val, dict) else None,
                source="faers",
            ))

    for sig in _load_supplementary_signals():
        if sig.get("drug_name", "").lower() == drug_lower:
            prr_val = sig.get("prr", {})
            ror_val = sig.get("ror", {})
            results.append(FAERSSignal(
                drug_name=sig["drug_name"],
                reaction=sig["reaction"],
                prr=prr_val.get("value") if isinstance(prr_val, dict) else None,
                ror=ror_val.get("value") if isinstance(ror_val, dict) else None,
                source="literature_curated",
            ))

    return tuple(results)


# ── Interaction Detection ────────────────────────────────────────────────────

# Known high-risk poly-drug combinations (substance pair → query term)
_INTERACTION_PAIRS: dict[frozenset[str], str] = {
    frozenset({"opioid", "alcohol"}): "opioid alcohol respiratory depression interaction",
    frozenset({"opioid", "benzo"}): "opioid benzodiazepine respiratory depression interaction",
    frozenset({"alcohol", "benzo"}): "alcohol benzodiazepine GABA interaction",
    frozenset({"opioid", "stimulant"}): "speedball opioid stimulant interaction",
    frozenset({"stimulant", "alcohol"}): "stimulant alcohol cocaethylene interaction",
}


def detect_interactions(
    substance_classes: list[str],
    retriever: HybridRetriever,
) -> tuple[InteractionWarning, ...]:
    """Check for poly-drug interaction warnings between detected substance classes."""
    unique_classes = set(c.lower() for c in substance_classes)
    warnings: list[InteractionWarning] = []

    for pair, query in _INTERACTION_PAIRS.items():
        if pair.issubset(unique_classes):
            hits = retriever.query(query, top_k=2)
            for hit in hits:
                if hit.score > 0.3:
                    warnings.append(InteractionWarning(
                        substances=tuple(sorted(pair)),
                        risk_description=hit.chunk.text[:300],
                        source_chunk=hit.chunk.filename,
                    ))
                    break  # one warning per pair

    return tuple(warnings)


# ── Stage-Specific Risk Annotation ───────────────────────────────────────────

_STAGE_RISK_NOTES: dict[str, dict[str, str]] = {
    "Crisis": {
        "opioid": "CRITICAL: Crisis stage + opioid indicates active overdose risk. Naloxone availability should be assessed.",
        "alcohol": "CRITICAL: Crisis stage + alcohol indicates risk of acute alcohol poisoning or delirium tremens.",
        "benzo": "CRITICAL: Crisis stage + benzodiazepine indicates overdose risk, especially with concurrent CNS depressants.",
        "stimulant": "CRITICAL: Crisis stage + stimulant indicates risk of cardiac emergency, stroke, or psychosis.",
    },
    "Dependence": {
        "opioid": "WARNING: Dependence stage + opioid indicates withdrawal risk if supply interrupted. MAT referral indicated.",
        "alcohol": "WARNING: Dependence stage + alcohol indicates potentially fatal withdrawal risk. Medical detox recommended.",
        "benzo": "WARNING: Dependence stage + benzodiazepine indicates seizure risk on abrupt cessation. Supervised taper needed.",
        "stimulant": "NOTE: Dependence stage + stimulant. No fatal withdrawal, but severe dysphoria and relapse risk.",
    },
}


# ── Main API ─────────────────────────────────────────────────────────────────

def build_clinical_context(
    substance_name: str,
    drug_class: str,
    narrative_stage: str,
    retriever: HybridRetriever,
    top_k: int = 5,
) -> ClinicalContext:
    """Build clinical grounding for a single substance.

    Args:
        substance_name: Clinical name (e.g., "fentanyl", "alprazolam").
        drug_class: Drug class label (e.g., "opioid", "benzo").
        narrative_stage: Detected narrative stage (e.g., "Crisis").
        retriever: Initialized HybridRetriever.
        top_k: Number of knowledge chunks to retrieve.

    Returns:
        ClinicalContext with evidence, signals, and stage annotation.
    """
    # Retrieve relevant knowledge chunks
    query = f"{substance_name} {drug_class}"
    hits: list[RetrievalResult] = retriever.query(query, top_k=top_k)

    evidence = tuple(
        RetrievedEvidence(
            chunk_filename=h.chunk.filename,
            chunk_type=h.chunk.chunk_type,
            drug_name=h.chunk.drug_name,
            relevance_score=h.score,
            text_snippet=h.chunk.text[:500],
        )
        for h in hits
    )

    # FAERS signal lookup
    signals = lookup_faers_signals(substance_name)

    return ClinicalContext(
        substance=substance_name,
        drug_class=drug_class,
        evidence=evidence,
        faers_signals=signals,
        interactions=(),  # filled by contextualize_all
        narrative_stage=narrative_stage,
    )


def contextualize_all(
    ensemble_result: EnsembleResult,
    narrative_stage: str,
    retriever: HybridRetriever,
    top_k: int = 5,
) -> tuple[ClinicalContext, ...]:
    """Build clinical contexts for all substances detected in a post.

    Also runs poly-drug interaction detection across all substance classes.

    Args:
        ensemble_result: Output from substance ensemble (Layer 1).
        narrative_stage: Top-predicted narrative stage name (Layer 2).
        retriever: Initialized HybridRetriever.
        top_k: Chunks per substance.

    Returns:
        Tuple of ClinicalContext, one per unique detected substance.
    """
    # Deduplicate by clinical_name (ensemble may have multiple matches for same substance)
    seen: set[str] = set()
    substance_pairs: list[tuple[str, str]] = []
    for match in ensemble_result.matches:
        if match.clinical_name not in seen and not match.is_negated:
            seen.add(match.clinical_name)
            substance_pairs.append((match.clinical_name, match.drug_class))

    if not substance_pairs:
        return ()

    # Build per-substance contexts
    contexts: list[ClinicalContext] = []
    for name, drug_class in substance_pairs:
        ctx = build_clinical_context(name, drug_class, narrative_stage, retriever, top_k)
        contexts.append(ctx)

    # Detect poly-drug interactions
    all_classes = [dc for _, dc in substance_pairs]
    interactions = detect_interactions(all_classes, retriever)

    # Attach interactions to all contexts (they're cross-substance warnings)
    if interactions:
        contexts = [
            ClinicalContext(
                substance=ctx.substance,
                drug_class=ctx.drug_class,
                evidence=ctx.evidence,
                faers_signals=ctx.faers_signals,
                interactions=interactions,
                narrative_stage=ctx.narrative_stage,
            )
            for ctx in contexts
        ]

    return tuple(contexts)
