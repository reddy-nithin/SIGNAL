"""
SIGNAL — Full Analysis Pipeline
=================================
Orchestrates all 4 layers into a single analyze() call:
  Layer 1: Substance Resolution (ensemble)
  Layer 2: Narrative Stage Classification (ensemble)
  Layer 3: Clinical Grounding (retrieval + FAERS)
  Layer 4: Analyst Brief (Gemini synthesis)

Public API:
    SIGNALPipeline  — stateful pipeline with lazy-initialized components
"""
from __future__ import annotations

import logging
import time
import uuid

from signal.grounding.clinical_contextualizer import contextualize_all
from signal.grounding.indexer import HybridRetriever
from signal.grounding.types import ClinicalContext, SignalReport
from signal.ingestion.post_ingester import Post
from signal.narrative.types import NarrativeEnsembleResult
from signal.substance.types import EnsembleResult
from signal.synthesis.brief_generator import generate_brief

logger = logging.getLogger(__name__)


class SIGNALPipeline:
    """Full 4-layer SIGNAL analysis pipeline.

    Components are lazy-initialized on first call to avoid import-time overhead.

    Usage:
        pipeline = SIGNALPipeline()
        report = pipeline.analyze("I've been mixing lean with xans and I can't stop")
    """

    def __init__(self, force_rebuild_index: bool = False) -> None:
        self._retriever: HybridRetriever | None = None
        self._force_rebuild = force_rebuild_index

    def _ensure_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(force_rebuild=self._force_rebuild)
        return self._retriever

    def _make_post(self, text: str, post_id: str) -> Post:
        """Create a Post object from raw text."""
        return Post(
            id=post_id or str(uuid.uuid4())[:8],
            text=text,
            raw_text=text,
            source="pipeline_input",
        )

    def analyze(
        self,
        text: str,
        post_id: str = "",
        skip_brief: bool = False,
    ) -> SignalReport:
        """Run full 4-layer SIGNAL analysis on a single post.

        Args:
            text: Social media post text.
            post_id: Optional identifier. Generated if empty.
            skip_brief: If True, skip Layer 4 (Gemini brief) for faster results.

        Returns:
            SignalReport with all layers populated.
        """
        t0 = time.perf_counter()
        text = text.strip() if text else ""
        if not text:
            raise ValueError("Input text must not be empty.")
        if len(text) > 3000:
            raise ValueError("Input text exceeds the 3000-character limit.")
        post = self._make_post(text, post_id)
        retriever = self._ensure_retriever()

        # Layer 1: Substance Resolution
        from signal.substance import ensemble as substance_ensemble
        substance_result: EnsembleResult = substance_ensemble.detect(post)
        logger.info(
            "Layer 1: %d substances detected in post %s",
            len(substance_result.matches), post.id,
        )

        # Layer 2: Narrative Stage Classification
        from signal.narrative import ensemble as narrative_ensemble
        narrative_result: NarrativeEnsembleResult = narrative_ensemble.classify(post)
        stage_name = narrative_result.top_stage.stage
        stage_conf = narrative_result.top_stage.confidence
        logger.info(
            "Layer 2: stage=%s (conf=%.2f) for post %s",
            stage_name, stage_conf, post.id,
        )

        # Layer 3: Clinical Grounding
        contexts: tuple[ClinicalContext, ...] = contextualize_all(
            ensemble_result=substance_result,
            narrative_stage=stage_name,
            retriever=retriever,
        )
        logger.info(
            "Layer 3: %d clinical contexts for post %s",
            len(contexts), post.id,
        )

        # Layer 4: Analyst Brief
        if skip_brief or not contexts:
            brief = ""
        else:
            try:
                brief = generate_brief(
                    original_text=text,
                    narrative_stage=stage_name,
                    narrative_confidence=stage_conf,
                    contexts=contexts,
                )
            except Exception as exc:
                logger.error("Layer 4 failed for post %s: %r", post.id, exc)
                brief = f"[Brief generation failed: {exc!r}]"

        elapsed = (time.perf_counter() - t0) * 1000

        return SignalReport(
            post_id=post.id,
            original_text=text,
            substance_results=substance_result,
            narrative_results=narrative_result,
            clinical_contexts=contexts,
            analyst_brief=brief,
            elapsed_ms=round(elapsed, 2),
        )

    def analyze_batch(
        self,
        texts: list[str],
        skip_brief: bool = False,
    ) -> list[SignalReport]:
        """Run analysis on multiple posts sequentially.

        Args:
            texts: List of post texts.
            skip_brief: If True, skip Gemini brief generation.

        Returns:
            List of SignalReport, one per input text.
        """
        return [
            self.analyze(text, post_id=f"batch_{i}", skip_brief=skip_brief)
            for i, text in enumerate(texts)
        ]
