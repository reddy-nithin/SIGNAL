"""
LLM-based narrative stage classifier.
=======================================
Gemini few-shot classification with exemplar context.
All calls cached to disk via SHA256 content hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from signal.config import (
    GEMINI_MODEL,
    GEMINI_NARRATIVE_CACHE_DIR,
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
    NARRATIVE_STAGES,
    STAGE_NAMES,
    STAGE_COUNT,
)
from signal.ingestion.post_ingester import Post
from signal.narrative.types import StageClassification, ClassificationResult
from signal.narrative.stage_exemplars import load_exemplars, Exemplar, EXEMPLARS_PATH

logger = logging.getLogger(__name__)

# ── Caching (same pattern as substance/llm_detector.py) ─────────────────────


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    d = GEMINI_NARRATIVE_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _get_cached(prompt: str) -> str | None:
    p = _cache_path(_cache_key(prompt))
    if p.exists():
        return json.loads(p.read_text())["response"]
    return None


def _set_cached(prompt: str, response: str) -> None:
    p = _cache_path(_cache_key(prompt))
    p.write_text(json.dumps({"prompt_hash": _cache_key(prompt), "response": response}))


# ── Gemini call ──────────────────────────────────────────────────────────────


def _call_gemini(prompt: str) -> str:
    """Call Gemini with disk caching."""
    cached = _get_cached(prompt)
    if cached is not None:
        return cached

    from signal.config import _get_secret  # noqa: PLC0415
    api_key = _get_secret("GEMINI_API_KEY")
    if api_key:
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text
    else:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = response.text

    _set_cached(prompt, text)
    return text


def _parse_response(response_text: str) -> dict:
    """Parse JSON object from Gemini response, stripping markdown fences."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini narrative response: %s...", text[:200])
        return {}


# ── Few-shot prompt ──────────────────────────────────────────────────────────

_exemplars_cache: list[Exemplar] | None = None


def _load_exemplars_cached() -> list[Exemplar]:
    global _exemplars_cache
    if _exemplars_cache is None:
        try:
            _exemplars_cache = load_exemplars(EXEMPLARS_PATH)
        except FileNotFoundError:
            logger.warning("No validated exemplars found at %s", EXEMPLARS_PATH)
            _exemplars_cache = []
    return _exemplars_cache


def _build_few_shot_prompt(post_text: str, exemplars_per_stage: int = 4) -> str:
    """Build a few-shot classification prompt with exemplars and chain-of-thought."""
    exemplars = _load_exemplars_cached()

    # Select top exemplars per stage (highest confidence validated ones)
    by_stage: dict[str, list[Exemplar]] = {s: [] for s in STAGE_NAMES}
    for ex in exemplars:
        if ex.stage in by_stage:
            by_stage[ex.stage].append(ex)

    examples_block = ""
    for stage_name in STAGE_NAMES:
        stage_exs = sorted(by_stage[stage_name], key=lambda e: e.confidence, reverse=True)
        for ex in stage_exs[:exemplars_per_stage]:
            txt = ex.text[:300].replace('"', '\\"')
            examples_block += f'Post: "{txt}"\nStage: {stage_name}\n\n'

    stage_descriptions = "\n".join(
        f"- {s.name} (index {s.index}): {s.description}. Signals: {', '.join(s.key_signals[:3])}"
        for s in NARRATIVE_STAGES
    )

    return f"""You are an expert in addiction science and behavioral health. Your task is to classify a social media post into one of 6 narrative stages of the addiction arc.

## 6 Narrative Stages
{stage_descriptions}

## Examples
{examples_block}
## Task
Before classifying, reason through these 5 questions:
1. What substances are mentioned or implied?
2. What is the temporal framing? (past experience, present state, future intent, hypothetical)
3. What emotional tone dominates? (curious, casual/recreational, functional, distressed, hopeful)
4. Does the language indicate control over use or loss of control?
5. Are there consequences mentioned? (social, health, legal, financial)

Then classify the post into exactly ONE stage. For posts that could belong to multiple stages, weight the most prominent narrative signal. If truly ambiguous, distribute confidence more evenly across plausible stages rather than forcing a high-confidence single pick.

Post: "{post_text[:2000]}"

Return ONLY a JSON object with this structure:
{{"stage": "StageName", "stage_index": 0, "confidence": 0.85, "reasoning_steps": "1. Mentions fentanyl. 2. Present tense, ongoing. 3. Distressed tone. 4. Loss of control — can't stop. 5. Health consequences.", "all_stages": [{{"stage": "Curiosity", "score": 0.05}}, {{"stage": "Experimentation", "score": 0.05}}, {{"stage": "Regular Use", "score": 0.10}}, {{"stage": "Dependence", "score": 0.70}}, {{"stage": "Crisis", "score": 0.08}}, {{"stage": "Recovery", "score": 0.02}}], "reasoning": "Brief explanation"}}"""


# ── Classification ───────────────────────────────────────────────────────────

_UNIFORM = 1.0 / STAGE_COUNT


def classify(post: Post) -> ClassificationResult:
    """Classify a post into a narrative stage via Gemini few-shot."""
    t0 = time.perf_counter()

    prompt = _build_few_shot_prompt(post.text)
    response = _call_gemini(prompt)
    parsed = _parse_response(response)

    # Extract per-stage scores
    score_map: dict[str, float] = {}
    all_stages_raw = parsed.get("all_stages", [])
    for entry in all_stages_raw:
        if isinstance(entry, dict):
            name = entry.get("stage", "")
            score = float(entry.get("score", 0.0))
            if name in STAGE_NAMES:
                score_map[name] = min(max(score, 0.0), 1.0)

    reasoning = parsed.get("reasoning", "Gemini few-shot classification")

    # Fallback: if parse failed or incomplete, use uniform scores
    if len(score_map) < STAGE_COUNT:
        # Try to salvage top-level stage/confidence
        top_stage_name = parsed.get("stage", "")
        top_conf = float(parsed.get("confidence", 0.0))
        if top_stage_name in STAGE_NAMES and top_conf > 0:
            for name in STAGE_NAMES:
                score_map[name] = score_map.get(name, (1 - top_conf) / max(STAGE_COUNT - 1, 1))
            score_map[top_stage_name] = top_conf
        else:
            score_map = {name: _UNIFORM for name in STAGE_NAMES}
            reasoning = "Parse failure — uniform fallback"

    elapsed = (time.perf_counter() - t0) * 1000

    all_stages = tuple(
        StageClassification(
            stage=name,
            stage_index=i,
            confidence=round(score_map.get(name, _UNIFORM), 4),
            method="llm",
            reasoning=reasoning,
        )
        for i, name in enumerate(STAGE_NAMES)
    )

    top_idx = max(range(STAGE_COUNT), key=lambda i: all_stages[i].confidence)
    return ClassificationResult(
        post_id=post.id,
        top_stage=all_stages[top_idx],
        all_stages=all_stages,
        method="llm",
        elapsed_ms=round(elapsed, 2),
    )


def classify_batch(posts: list[Post]) -> list[ClassificationResult]:
    """Classify a batch of posts (sequential, cache-backed)."""
    return [classify(p) for p in posts]
