"""
SIGNAL — Analyst Brief Generator (Layer 4)
=============================================
Assembles all 4 layers into a Gemini-synthesized intelligence brief
with citation enforcement. Every claim references [KB:filename] or
[FAERS:drug+reaction].

All Gemini calls are cached to disk via SHA256 content hash.

Public API:
    generate_brief()  — produce analyst brief from clinical contexts
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from signal.config import (
    GEMINI_MODEL,
    GEMINI_BRIEF_CACHE_DIR,
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
)
from signal.grounding.types import ClinicalContext

logger = logging.getLogger(__name__)

# ── Caching (same pattern as narrative/llm_classifier.py) ────────────────────


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    d = GEMINI_BRIEF_CACHE_DIR
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


# ── Prompt Assembly ──────────────────────────────────────────────────────────


def _format_evidence_block(ctx: ClinicalContext) -> str:
    """Format a single substance's clinical context for the prompt."""
    lines: list[str] = []
    lines.append(f"## Substance: {ctx.substance} (class: {ctx.drug_class})")
    lines.append(f"Narrative Stage: {ctx.narrative_stage}")
    lines.append("")

    # Knowledge base evidence
    lines.append("### Retrieved Knowledge Chunks:")
    for ev in ctx.evidence[:5]:
        lines.append(f"- [KB:{ev.chunk_filename}] (score={ev.relevance_score:.2f})")
        lines.append(f"  {ev.text_snippet[:200]}...")
    lines.append("")

    # FAERS signals
    if ctx.faers_signals:
        lines.append("### Adverse Event Signals:")
        for sig in ctx.faers_signals[:10]:
            prr_str = f"PRR={sig.prr:.1f}" if sig.prr else "PRR=N/A"
            ror_str = f"ROR={sig.ror:.1f}" if sig.ror else "ROR=N/A"
            lines.append(
                f"- [FAERS:{sig.drug_name}+{sig.reaction}] {prr_str}, {ror_str} (source: {sig.source})"
            )
    lines.append("")

    # Interaction warnings
    if ctx.interactions:
        lines.append("### Poly-Drug Interaction Warnings:")
        for warn in ctx.interactions:
            lines.append(
                f"- Substances: {', '.join(warn.substances)} — {warn.risk_description[:200]}"
            )
            lines.append(f"  Source: [KB:{warn.source_chunk}]")

    return "\n".join(lines)


_BRIEF_SYSTEM_PROMPT = """You are a SIGNAL Intelligence Analyst producing a structured brief for public health workers monitoring substance abuse signals on social media.

CITATION RULES (MANDATORY):
- Every factual claim MUST cite its source using [KB:filename] or [FAERS:drug+reaction] format.
- If you cannot cite a claim, do not include it.
- Do not invent citations that are not in the provided evidence.

OUTPUT FORMAT:
---
SIGNAL INTELLIGENCE BRIEF
Date: {date}

1. SUBSTANCE IDENTIFICATION
[Detected substances with drug classes]

2. NARRATIVE STAGE ASSESSMENT
[Stage classification with confidence and reasoning]

3. CLINICAL RISK PROFILE
[For each substance: pharmacology summary, key adverse events, citing KB and FAERS]

4. POLY-DRUG INTERACTION RISKS
[If multiple substances: interaction warnings with citations]

5. STAGE-SPECIFIC RISK ANNOTATION
[Tailored risk assessment based on narrative stage + substances]

6. RECOMMENDED ACTIONS
[Public health response suggestions based on evidence]
---
"""


def _build_prompt(
    original_text: str,
    narrative_stage: str,
    narrative_confidence: float,
    contexts: tuple[ClinicalContext, ...],
) -> str:
    """Assemble the full prompt for the analyst brief."""
    evidence_blocks = "\n\n".join(_format_evidence_block(ctx) for ctx in contexts)
    substance_list = ", ".join(f"{ctx.substance} ({ctx.drug_class})" for ctx in contexts)

    return f"""{_BRIEF_SYSTEM_PROMPT}

# INPUT POST
\"{original_text}\"

# LAYER 1 — SUBSTANCE RESOLUTION
Detected substances: {substance_list}

# LAYER 2 — NARRATIVE STAGE
Stage: {narrative_stage} (confidence: {narrative_confidence:.2f})

# LAYER 3 — CLINICAL EVIDENCE
{evidence_blocks}

Produce the SIGNAL INTELLIGENCE BRIEF now. Cite every claim."""


# ── Public API ───────────────────────────────────────────────────────────────


def generate_brief(
    original_text: str,
    narrative_stage: str,
    narrative_confidence: float,
    contexts: tuple[ClinicalContext, ...],
) -> str:
    """Generate an analyst brief by synthesizing all 4 SIGNAL layers.

    Args:
        original_text: The original social media post text.
        narrative_stage: Top predicted stage name from Layer 2.
        narrative_confidence: Confidence score of the top stage.
        contexts: Clinical context objects from Layer 3.

    Returns:
        Formatted SIGNAL INTELLIGENCE BRIEF string.
        Returns a fallback message if Gemini call fails.
    """
    if not contexts:
        return "SIGNAL INTELLIGENCE BRIEF\n\nNo substances detected. No clinical grounding available."

    prompt = _build_prompt(original_text, narrative_stage, narrative_confidence, contexts)

    try:
        brief = _call_gemini(prompt)
        return brief.strip()
    except Exception as exc:
        logger.error("Brief generation failed: %r", exc)
        return (
            "SIGNAL INTELLIGENCE BRIEF\n\n"
            "[Brief generation unavailable — Gemini API error]\n\n"
            f"Detected substances: {', '.join(ctx.substance for ctx in contexts)}\n"
            f"Narrative stage: {narrative_stage} (confidence: {narrative_confidence:.2f})\n"
            f"FAERS signals found: {sum(len(ctx.faers_signals) for ctx in contexts)}\n"
            f"Knowledge chunks retrieved: {sum(len(ctx.evidence) for ctx in contexts)}"
        )
