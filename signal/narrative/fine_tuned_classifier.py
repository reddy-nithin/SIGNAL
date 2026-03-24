"""
Fine-tuned DistilBERT narrative stage classifier — inference.
==============================================================
Loads the best checkpoint from models/distilbert_narrative/
and classifies posts into 6 narrative stages.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from signal.config import (
    DISTILBERT_CHECKPOINT_DIR,
    DISTILBERT_MODEL_NAME,
    DISTILBERT_MAX_LENGTH,
    STAGE_NAMES,
    STAGE_COUNT,
)
from signal.ingestion.post_ingester import Post
from signal.narrative.types import StageClassification, ClassificationResult

logger = logging.getLogger(__name__)

# Module-level model cache
_model: DistilBertForSequenceClassification | None = None
_tokenizer: DistilBertTokenizerFast | None = None
_device: torch.device | None = None


def is_model_available() -> bool:
    """Check if a trained checkpoint exists."""
    return (DISTILBERT_CHECKPOINT_DIR / "config.json").exists()


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model() -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, torch.device]:
    """Load model from checkpoint (cached after first call)."""
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None and _device is not None:
        return _model, _tokenizer, _device

    _device = _get_device()

    if is_model_available():
        logger.info("Loading fine-tuned model from %s", DISTILBERT_CHECKPOINT_DIR)
        _model = DistilBertForSequenceClassification.from_pretrained(
            str(DISTILBERT_CHECKPOINT_DIR),
        )
        _tokenizer = DistilBertTokenizerFast.from_pretrained(str(DISTILBERT_CHECKPOINT_DIR))
    else:
        logger.warning(
            "No fine-tuned checkpoint found at %s — using base model (untrained)",
            DISTILBERT_CHECKPOINT_DIR,
        )
        _model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_NAME, num_labels=STAGE_COUNT,
        )
        _tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)

    _model.to(_device)
    _model.eval()
    return _model, _tokenizer, _device


def classify(post: Post) -> ClassificationResult:
    """Classify a post into a narrative stage via fine-tuned DistilBERT."""
    t0 = time.perf_counter()

    model, tokenizer, device = _load_model()

    encoding = tokenizer(
        post.text,
        truncation=True,
        padding=True,
        max_length=DISTILBERT_MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy().flatten()

    elapsed = (time.perf_counter() - t0) * 1000

    all_stages = tuple(
        StageClassification(
            stage=STAGE_NAMES[i],
            stage_index=i,
            confidence=round(float(probs[i]), 4),
            method="fine_tuned",
            reasoning=f"DistilBERT softmax probability {probs[i]:.3f}",
        )
        for i in range(STAGE_COUNT)
    )

    top_idx = int(np.argmax(probs))
    return ClassificationResult(
        post_id=post.id,
        top_stage=all_stages[top_idx],
        all_stages=all_stages,
        method="fine_tuned",
        elapsed_ms=round(elapsed, 2),
    )


def classify_batch(posts: list[Post], batch_size: int = 32) -> list[ClassificationResult]:
    """Classify a batch of posts with batched inference."""
    if not posts:
        return []

    t0 = time.perf_counter()
    model, tokenizer, device = _load_model()

    results: list[ClassificationResult] = []
    for i in range(0, len(posts), batch_size):
        batch_posts = posts[i : i + batch_size]
        texts = [p.text for p in batch_posts]

        encoding = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=DISTILBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

        elapsed_per = (time.perf_counter() - t0) * 1000 / len(batch_posts)

        for j, post in enumerate(batch_posts):
            p = probs[j]
            all_stages = tuple(
                StageClassification(
                    stage=STAGE_NAMES[k],
                    stage_index=k,
                    confidence=round(float(p[k]), 4),
                    method="fine_tuned",
                    reasoning=f"DistilBERT softmax probability {p[k]:.3f}",
                )
                for k in range(STAGE_COUNT)
            )
            top_idx = int(np.argmax(p))
            results.append(ClassificationResult(
                post_id=post.id,
                top_stage=all_stages[top_idx],
                all_stages=all_stages,
                method="fine_tuned",
                elapsed_ms=round(elapsed_per, 2),
            ))

    return results
