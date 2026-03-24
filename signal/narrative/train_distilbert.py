"""
DistilBERT narrative stage classifier — training script.
==========================================================
5-fold CV, class-weighted loss, Gemini-augmented training data.
Saves best fold checkpoint to models/distilbert_narrative/.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from signal.config import (
    DISTILBERT_MODEL_NAME,
    DISTILBERT_CHECKPOINT_DIR,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_EPOCHS,
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_LEARNING_RATE,
    DISTILBERT_CV_FOLDS,
    STAGE_NAMES,
    STAGE_COUNT,
    TRAINING_EXEMPLARS_PATH,
)
from signal.narrative.stage_exemplars import (
    Exemplar,
    load_exemplars,
    generate_synthetic_exemplars,
    save_exemplars,
    EXEMPLARS_PATH,
)

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class NarrativeStageDataset(Dataset):
    """PyTorch Dataset for narrative stage classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: DistilBertTokenizerFast,
        max_length: int = DISTILBERT_MAX_LENGTH,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Data Preparation ────────────────────────────────────────────────────────

def prepare_training_data(target_total: int = 600) -> tuple[list[str], list[int]]:
    """Load validated exemplars + generate synthetic to reach target_total.

    Returns (texts, labels) with labels as stage indices 0-5.
    """
    exemplars = load_exemplars(EXEMPLARS_PATH)
    logger.info("Loaded %d validated exemplars", len(exemplars))

    # Count per stage
    counts = Counter(e.stage for e in exemplars)
    target_per_stage = target_total // STAGE_COUNT

    # Generate synthetic exemplars for under-represented stages
    augmented = list(exemplars)
    for stage_name in STAGE_NAMES:
        current = counts.get(stage_name, 0)
        needed = target_per_stage - current
        if needed > 0:
            logger.info("Generating %d synthetic exemplars for %s", needed, stage_name)
            synthetic = generate_synthetic_exemplars(stage_name, count=needed)
            augmented.extend(synthetic)

    # Deduplicate by exact text
    seen: set[str] = set()
    unique: list[Exemplar] = []
    for ex in augmented:
        text_key = ex.text.strip().lower()
        if text_key not in seen:
            seen.add(text_key)
            unique.append(ex)

    # Save augmented set
    TRAINING_EXEMPLARS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_exemplars(unique, TRAINING_EXEMPLARS_PATH)

    texts = [ex.text for ex in unique]
    labels = [ex.stage_index for ex in unique]
    logger.info(
        "Training data: %d samples, distribution: %s",
        len(texts),
        Counter(labels),
    )
    return texts, labels


# ── Training ─────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_fold(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    class_weights: torch.Tensor,
    fold_idx: int,
    device: torch.device,
) -> dict:
    """Train a single fold and return metrics."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME, num_labels=STAGE_COUNT,
    )
    model.to(device)

    train_ds = NarrativeStageDataset(train_texts, train_labels, tokenizer)
    val_ds = NarrativeStageDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=DISTILBERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DISTILBERT_BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=DISTILBERT_LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * DISTILBERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(DISTILBERT_EPOCHS):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds: list[int] = []
        all_true: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(batch["labels"].tolist())

        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)

        logger.info(
            "Fold %d Epoch %d: loss=%.4f val_acc=%.4f val_f1_macro=%.4f",
            fold_idx, epoch + 1, avg_loss, val_acc, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best state for final report
    if best_state:
        model.load_state_dict(best_state)

    report = classification_report(
        all_true, all_preds, target_names=list(STAGE_NAMES), zero_division=0,
    )

    return {
        "fold": fold_idx,
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(best_val_f1, 4),
        "classification_report": report,
        "model_state": best_state,
    }


def run_training(target_total: int = 600) -> dict:
    """Run 5-fold cross-validation and save the best fold's model.

    Returns full CV report with per-fold and aggregate metrics.
    """
    device = _get_device()
    logger.info("Training on device: %s", device)

    texts, labels = prepare_training_data(target_total)
    labels_arr = np.array(labels)

    # Compute class weights
    cw = compute_class_weight("balanced", classes=np.arange(STAGE_COUNT), y=labels_arr)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    logger.info("Class weights: %s", cw.round(3))

    skf = StratifiedKFold(n_splits=DISTILBERT_CV_FOLDS, shuffle=True, random_state=42)
    fold_results: list[dict] = []
    best_fold_idx = -1
    best_fold_f1 = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info("=== Fold %d/%d ===", fold_idx + 1, DISTILBERT_CV_FOLDS)
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        result = train_one_fold(
            train_texts, train_labels, val_texts, val_labels,
            class_weights, fold_idx, device,
        )
        fold_results.append(result)

        if result["val_f1_macro"] > best_fold_f1:
            best_fold_f1 = result["val_f1_macro"]
            best_fold_idx = fold_idx

    # Save best fold model
    best_state = fold_results[best_fold_idx]["model_state"]
    DISTILBERT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME, num_labels=STAGE_COUNT,
    )
    model.load_state_dict(best_state)
    model.save_pretrained(str(DISTILBERT_CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_CHECKPOINT_DIR))
    logger.info("Saved best model (fold %d, F1=%.4f) to %s", best_fold_idx, best_fold_f1, DISTILBERT_CHECKPOINT_DIR)

    # Aggregate metrics
    f1_scores = [r["val_f1_macro"] for r in fold_results]
    acc_scores = [r["val_accuracy"] for r in fold_results]

    # Clean up model states from report (not serializable)
    for r in fold_results:
        del r["model_state"]

    report = {
        "n_folds": DISTILBERT_CV_FOLDS,
        "n_samples": len(texts),
        "label_distribution": dict(Counter(labels)),
        "class_weights": cw.tolist(),
        "best_fold": best_fold_idx,
        "mean_f1_macro": round(float(np.mean(f1_scores)), 4),
        "std_f1_macro": round(float(np.std(f1_scores)), 4),
        "mean_accuracy": round(float(np.mean(acc_scores)), 4),
        "std_accuracy": round(float(np.std(acc_scores)), 4),
        "fold_results": fold_results,
    }

    # Save report
    report_path = DISTILBERT_CHECKPOINT_DIR / "cv_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("CV report saved to %s", report_path)

    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Samples: {len(texts)} | Best fold: {best_fold_idx}")
    print(f"Mean F1 (macro): {report['mean_f1_macro']:.4f} +/- {report['std_f1_macro']:.4f}")
    print(f"Mean Accuracy:   {report['mean_accuracy']:.4f} +/- {report['std_accuracy']:.4f}")
    print(f"Checkpoint: {DISTILBERT_CHECKPOINT_DIR}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_training()
