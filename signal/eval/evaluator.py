"""
Phase 2 Evaluator — Substance Detection Validation.
=====================================================
Evaluates all 3 detection methods + ensemble against UCI Drug Review
ground truth and computes precision/recall/F1/agreement metrics.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from signal.config import DATASETS_DIR, EVIDENCE_DIR
from signal.ingestion.post_ingester import Post, load_uci_drug_reviews
from signal.substance import rule_based_detector
from signal.substance.ensemble import (
    compute_agreement_stats,
    detect_from_results,
)
from signal.substance.slang_lexicon import (
    CLINICAL_TO_CLASS,
    SLANG_LEXICON,
    get_clinical_name,
)
from signal.substance.types import DetectionResult, EnsembleResult

# ── Brand → clinical alias map ──────────────────────────────────────────────
# UCI Drug Reviews use brand names; our lexicon uses clinical names.
# This maps UCI drugName (lowercased) → our clinical_name.

_BRAND_TO_CLINICAL: dict[str, str] = {
    # Opioids
    "oxycontin": "oxycodone",
    "percocet": "oxycodone",
    "roxicodone": "oxycodone",
    "vicodin": "hydrocodone",
    "norco": "hydrocodone",
    "lortab": "hydrocodone",
    "dilaudid": "hydromorphone",
    "opana": "oxymorphone",
    "suboxone": "buprenorphine",
    "subutex": "buprenorphine",
    "methadose": "methadone",
    "demerol": "meperidine",
    "ultram": "tramadol",
    "nucynta": "tapentadol",
    "narcan": "naloxone",
    "vivitrol": "naltrexone",
    "revia": "naltrexone",
    "ms contin": "morphine",
    # Benzos
    "xanax": "alprazolam",
    "klonopin": "clonazepam",
    "valium": "diazepam",
    "ativan": "lorazepam",
    "restoril": "temazepam",
    "lyrica": "pregabalin",
    "neurontin": "gabapentin",
    # Stimulants
    "adderall": "amphetamine",
    "dexedrine": "amphetamine",
    "ritalin": "methylphenidate",
    "concerta": "methylphenidate",
}

# Build lookup: also include all clinical names that map to themselves
_UCI_DRUG_LOOKUP: dict[str, str] = {}
for entry in SLANG_LEXICON:
    _UCI_DRUG_LOOKUP[entry.clinical_name] = entry.clinical_name
for brand, clinical in _BRAND_TO_CLINICAL.items():
    _UCI_DRUG_LOOKUP[brand] = clinical


def _normalize_uci_drug(drug_name: str) -> str | None:
    """Map a UCI drugName to our clinical name, or None if not in our scope."""
    key = drug_name.strip().lower()
    if key in _UCI_DRUG_LOOKUP:
        return _UCI_DRUG_LOOKUP[key]
    # Try lexicon lookup (handles slang terms too)
    entry = get_clinical_name(key)
    if entry is not None:
        return entry.clinical_name
    return None


def load_uci_substance_subset(limit: int = 2000) -> list[tuple[Post, str]]:
    """Load UCI posts filtered to drugs in our lexicon.

    Returns list of (Post, ground_truth_clinical_name) tuples.
    """
    posts = load_uci_drug_reviews(max_rows=50_000)

    result: list[tuple[Post, str]] = []
    for post in posts:
        if post.drug_name is None:
            continue
        clinical = _normalize_uci_drug(post.drug_name)
        if clinical is None:
            continue
        result.append((post, clinical))
        if len(result) >= limit:
            break

    return result


# ── Per-method evaluation ────────────────────────────────────────────────────


def evaluate_rule_based(
    posts_with_truth: list[tuple[Post, str]],
) -> dict:
    """Evaluate rule-based detector against ground truth.

    Returns dict with precision, recall, f1, per_class breakdown, and raw counts.
    """
    tp = 0
    fp = 0
    fn = 0
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for post, truth_clinical in posts_with_truth:
        result = rule_based_detector.detect(post)
        detected_clinicals = {m.clinical_name for m in result.matches if not m.is_negated}

        truth_class = CLINICAL_TO_CLASS.get(truth_clinical, "other")

        if truth_clinical in detected_clinicals:
            tp += 1
            per_class[truth_class]["tp"] += 1
        else:
            fn += 1
            per_class[truth_class]["fn"] += 1

        # FP: anything detected that isn't the truth
        false_positives = detected_clinicals - {truth_clinical}
        fp += len(false_positives)
        for fp_clinical in false_positives:
            fp_class = CLINICAL_TO_CLASS.get(fp_clinical, "other")
            per_class[fp_class]["fp"] += 1

    return _compute_metrics("rule_based", tp, fp, fn, dict(per_class))


def evaluate_ensemble_from_rule_based(
    posts_with_truth: list[tuple[Post, str]],
) -> dict:
    """Evaluate ensemble using only rule_based results (no API calls).

    This provides a realistic evaluation of what the ensemble can produce
    given rule-based input. For a full evaluation with embedding + LLM,
    use evaluate_full_ensemble().
    """
    tp = 0
    fp = 0
    fn = 0
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    ensemble_results: list[EnsembleResult] = []

    for post, truth_clinical in posts_with_truth:
        rb_result = rule_based_detector.detect(post)
        # Create empty results for embedding/llm to simulate offline evaluation
        empty_emb = DetectionResult(post_id=post.id, matches=(), method="embedding", elapsed_ms=0.0)
        empty_llm = DetectionResult(post_id=post.id, matches=(), method="llm", elapsed_ms=0.0)
        ens_result = detect_from_results(
            post.id,
            (rb_result, empty_emb, empty_llm),
        )
        ensemble_results.append(ens_result)

        detected_clinicals = {m.clinical_name for m in ens_result.matches if not m.is_negated}
        truth_class = CLINICAL_TO_CLASS.get(truth_clinical, "other")

        if truth_clinical in detected_clinicals:
            tp += 1
            per_class[truth_class]["tp"] += 1
        else:
            fn += 1
            per_class[truth_class]["fn"] += 1

        false_positives = detected_clinicals - {truth_clinical}
        fp += len(false_positives)
        for fp_clinical in false_positives:
            fp_class = CLINICAL_TO_CLASS.get(fp_clinical, "other")
            per_class[fp_class]["fp"] += 1

    return _compute_metrics("ensemble_rb_only", tp, fp, fn, dict(per_class))


def _compute_metrics(
    method: str,
    tp: int,
    fp: int,
    fn: int,
    per_class: dict[str, dict[str, int]],
) -> dict:
    """Compute precision/recall/F1 from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    class_metrics = {}
    for cls, counts in per_class.items():
        c_tp, c_fp, c_fn = counts["tp"], counts["fp"], counts["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        class_metrics[cls] = {
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
            "support": c_tp + c_fn,
        }

    return {
        "method": method,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": tp + fn,
        "per_class": class_metrics,
    }


def build_drug_class_distribution(
    posts_with_truth: list[tuple[Post, str]],
) -> dict[str, int]:
    """Count how many posts per drug class in the evaluation set."""
    counts: Counter[str] = Counter()
    for _, truth_clinical in posts_with_truth:
        cls = CLINICAL_TO_CLASS.get(truth_clinical, "other")
        counts[cls] += 1
    return dict(counts.most_common())


# ── Main evaluator ───────────────────────────────────────────────────────────


def run_phase2_evaluation(
    limit: int = 2000,
    save: bool = True,
) -> dict:
    """Run full Phase 2 substance detection evaluation.

    1. Load UCI subset (filtered to our substances)
    2. Evaluate rule-based detector
    3. Evaluate ensemble (rule-based only, no API)
    4. Compute drug class distribution
    5. Optionally save results to evidence/phase2/

    Returns comprehensive evaluation report.
    """
    print(f"Loading UCI substance subset (limit={limit})...")
    t0 = time.perf_counter()
    data = load_uci_substance_subset(limit=limit)
    print(f"  Found {len(data)} posts matching our lexicon")

    # Drug class distribution
    class_dist = build_drug_class_distribution(data)
    print(f"  Drug class distribution: {class_dist}")

    # Evaluate rule-based
    print("\nEvaluating rule-based detector...")
    rb_metrics = evaluate_rule_based(data)
    _print_metrics(rb_metrics)

    # Evaluate ensemble (rule-based only)
    print("\nEvaluating ensemble (rule-based only)...")
    ens_metrics = evaluate_ensemble_from_rule_based(data)
    _print_metrics(ens_metrics)

    elapsed = time.perf_counter() - t0
    report = {
        "evaluation_date": "2026-03-23",
        "dataset": "uci_drug_reviews",
        "n_posts": len(data),
        "class_distribution": class_dist,
        "rule_based": rb_metrics,
        "ensemble_rb_only": ens_metrics,
        "elapsed_seconds": round(elapsed, 2),
        "notes": (
            "Embedding and LLM detectors require API calls. "
            "This offline evaluation uses rule-based only. "
            "Full 3-method evaluation requires running with API access."
        ),
    }

    if save:
        out_dir = EVIDENCE_DIR / "phase2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "substance_eval_results.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return report


def _print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n  Method: {metrics['method']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Total={metrics['total']}")
    if metrics.get("per_class"):
        print("  Per-class:")
        for cls, m in sorted(metrics["per_class"].items()):
            print(f"    {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")


if __name__ == "__main__":
    run_phase2_evaluation(limit=2000, save=True)
