"""
SIGNAL — Day 1 Audit
======================
Validates all Phase 0 deliverables: knowledge chunks, FAERS signals,
FAISS/BM25 indices, hybrid search quality, and datasets.

Run from project root: python scripts/audit_day1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import numpy as np

from signal.config import (
    DATASETS_DIR,
    FAERS_SIGNAL_PATH,
    KNOWLEDGE_CHUNKS_DIR,
    MANIFEST_PATH,
)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> dict:
    status = PASS if condition else (WARN if warn_only else FAIL)
    mark = "✓" if status == PASS else ("⚠" if status == WARN else "✗")
    line = f"  {mark}  {label}"
    if detail:
        line += f": {detail}"
    print(line)
    return {"label": label, "status": status, "detail": detail}


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_audit() -> dict:
    results = []
    t_start = time.time()

    # ── 1. Knowledge Chunks ─────────────────────────────────────────────────
    section("1 / 5 — Knowledge Chunks")

    txt_files = list(KNOWLEDGE_CHUNKS_DIR.glob("*.txt"))
    results.append(check("58 chunk files exist", len(txt_files) == 58,
                          f"found {len(txt_files)}"))

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    results.append(check("manifest.json loads", True, f"{manifest['total_chunks']} entries"))
    results.append(check("manifest total_chunks == 58",
                          manifest["total_chunks"] == 58,
                          str(manifest["total_chunks"])))

    empty = [e["filename"] for e in manifest["chunks"]
             if (KNOWLEDGE_CHUNKS_DIR / e["filename"]).read_text().strip() == ""]
    results.append(check("all chunks non-empty", len(empty) == 0,
                          f"{len(empty)} empty" if empty else "all populated"))

    chunk_types = {e["type"] for e in manifest["chunks"]}
    results.append(check("chunk types present",
                          {"classification", "pharmacology", "safety"}.issubset(chunk_types),
                          str(sorted(chunk_types))))

    # ── 2. FAERS Signals ────────────────────────────────────────────────────
    section("2 / 5 — FAERS Signal Data")

    with open(FAERS_SIGNAL_PATH) as f:
        faers = json.load(f)
    results.append(check("faers_signal_results.json loads", True))
    results.append(check("has 'metadata' key", "metadata" in faers))
    results.append(check("has 'signals' key", "signals" in faers))

    consensus = [s for s in faers["signals"] if s.get("consensus_signal")]
    results.append(check("204 consensus signals", len(consensus) == 204,
                          f"found {len(consensus)}"))

    drugs = {s["drug_name"] for s in faers["signals"]}
    results.append(check("14 drugs in signals", len(drugs) == 14, f"found {len(drugs)}"))

    # ── 3. FAISS + BM25 Indices ─────────────────────────────────────────────
    section("3 / 5 — FAISS / BM25 Indices")

    from signal.grounding.indexer import (
        HybridRetriever,
        build_bm25_index,
        build_faiss_index,
        embed_texts,
        load_chunks,
    )

    chunks = load_chunks()
    results.append(check("load_chunks() returns 58", len(chunks) == 58, f"{len(chunks)}"))

    texts = [c.text for c in chunks]
    t_embed = time.time()
    embeddings, dim = embed_texts(texts[:2])  # quick dim check
    results.append(check("embedding dim > 0", dim > 0, f"dim={dim}"))
    results.append(check("embeddings L2-normalized",
                          abs(float(np.linalg.norm(embeddings[0])) - 1.0) < 1e-4,
                          f"norm={np.linalg.norm(embeddings[0]):.4f}"))

    # Build full index (cached after first run)
    t_build = time.time()
    retriever = HybridRetriever()
    elapsed = time.time() - t_build
    results.append(check("HybridRetriever builds",
                          retriever.chunk_count == 58,
                          f"{retriever.chunk_count} chunks, {elapsed:.1f}s"))
    results.append(check("embedding dim stored",
                          retriever.embedding_dim in (384, 768),
                          f"dim={retriever.embedding_dim}"))

    # ── 4. Hybrid Search Quality ────────────────────────────────────────────
    section("4 / 5 — Hybrid Search Quality")

    test_queries = [
        ("fentanyl overdose respiratory depression",  "fentanyl",  "pharmacology"),
        ("morphine pain management analgesic",         "morphine",  "pharmacology"),
        ("FAERS adverse event signal naloxone",        "naloxone",  "faers_signals"),
        ("opioid epidemic three waves CDC mortality",  None,        "epidemiology"),
    ]

    for query, expected_drug, expected_type in test_queries:
        results_q = retriever.query(query, top_k=5)
        top = results_q[0] if results_q else None

        has_results = len(results_q) > 0
        results.append(check(f"  query '{query[:40]}...' returns results", has_results,
                              f"{len(results_q)} results"))

        if top and expected_drug:
            drug_found = any(expected_drug in (r.chunk.drug_name or "") for r in results_q)
            results.append(check(f"    → finds '{expected_drug}' chunk", drug_found,
                                  f"top: {top.chunk.filename} (score={top.score:.3f})"))

        if top and expected_type:
            type_found = any(r.chunk.chunk_type == expected_type for r in results_q)
            results.append(check(f"    → finds type '{expected_type}'", type_found,
                                  f"types: {[r.chunk.chunk_type for r in results_q]}",
                                  warn_only=True))

    # Check score ordering
    big_results = retriever.query("opioid receptor binding Ki affinity", top_k=10)
    scores = [r.score for r in big_results]
    results.append(check("scores sorted descending",
                          scores == sorted(scores, reverse=True),
                          f"scores: {[f'{s:.3f}' for s in scores[:3]]}"))

    # ── 5. Datasets ─────────────────────────────────────────────────────────
    section("5 / 5 — Datasets")

    dataset_dirs = {
        "reddit_mh_rmhd": 100_000,
        "reddit_mh_labeled": 100_000,
        "reddit_mh_cleaned": 1_000,
        "reddit_mh_research": 1_000,
        "uci_drug_reviews": 50_000,
        "depression_emo": 1_000,
    }

    total_rows = 0
    for dir_name, min_rows in dataset_dirs.items():
        ds_path = DATASETS_DIR / dir_name / "data.csv"
        exists = ds_path.exists()
        if exists:
            import pandas as pd
            try:
                nrows = len(pd.read_csv(ds_path))
                results.append(check(f"  {dir_name}", nrows >= min_rows,
                                      f"{nrows:,} rows"))
                total_rows += nrows
            except Exception as e:
                results.append(check(f"  {dir_name}", False, f"CSV load failed: {e}"))
        else:
            results.append(check(f"  {dir_name}", False, "data.csv missing"))

    results.append(check("total rows > 1M", total_rows > 1_000_000, f"{total_rows:,} rows"))

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    passes = sum(1 for r in results if r["status"] == PASS)
    warns = sum(1 for r in results if r["status"] == WARN)
    fails = sum(1 for r in results if r["status"] == FAIL)

    print(f"\n{'='*60}")
    print(f"  DAY 1 AUDIT COMPLETE — {elapsed_total:.1f}s")
    print(f"  {passes} PASS  |  {warns} WARN  |  {fails} FAIL")
    print(f"{'='*60}")

    if fails > 0:
        print("\nFailed checks:")
        for r in results:
            if r["status"] == FAIL:
                print(f"  ✗ {r['label']}: {r['detail']}")

    return {
        "passes": passes,
        "warns": warns,
        "fails": fails,
        "total_rows": total_rows,
        "embedding_dim": retriever.embedding_dim,
        "chunk_count": retriever.chunk_count,
        "results": results,
    }


if __name__ == "__main__":
    audit = run_audit()
    sys.exit(0 if audit["fails"] == 0 else 1)
