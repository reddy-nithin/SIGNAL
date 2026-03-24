"""
Phase 0 test suite — validates config, chunk loading, FAISS, BM25, and hybrid search.

Fast tests use synthetic data (3-5 fake chunks) and don't call Vertex AI.
Vertex AI tests are skipped if no credentials are available.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_vertex_credentials() -> bool:
    """Check if Vertex AI credentials are available."""
    try:
        import vertexai  # noqa
        import google.auth
        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        return False


HAS_VERTEX = _has_vertex_credentials()

SYNTHETIC_TEXTS = [
    "fentanyl is a synthetic opioid with high potency and respiratory depression risk",
    "morphine is a natural opioid used for pain management and palliative care",
    "naloxone reverses opioid overdose by blocking mu-opioid receptors rapidly",
    "buprenorphine is used in medication-assisted treatment for opioid dependence",
    "heroin addiction leads to severe withdrawal including nausea vomiting anxiety",
]


def _make_synthetic_chunks():
    """Create minimal ChunkRecord list for FAISS/BM25 unit tests."""
    from signal.grounding.indexer import ChunkRecord
    return [
        ChunkRecord(
            index=i,
            filename=f"test_{i}.txt",
            chunk_type="pharmacology",
            drug_name=None,
            rxcui=None,
            text=text,
            token_estimate=len(text.split()),
        )
        for i, text in enumerate(SYNTHETIC_TEXTS)
    ]


# ── TestConfig ────────────────────────────────────────────────────────────────

class TestConfig:
    def test_project_root_exists(self):
        from signal.config import PROJECT_ROOT
        assert PROJECT_ROOT.exists(), f"PROJECT_ROOT not found: {PROJECT_ROOT}"

    def test_opioid_data_dir_exists(self):
        from signal.config import OPIOID_DATA_DIR
        assert OPIOID_DATA_DIR.exists()

    def test_knowledge_chunks_dir_exists(self):
        from signal.config import KNOWLEDGE_CHUNKS_DIR
        assert KNOWLEDGE_CHUNKS_DIR.exists()

    def test_chunk_files_at_least_84(self):
        from signal.config import KNOWLEDGE_CHUNKS_DIR
        txt_files = list(KNOWLEDGE_CHUNKS_DIR.glob("*.txt"))
        assert len(txt_files) >= 84, f"Expected >=84 chunks, found {len(txt_files)}"

    def test_manifest_exists(self):
        from signal.config import MANIFEST_PATH
        assert MANIFEST_PATH.exists()

    def test_faers_json_exists(self):
        from signal.config import FAERS_SIGNAL_PATH
        assert FAERS_SIGNAL_PATH.exists()

    def test_faers_json_structure(self):
        from signal.config import FAERS_SIGNAL_PATH
        with open(FAERS_SIGNAL_PATH) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "signals" in data
        consensus = [s for s in data["signals"] if s.get("consensus_signal")]
        assert len(consensus) == 204, f"Expected 204 consensus signals, got {len(consensus)}"

    def test_6_narrative_stages(self):
        from signal.config import NARRATIVE_STAGES, STAGE_COUNT
        assert STAGE_COUNT == 6
        assert len(NARRATIVE_STAGES) == 6

    def test_stage_names_correct(self):
        from signal.config import STAGE_NAMES
        expected = ("Curiosity", "Experimentation", "Regular Use", "Dependence", "Crisis", "Recovery")
        assert STAGE_NAMES == expected

    def test_stages_are_frozen(self):
        from signal.config import NARRATIVE_STAGES
        with pytest.raises((TypeError, AttributeError)):
            NARRATIVE_STAGES[0].name = "Changed"  # type: ignore

    def test_21_safety_terms(self):
        from signal.config import OPIOID_SAFETY_TERMS
        assert len(OPIOID_SAFETY_TERMS) == 21

    def test_14_opioids(self):
        from signal.config import MUST_INCLUDE_OPIOIDS
        assert len(MUST_INCLUDE_OPIOIDS) == 14

    def test_14_mme_factors(self):
        from signal.config import CDC_MME_FACTORS
        assert len(CDC_MME_FACTORS) == 14

    def test_paths_are_pathlib(self):
        from signal import config
        for name in ["PROJECT_ROOT", "OPIOID_DATA_DIR", "KNOWLEDGE_CHUNKS_DIR",
                     "FAISS_INDEX_PATH", "BM25_INDEX_PATH"]:
            val = getattr(config, name)
            assert isinstance(val, Path), f"{name} should be Path, got {type(val)}"

    def test_embedding_dim(self):
        from signal.config import EMBEDDING_DIM, FALLBACK_EMBEDDING_DIM
        assert EMBEDDING_DIM == 768
        assert FALLBACK_EMBEDDING_DIM == 384


# ── TestChunkLoading ──────────────────────────────────────────────────────────

class TestChunkLoading:
    def test_load_returns_at_least_84(self):
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        assert len(chunks) >= 84

    def test_all_chunks_non_empty(self):
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        for c in chunks:
            assert len(c.text) > 0, f"Empty chunk: {c.filename}"

    def test_chunk_types_known(self):
        from signal.config import CHUNK_TYPES
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        for c in chunks:
            assert c.chunk_type in CHUNK_TYPES, f"Unknown type: {c.chunk_type} in {c.filename}"

    def test_chunk_indices_sequential(self):
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_chunk_record_is_frozen(self):
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        with pytest.raises((TypeError, AttributeError)):
            chunks[0].text = "mutated"  # type: ignore

    def test_ingredient_pharmacology_chunks_have_drug_name(self):
        from signal.grounding.indexer import load_chunks
        chunks = load_chunks()
        # Only ingredient-level pharmacology chunks must have drug_name
        # Class-level chunks (benzo_pharmacology, stimulant_pharmacology, etc.) may not
        ingredient_pharma = [
            c for c in chunks
            if c.chunk_type == "pharmacology" and c.filename.startswith("ingredient_")
        ]
        assert len(ingredient_pharma) > 0
        for c in ingredient_pharma:
            assert c.drug_name is not None, f"Ingredient chunk missing drug_name: {c.filename}"

    def test_missing_chunk_raises(self):
        from signal.grounding.indexer import load_chunks
        with pytest.raises(FileNotFoundError):
            load_chunks(chunks_dir=Path("/nonexistent/dir"))


# ── TestFAISSIndex ────────────────────────────────────────────────────────────

class TestFAISSIndex:
    def _small_embeddings(self, n: int = 5, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.random((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_build_correct_dimension(self):
        from signal.grounding.indexer import build_faiss_index
        emb = self._small_embeddings(dim=16)
        idx = build_faiss_index(emb)
        assert idx.d == 16

    def test_build_correct_count(self):
        from signal.grounding.indexer import build_faiss_index
        emb = self._small_embeddings(n=5)
        idx = build_faiss_index(emb)
        assert idx.ntotal == 5

    def test_search_returns_k_results(self):
        from signal.grounding.indexer import build_faiss_index
        emb = self._small_embeddings(n=10, dim=8)
        idx = build_faiss_index(emb)
        query = emb[0:1]
        scores, indices = idx.search(query, 3)
        assert scores.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_save_load_roundtrip(self, tmp_path):
        from signal.grounding.indexer import build_faiss_index, save_faiss_index, load_faiss_index
        emb = self._small_embeddings(n=5, dim=8)
        idx = build_faiss_index(emb)
        path = tmp_path / "test.bin"
        save_faiss_index(idx, path)
        assert path.exists()
        loaded = load_faiss_index(path)
        assert loaded.d == idx.d
        assert loaded.ntotal == idx.ntotal

    def test_self_similarity_is_highest(self):
        from signal.grounding.indexer import build_faiss_index
        emb = self._small_embeddings(n=5, dim=8)
        idx = build_faiss_index(emb)
        scores, indices = idx.search(emb[0:1], 5)
        assert indices[0][0] == 0, "First result should be the query itself"


# ── TestBM25Index ─────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_returns_scores(self):
        from signal.grounding.indexer import build_bm25_index
        bm25, tokenized = build_bm25_index(SYNTHETIC_TEXTS)
        scores = bm25.get_scores(["fentanyl", "overdose"])
        assert len(scores) == len(SYNTHETIC_TEXTS)
        assert scores[0] > 0, "First doc (about fentanyl) should score > 0 for 'fentanyl'"

    def test_relevant_doc_scores_highest(self):
        from signal.grounding.indexer import build_bm25_index
        bm25, _ = build_bm25_index(SYNTHETIC_TEXTS)
        scores = bm25.get_scores(["naloxone", "reversal", "overdose"])
        best_idx = int(np.argmax(scores))
        assert best_idx == 2, f"Naloxone doc (idx=2) should score highest, got idx={best_idx}"

    def test_save_load_roundtrip(self, tmp_path):
        from signal.grounding.indexer import build_bm25_index, save_bm25_index, load_bm25_index
        bm25, tokenized = build_bm25_index(SYNTHETIC_TEXTS)
        path = tmp_path / "bm25.pkl"
        save_bm25_index(bm25, tokenized, path)
        assert path.exists()
        loaded_bm25, loaded_tok = load_bm25_index(path)
        scores_orig = bm25.get_scores(["fentanyl"])
        scores_loaded = loaded_bm25.get_scores(["fentanyl"])
        np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)

    def test_tokenization_lowercases(self):
        from signal.grounding.indexer import _tokenize
        tokens = _tokenize("Fentanyl OVERDOSE Risk")
        assert all(t == t.lower() for t in tokens)


# ── TestHybridSearch ──────────────────────────────────────────────────────────

class TestHybridSearch:
    """Uses synthetic embeddings to avoid Vertex AI calls."""

    def _setup(self):
        """Build synthetic FAISS + BM25 indices over SYNTHETIC_TEXTS."""
        from signal.grounding.indexer import build_faiss_index, build_bm25_index, _l2_normalize
        chunks = _make_synthetic_chunks()
        rng = np.random.default_rng(0)
        emb = rng.random((len(SYNTHETIC_TEXTS), 8)).astype(np.float32)
        emb = _l2_normalize(emb)
        faiss_idx = build_faiss_index(emb)
        bm25, tokenized = build_bm25_index(SYNTHETIC_TEXTS)
        return chunks, faiss_idx, bm25, tokenized, emb

    def _run_search(self, query_text: str, emb: np.ndarray, chunks, faiss_idx, bm25, tokenized,
                    top_k=3, alpha=0.7, chunk_type_filter=None):
        """Run hybrid_search with a pre-computed query vector (skip embedding)."""
        import faiss as _faiss
        from signal.grounding.indexer import _min_max_normalize, _tokenize, RetrievalResult

        n = len(chunks)
        # Use BM25 as proxy for "dense" in unit tests (no Vertex AI)
        query_tokens = _tokenize(query_text)
        sparse_raw = np.array(bm25.get_scores(query_tokens), dtype=np.float32)

        # Synthetic dense: first vector from emb
        query_vec = emb[0:1]
        dense_scores_raw, indices_arr = faiss_idx.search(query_vec, n)
        dense_by_chunk = np.zeros(n, dtype=np.float32)
        for rank_i, chunk_i in enumerate(indices_arr[0]):
            dense_by_chunk[chunk_i] = dense_scores_raw[0][rank_i]

        dense_norm = _min_max_normalize(dense_by_chunk)
        sparse_norm = _min_max_normalize(sparse_raw)
        combined = alpha * dense_norm + (1 - alpha) * sparse_norm

        if chunk_type_filter:
            for i, c in enumerate(chunks):
                if c.chunk_type != chunk_type_filter:
                    combined[i] = 0.0

        sorted_i = np.argsort(combined)[::-1][:top_k]
        return [
            RetrievalResult(
                chunk=chunks[i], score=float(combined[i]),
                dense_score=float(dense_norm[i]), sparse_score=float(sparse_norm[i]),
            )
            for i in sorted_i if combined[i] > 0
        ]

    def test_returns_top_k(self):
        chunks, faiss_idx, bm25, tokenized, emb = self._setup()
        results = self._run_search("fentanyl overdose", emb, chunks, faiss_idx, bm25, tokenized, top_k=3)
        assert len(results) <= 3

    def test_results_are_retrieval_result(self):
        from signal.grounding.indexer import RetrievalResult
        chunks, faiss_idx, bm25, tokenized, emb = self._setup()
        results = self._run_search("morphine pain", emb, chunks, faiss_idx, bm25, tokenized)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_scores_in_unit_interval(self):
        chunks, faiss_idx, bm25, tokenized, emb = self._setup()
        results = self._run_search("naloxone reversal", emb, chunks, faiss_idx, bm25, tokenized)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_sorted_descending(self):
        chunks, faiss_idx, bm25, tokenized, emb = self._setup()
        results = self._run_search("buprenorphine treatment", emb, chunks, faiss_idx, bm25, tokenized, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_type_filter_restricts(self):
        """Type filter returns only matching chunk types."""
        from signal.grounding.indexer import _min_max_normalize, _tokenize, RetrievalResult
        # Build chunks with mixed types
        from signal.grounding.indexer import ChunkRecord
        mixed_chunks = [
            ChunkRecord(0, "a.txt", "pharmacology", "fentanyl", None, SYNTHETIC_TEXTS[0], 10),
            ChunkRecord(1, "b.txt", "safety", None, None, SYNTHETIC_TEXTS[1], 10),
            ChunkRecord(2, "c.txt", "pharmacology", "naloxone", None, SYNTHETIC_TEXTS[2], 10),
        ]
        import faiss as _faiss
        from signal.grounding.indexer import build_faiss_index, build_bm25_index, _l2_normalize
        rng = np.random.default_rng(1)
        emb = _l2_normalize(rng.random((3, 8)).astype(np.float32))
        fi = build_faiss_index(emb)
        bm25_, tok_ = build_bm25_index([c.text for c in mixed_chunks])
        results = self._run_search(
            "fentanyl", emb, mixed_chunks, fi, bm25_, tok_,
            chunk_type_filter="pharmacology"
        )
        for r in results:
            assert r.chunk.chunk_type == "pharmacology"

    def test_min_max_normalize_range(self):
        from signal.grounding.indexer import _min_max_normalize
        arr = np.array([1.0, 3.0, 5.0, 2.0], dtype=np.float32)
        normed = _min_max_normalize(arr)
        assert float(normed.min()) >= 0.0
        assert float(normed.max()) <= 1.0 + 1e-6

    def test_min_max_normalize_uniform_input(self):
        """Uniform input should not produce NaN (epsilon guard)."""
        from signal.grounding.indexer import _min_max_normalize
        arr = np.ones(5, dtype=np.float32)
        normed = _min_max_normalize(arr)
        assert not np.any(np.isnan(normed))


# ── TestHybridRetriever ───────────────────────────────────────────────────────

class TestHybridRetriever:
    """Integration tests that load real chunks and use SBERT embeddings."""

    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory):
        """Build a HybridRetriever with SBERT (no Vertex AI) into a temp dir."""
        import importlib
        tmp = tmp_path_factory.mktemp("indices")
        from signal.grounding.indexer import HybridRetriever
        r = HybridRetriever(
            faiss_path=tmp / "faiss.bin",
            bm25_path=tmp / "bm25.pkl",
            metadata_path=tmp / "meta.json",
        )
        return r

    def test_chunk_count_is_at_least_84(self, retriever):
        assert retriever.chunk_count >= 84

    def test_embedding_dim_is_positive(self, retriever):
        assert retriever.embedding_dim > 0

    def test_query_returns_list(self, retriever):
        results = retriever.query("fentanyl overdose", top_k=5)
        assert isinstance(results, list)

    def test_query_top_k_respected(self, retriever):
        results = retriever.query("morphine pain management", top_k=3)
        assert len(results) <= 3

    def test_fentanyl_query_finds_fentanyl_chunk(self, retriever):
        results = retriever.query("fentanyl pharmacology potency", top_k=10)
        filenames = [r.chunk.filename for r in results]
        assert any("fentanyl" in f for f in filenames), (
            f"Expected fentanyl chunk in results, got: {filenames}"
        )

    def test_signals_query_finds_signals_chunk(self, retriever):
        results = retriever.query("FAERS adverse events signal morphine", top_k=10)
        chunk_types = [r.chunk.chunk_type for r in results]
        assert "faers_signals" in chunk_types

    def test_type_filter_pharmacology(self, retriever):
        results = retriever.query("opioid receptor binding", top_k=5, chunk_type_filter="pharmacology")
        for r in results:
            assert r.chunk.chunk_type == "pharmacology"

    def test_load_from_disk(self, retriever, tmp_path_factory):
        """Loading pre-built indices should return same results as building."""
        from signal.grounding.indexer import HybridRetriever
        r2 = HybridRetriever(
            faiss_path=retriever._faiss_path,
            bm25_path=retriever._bm25_path,
            metadata_path=retriever._metadata_path,
        )
        assert r2.chunk_count == retriever.chunk_count
        assert r2.embedding_dim == retriever.embedding_dim

    @pytest.mark.skipif(not HAS_VERTEX, reason="No Vertex AI credentials")
    def test_vertex_query_returns_results(self, tmp_path_factory):
        """Vertex AI integration test — runs with credentials, accepts SBERT fallback."""
        from signal.grounding.indexer import HybridRetriever
        from signal.config import EMBEDDING_DIM, FALLBACK_EMBEDDING_DIM
        tmp = tmp_path_factory.mktemp("vertex_indices")
        r = HybridRetriever(
            faiss_path=tmp / "faiss.bin",
            bm25_path=tmp / "bm25.pkl",
            metadata_path=tmp / "meta.json",
        )
        results = r.query("fentanyl respiratory depression overdose", top_k=5)
        assert len(results) > 0
        # Accept either Vertex AI (768) or SBERT fallback (384) dim
        assert r.embedding_dim in (EMBEDDING_DIM, FALLBACK_EMBEDDING_DIM)
