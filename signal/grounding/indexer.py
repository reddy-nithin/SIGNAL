"""
SIGNAL — FAISS/BM25 Hybrid Retrieval Indexer
=============================================
Loads 58 knowledge chunks, embeds them with Vertex AI (fallback: SBERT),
builds a FAISS dense index + BM25 sparse index, and supports hybrid search
with configurable dense/sparse weighting.

Public API:
    HybridRetriever  — stateful facade (load-or-build on first use)
    load_chunks()    — read .txt files + manifest metadata
    hybrid_search()  — score-fused retrieval over both indices
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from signal.config import (
    BM25_INDEX_PATH,
    CHUNK_METADATA_PATH,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FALLBACK_EMBEDDING_DIM,
    FALLBACK_EMBEDDING_MODEL,
    KNOWLEDGE_CHUNKS_DIR,
    MANIFEST_PATH,
    VERTEX_LOCATION,
    VERTEX_PROJECT_ID,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkRecord:
    """One knowledge chunk with its manifest metadata and raw text."""
    index: int
    filename: str
    chunk_type: str
    drug_name: Optional[str]
    rxcui: Optional[str]
    text: str
    token_estimate: int


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval hit with combined and component scores."""
    chunk: ChunkRecord
    score: float        # hybrid combined score [0, 1]
    dense_score: float  # normalized FAISS cosine similarity
    sparse_score: float # normalized BM25 score


# ── Chunk Loading ─────────────────────────────────────────────────────────────

def load_chunks(
    chunks_dir: Path = KNOWLEDGE_CHUNKS_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> list[ChunkRecord]:
    """Load all knowledge chunks from disk using manifest for metadata.

    Args:
        chunks_dir: Directory containing .txt chunk files.
        manifest_path: Path to manifest.json describing each chunk.

    Returns:
        Ordered list of ChunkRecord (one per manifest entry).

    Raises:
        FileNotFoundError: If manifest or a chunk file is missing.
        ValueError: If a chunk file is empty.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    records: list[ChunkRecord] = []
    for i, entry in enumerate(manifest["chunks"]):
        chunk_path = chunks_dir / entry["filename"]
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        text = chunk_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Chunk file is empty: {chunk_path}")

        records.append(ChunkRecord(
            index=i,
            filename=entry["filename"],
            chunk_type=entry["type"],
            drug_name=entry.get("drug_name"),
            rxcui=entry.get("rxcui"),
            text=text,
            token_estimate=entry.get("token_estimate", 0),
        ))

    return records


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_vertex(
    texts: list[str],
    task_type: str,
    batch_size: int,
) -> np.ndarray:
    """Embed texts via Vertex AI text-embedding-004."""
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in batch]
        result = model.get_embeddings(inputs)
        all_embeddings.extend([r.values for r in result])

    return np.array(all_embeddings, dtype=np.float32)


_sbert_model: object = None  # module-level singleton used outside Streamlit context


def get_sbert_model():
    """Return the SBERT model singleton.

    Outside Streamlit: uses module-level global, loaded once per process.
    Inside Streamlit: callers should wrap this with @st.cache_resource at the
    call site so the model object is reused across reruns.
    """
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
    return _sbert_model


def _embed_sbert(texts: list[str]) -> np.ndarray:
    """Embed texts via local SentenceTransformer (fallback)."""
    model = get_sbert_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
    return vectors / norms


def embed_texts(
    texts: list[str],
    batch_size: int = 5,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> tuple[np.ndarray, int]:
    """Embed a list of texts, returning L2-normalized vectors.

    Uses Vertex AI when GCP_PROJECT_ID is configured; otherwise goes straight to
    local SBERT to avoid a slow failed-attempt on every call.

    Args:
        texts: Strings to embed.
        batch_size: Vertex AI batch size (max 5 for text-embedding-004).
        task_type: "RETRIEVAL_DOCUMENT" for chunks, "RETRIEVAL_QUERY" for queries.

    Returns:
        (embeddings, dim) — float32 array of shape (len(texts), dim),
        and the embedding dimension used.
    """
    if VERTEX_PROJECT_ID:
        try:
            vecs = _embed_vertex(texts, task_type=task_type, batch_size=batch_size)
            logger.info("Embedded %d texts via Vertex AI (dim=%d)", len(texts), EMBEDDING_DIM)
            return _l2_normalize(vecs), EMBEDDING_DIM
        except Exception as exc:
            warnings.warn(
                f"Vertex AI embedding failed ({exc!r}), falling back to SBERT.",
                stacklevel=2,
            )

    vecs = _embed_sbert(texts)
    logger.info("Embedded %d texts via SBERT (dim=%d)", len(texts), FALLBACK_EMBEDDING_DIM)
    return _l2_normalize(vecs), FALLBACK_EMBEDDING_DIM


def embed_query(text: str) -> tuple[np.ndarray, int]:
    """Embed a single query string with RETRIEVAL_QUERY task type."""
    vecs, dim = embed_texts([text], task_type="RETRIEVAL_QUERY")
    return vecs[0:1], dim  # shape (1, dim)


# ── FAISS Index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS IndexFlatIP from L2-normalized embeddings.

    IndexFlatIP with L2-normalized vectors computes cosine similarity.
    Brute-force is appropriate for 58 chunks (microseconds per query).

    Returns:
        faiss.IndexFlatIP
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, path: Path) -> None:
    import faiss
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path):
    import faiss
    return faiss.read_index(str(path))


# ── BM25 Index ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization — simple but avoids mangling pharmacological terms."""
    return text.lower().split()


def build_bm25_index(texts: list[str]):
    """Build a BM25Okapi index from a list of document strings.

    Returns:
        (bm25, tokenized_corpus) tuple.
    """
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def save_bm25_index(bm25, tokenized_corpus: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)


def load_bm25_index(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["tokenized_corpus"]


# ── Chunk Metadata Persistence ────────────────────────────────────────────────

def save_chunk_metadata(chunks: list[ChunkRecord], path: Path) -> None:
    """Save chunk metadata (without text) to JSON for fast re-loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "index": c.index,
            "filename": c.filename,
            "chunk_type": c.chunk_type,
            "drug_name": c.drug_name,
            "rxcui": c.rxcui,
            "token_estimate": c.token_estimate,
        }
        for c in chunks
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_chunk_metadata(
    path: Path,
    chunks_dir: Path = KNOWLEDGE_CHUNKS_DIR,
) -> list[ChunkRecord]:
    """Re-hydrate ChunkRecords from saved metadata + fresh text from disk."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    chunks: list[ChunkRecord] = []
    for r in records:
        text = (chunks_dir / r["filename"]).read_text(encoding="utf-8").strip()
        chunks.append(ChunkRecord(
            index=r["index"],
            filename=r["filename"],
            chunk_type=r["chunk_type"],
            drug_name=r["drug_name"],
            rxcui=r["rxcui"],
            text=text,
            token_estimate=r["token_estimate"],
        ))
    return chunks


# ── Hybrid Search ─────────────────────────────────────────────────────────────

def _min_max_normalize(scores: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize scores to [0, 1] via min-max scaling."""
    min_s, max_s = scores.min(), scores.max()
    return (scores - min_s) / (max_s - min_s + eps)


def hybrid_search(
    query_text: str,
    chunks: list[ChunkRecord],
    faiss_index,
    bm25,
    tokenized_corpus: list[list[str]],
    top_k: int = 5,
    alpha: float = 0.7,
    chunk_type_filter: Optional[str] = None,
) -> list[RetrievalResult]:
    """Score-fused hybrid search over FAISS (dense) and BM25 (sparse).

    Args:
        query_text: Natural language query string.
        chunks: Full list of ChunkRecords (must match index order).
        faiss_index: FAISS IndexFlatIP.
        bm25: BM25Okapi instance.
        tokenized_corpus: List of tokenized documents (parallel to chunks).
        top_k: Number of results to return.
        alpha: Weight for dense scores (1-alpha for sparse). Default 0.7.
        chunk_type_filter: If set, only return chunks of this type.

    Returns:
        List of RetrievalResult, sorted by combined score descending.
    """
    n = len(chunks)

    # Dense retrieval — embed query and search all n chunks
    query_vec, _ = embed_query(query_text)
    dense_scores_raw, indices = faiss_index.search(query_vec, n)
    dense_scores_raw = dense_scores_raw[0]  # shape (n,)

    # Re-order dense scores to match chunk index order
    dense_by_chunk = np.zeros(n, dtype=np.float32)
    for rank_i, chunk_i in enumerate(indices[0]):
        dense_by_chunk[chunk_i] = dense_scores_raw[rank_i]

    # Sparse retrieval — BM25 scores for all n chunks
    query_tokens = _tokenize(query_text)
    sparse_scores_raw = np.array(bm25.get_scores(query_tokens), dtype=np.float32)

    # Normalize both to [0, 1]
    dense_norm = _min_max_normalize(dense_by_chunk)
    sparse_norm = _min_max_normalize(sparse_scores_raw)

    # Fuse
    combined = alpha * dense_norm + (1.0 - alpha) * sparse_norm

    # Apply type filter
    if chunk_type_filter is not None:
        for i, chunk in enumerate(chunks):
            if chunk.chunk_type != chunk_type_filter:
                combined[i] = 0.0

    # Sort and take top_k
    sorted_indices = np.argsort(combined)[::-1][:top_k]

    return [
        RetrievalResult(
            chunk=chunks[i],
            score=float(combined[i]),
            dense_score=float(dense_norm[i]),
            sparse_score=float(sparse_norm[i]),
        )
        for i in sorted_indices
        if combined[i] > 0.0
    ]


# ── HybridRetriever (Facade) ──────────────────────────────────────────────────

class HybridRetriever:
    """Stateful retriever that loads or builds FAISS/BM25 indices on first use.

    On construction, checks for persisted indices:
    - If all three persistence files exist and force_rebuild=False, loads from disk.
    - Otherwise, loads chunks, embeds, builds indices, and saves to disk.

    Usage:
        retriever = HybridRetriever()
        results = retriever.query("fentanyl overdose respiratory depression", top_k=5)
    """

    def __init__(
        self,
        chunks_dir: Path = KNOWLEDGE_CHUNKS_DIR,
        manifest_path: Path = MANIFEST_PATH,
        faiss_path: Path = FAISS_INDEX_PATH,
        bm25_path: Path = BM25_INDEX_PATH,
        metadata_path: Path = CHUNK_METADATA_PATH,
        force_rebuild: bool = False,
    ) -> None:
        self._faiss_path = faiss_path
        self._bm25_path = bm25_path
        self._metadata_path = metadata_path

        can_load = (
            not force_rebuild
            and faiss_path.exists()
            and bm25_path.exists()
            and metadata_path.exists()
        )

        if can_load:
            logger.info("Loading persisted indices from %s", faiss_path.parent)
            self._chunks = load_chunk_metadata(metadata_path, chunks_dir)
            self._faiss = load_faiss_index(faiss_path)
            self._bm25, self._tokenized = load_bm25_index(bm25_path)
            self._dim = self._faiss.d
        else:
            logger.info("Building indices from %d chunks in %s", 0, chunks_dir)
            self._chunks = load_chunks(chunks_dir, manifest_path)
            texts = [c.text for c in self._chunks]

            embeddings, self._dim = embed_texts(texts)
            self._faiss = build_faiss_index(embeddings)
            self._bm25, self._tokenized = build_bm25_index(texts)

            save_faiss_index(self._faiss, faiss_path)
            save_bm25_index(self._bm25, self._tokenized, bm25_path)
            save_chunk_metadata(self._chunks, metadata_path)
            logger.info(
                "Indices built and saved: %d chunks, dim=%d", len(self._chunks), self._dim
            )

    def query(
        self,
        text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        chunk_type_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Run hybrid search and return top_k results."""
        return hybrid_search(
            query_text=text,
            chunks=self._chunks,
            faiss_index=self._faiss,
            bm25=self._bm25,
            tokenized_corpus=self._tokenized,
            top_k=top_k,
            alpha=alpha,
            chunk_type_filter=chunk_type_filter,
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def embedding_dim(self) -> int:
        return self._dim
