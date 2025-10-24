
"""
retrieval_tool.py

LangGraph-friendly retrieval tool over a ChromaDB collection with:
- Query embedding (uses the same SentenceTransformer model as indexing)
- Initial vector retrieval from Chroma
- MMR diversification
- Optional cross-encoder reranking
- Compact, typed outputs for agent tool-calling

Assumes your ingestion script used:
  - PersistentClient(persist_dir)
  - get_or_create_collection(name=..., embedding_function=SentenceTransformerEmbeddingFunction(model_name=...))
  - Metadata keys: source_path, filename, chunk_index, chunk_count
  - Defaults (from ingest_to_chroma.py):
      persist_dir: ./chroma_store
      collection:  papers
      model:       sentence-transformers/all-MiniLM-L6-v2

Usage (as a library):
    from retrieval_tool import RetrievalTool
    rt = RetrievalTool()
    result = rt.retrieve("What did the paper say about APOE?")

Command-line quick test:
    python retrieval_tool.py --query "example question"
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy is required. Please install with `pip install numpy`.") from e


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RetrievedChunk:
    text: str
    score: float                 # Higher is better after normalization/rerank
    source_path: str
    filename: str
    chunk_index: int
    chunk_count: int
    # Optional raw scores for debugging/tracing
    distance: Optional[float] = None   # Raw vector distance (lower is better if cosine distance)
    rerank_score: Optional[float] = None


@dataclass
class RetrievalTrace:
    query: str
    k_initial: int
    k_after_mmr: int
    k_after_rerank: int
    mmr_lambda: float
    timings_ms: Dict[str, float]
    notes: List[str]


@dataclass
class RetrievalResult:
    contexts: List[RetrievedChunk]   # Top-N chunks chosen for grounding
    trace: RetrievalTrace


# -----------------------------
# Utility functions
# -----------------------------

def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise to enable cosine similarity via dot product."""
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of L2-normalized vectors."""
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return a @ b.T  # (len(a), len(b))


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    lambda_mult: float = 0.5,
    k: int = 10
) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    Returns the indices of selected items from doc_vecs.

    Parameters
    ----------
    query_vec : np.ndarray
        Shape (d,) or (1, d)
    doc_vecs : np.ndarray
        Shape (N, d)
    lambda_mult : float
        Trade-off between relevance and diversity. 0 -> full diversity, 1 -> full relevance.
    k : int
        Number of items to select.
    """
    if doc_vecs.size == 0:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    sim_to_query = _cosine_sim_matrix(doc_vecs, query_vec).reshape(-1)  # (N,)
    # Start with the most relevant
    selected: List[int] = []
    candidates: List[int] = list(range(len(doc_vecs)))

    # Precompute doc-to-doc similarities
    doc_sims = _cosine_sim_matrix(doc_vecs, doc_vecs)  # (N, N)

    k = min(k, len(candidates))
    while len(selected) < k:
        if not selected:
            # pick the most relevant first
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            candidates.remove(idx)
            continue

        # For each candidate, compute MMR score
        mmr_scores = []
        for c in candidates:
            diversity = max(doc_sims[c, selected]) if selected else 0.0
            score = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * diversity
            mmr_scores.append((score, c))
        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected.append(chosen)
        candidates.remove(chosen)

    return selected


# -----------------------------
# Retrieval tool
# -----------------------------

class RetrievalTool:
    """
    Retrieval tool wrapping Chroma + MMR + optional reranking.
    Designed to be registered as a tool in an agent (e.g., LangGraph).
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        collection_name: str = "papers",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model

        # Init Chroma client + collection with the same embedding function used at ingest
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model_name
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Lazy-init reranker (only if used)
        self._reranker = None

    # ------------- public API -------------

    def retrieve(
        self,
        query: str,
        k_initial: int = 30,
        k_mmr: int = 15,
        mmr_lambda: float = 0.5,
        top_n: int = 8,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = True,
    ) -> RetrievalResult:
        """
        End-to-end retrieval with initial vector search, MMR, and optional reranking.

        Returns a RetrievalResult with top contexts and a trace dictionary.
        """
        import time

        t0 = time.time()

        # 1) Embed query
        q_vec = np.array(self.emb_fn([query])[0], dtype=np.float32)

        # 2) Initial vector retrieval from Chroma
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        raw = self.collection.query(
            query_texts=[query],
            n_results=k_initial,
            where=where or {},
            include=include,
        )
        t1 = time.time()

        docs: List[str] = raw.get("documents", [[]])[0]
        metas: List[Dict[str, Any]] = raw.get("metadatas", [[]])[0]
        dists: List[float] = raw.get("distances", [[]])[0]
        # embeddings can be missing depending on Chroma version; guard it
        doc_vecs_list = raw.get("embeddings", [[]])
        doc_vecs: Optional[np.ndarray] = None
        if doc_vecs_list and len(doc_vecs_list[0]) > 0:
            doc_vecs = np.array(doc_vecs_list[0], dtype=np.float32)

        notes: List[str] = []
        if not docs:
            trace = RetrievalTrace(
                query=query,
                k_initial=k_initial,
                k_after_mmr=0,
                k_after_rerank=0,
                mmr_lambda=mmr_lambda,
                timings_ms={
                    "embed_query": (t1 - t0) * 1000.0,
                    "vector_search": 0.0,
                    "mmr": 0.0,
                    "rerank": 0.0,
                    "total": (time.time() - t0) * 1000.0,
                },
                notes=["No documents returned from Chroma."],
            )
            return RetrievalResult(contexts=[], trace=trace)

        # 3) MMR diversification (requires embeddings). If unavailable, fall back to truncation.
        if doc_vecs is not None:
            mmr_indices = mmr(q_vec, doc_vecs, lambda_mult=mmr_lambda, k=min(k_mmr, len(docs)))
        else:
            notes.append("Embeddings not returned by Chroma; skipping MMR (fall back to top-k).")
            mmr_indices = list(range(min(k_mmr, len(docs))))
        t2 = time.time()

        # Build intermediate set after MMR
        mmr_docs = [docs[i] for i in mmr_indices]
        mmr_metas = [metas[i] for i in mmr_indices]
        mmr_dists = [dists[i] for i in mmr_indices]

        # 4) Optional reranking using a cross-encoder
        rerank_scores: Optional[List[float]] = None
        if self.use_reranker:
            try:
                reranker = self._get_reranker()
                pairs = [(query, d) for d in mmr_docs]
                # Higher is better
                rerank_scores = list(map(float, reranker.predict(pairs)))
            except Exception as e:
                notes.append(f"Reranker unavailable or failed ({type(e).__name__}: {e}); using MMR order.")
        t3 = time.time()

        # 5) Final top-N selection
        order = list(range(len(mmr_docs)))
        if rerank_scores is not None:
            order = sorted(order, key=lambda i: rerank_scores[i], reverse=True)

        top_n = min(top_n, len(order))
        chosen = order[:top_n]

        contexts: List[RetrievedChunk] = []
        for idx in chosen:
            meta = mmr_metas[idx] or {}
            contexts.append(
                RetrievedChunk(
                    text=mmr_docs[idx],
                    score=(rerank_scores[idx] if rerank_scores is not None else 1.0 - (mmr_dists[idx] if mmr_dists else 0.0)),
                    source_path=str(meta.get("source_path", "")),
                    filename=str(meta.get("filename", "")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    chunk_count=int(meta.get("chunk_count", -1)),
                    distance=float(mmr_dists[idx]) if mmr_dists else None,
                    rerank_score=float(rerank_scores[idx]) if rerank_scores is not None else None,
                )
            )

        trace = RetrievalTrace(
            query=query,
            k_initial=k_initial,
            k_after_mmr=len(mmr_docs),
            k_after_rerank=len(contexts),
            mmr_lambda=mmr_lambda,
            timings_ms={
                "embed_query": (t1 - t0) * 1000.0,
                "vector_search": (t1 - t0) * 1000.0,  # includes Chroma query call
                "mmr": (t2 - t1) * 1000.0,
                "rerank": (t3 - t2) * 1000.0,
                "total": (time.time() - t0) * 1000.0,
            },
            notes=notes,
        )
        return RetrievalResult(contexts=contexts, trace=trace)

    # ------------- internal -------------

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder  # lazy import
        self._reranker = CrossEncoder(self.reranker_model_name, trust_remote_code=True)
        return self._reranker


# -----------------------------
# CLI for quick testing
# -----------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Test the retrieval tool over a Chroma collection.")
    parser.add_argument("--persist_dir", type=str, default=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_store"))
    parser.add_argument("--collection", type=str, default=os.environ.get("CHROMA_COLLECTION", "papers"))
    parser.add_argument("--embed_model", type=str, default=os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--no_rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--reranker_model", type=str, default=os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    parser.add_argument("--k_initial", type=int, default=30)
    parser.add_argument("--k_mmr", type=int, default=15)
    parser.add_argument("--mmr_lambda", type=float, default=0.5)
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    tool = RetrievalTool(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        use_reranker=not args.no_rerank,
        reranker_model=args.reranker_model,
    )
    result = tool.retrieve(
        query=args.query,
        k_initial=args.k_initial,
        k_mmr=args.k_mmr,
        mmr_lambda=args.mmr_lambda,
        top_n=args.top_n,
    )

    # Pretty print
    print("\n=== Top Contexts ===")
    for i, c in enumerate(result.contexts, 1):
        print(f"[{i}] score={c.score:.4f}  file={c.filename}  idx={c.chunk_index}/{c.chunk_count}")
        print(c.text[:400].replace("\n", " ") + ("..." if len(c.text) > 400 else ""))
        print()

    print("=== Trace ===")
    print(f"Query: {result.trace.query}")
    print(f"k_initial: {result.trace.k_initial}  k_after_mmr: {result.trace.k_after_mmr}  k_after_rerank: {result.trace.k_after_rerank}")
    print(f"mmr_lambda: {result.trace.mmr_lambda}")
    print("timings_ms:", {k: round(v, 2) for k, v in result.trace.timings_ms.items()})
    if result.trace.notes:
        print("notes:", result.trace.notes)


if __name__ == "__main__":
    _cli()
