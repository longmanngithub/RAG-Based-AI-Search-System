"""
Vector Store & Retrieval Pipeline
- Embeddings: BGE-small-en-v1.5 (asymmetric queries/passages).
- Index: FAISS IndexFlatIP (exact cosine similarity via L2 normalization).
- Stage 1 (Recall): Bi-encoder retrieves a broad candidate pool.
- Stage 2 (Precision): Cross-encoder re-scores candidates for better accuracy.
- Output: Returns (Chunk, rerank_score) where score is a [0, 1] relevance probability.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F401  (import before faiss)

import faiss

from .ingest import Chunk

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".index_cache"

RERANK_CANDIDATE_POOL = 40


def _get_device() -> str:
	"""Detect available device: CUDA > XPU > MPS > CPU."""
	if torch.cuda.is_available():
		return "cuda"
	if torch.xpu.is_available():
		return "xpu"
	if torch.backends.mps.is_available():
		return "mps"
	return "cpu"


class VectorStore:
    def __init__(self):
        device = _get_device()
        self.model = SentenceTransformer(EMBED_MODEL, device=device)
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []

    @staticmethod
    def _fingerprint(texts: List[str]) -> str:
        """Stable hash of the embedding model + every chunk text, in order.

        Any change to the corpus, chunking, or model changes this, so a cache
        hit guarantees the on-disk index was built from exactly these texts in
        exactly this order.
        """
        h = hashlib.sha256()
        h.update(EMBED_MODEL.encode("utf-8"))
        h.update(b"\0")
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and index it in a FAISS IndexFlatIP."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        cache_file = CACHE_DIR / f"index_{self._fingerprint(texts)}.faiss"

        if cache_file.exists():
            try:
                self.index = faiss.read_index(str(cache_file))
                if self.index.ntotal == len(chunks):
                    return  
            except Exception:
                pass  

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(cache_file))
        except Exception:
            pass  

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, rerank_relevance) pairs for a query string.

        Two stages: (1) bi-encoder cosine retrieves RERANK_CANDIDATE_POOL
        candidates, (2) the cross-encoder re-scores each and the top_k by that
        score are returned. `score` is the rerank relevance. The query is
        embedded with the BGE instruction prefix for stage 1 (asymmetric
        bi-encoder), but passed *raw* to the cross-encoder in stage 2.
        """
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before query().")

        query_vec = self.model.encode(
            [QUERY_INSTRUCTION + query_text], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)
        pool = min(RERANK_CANDIDATE_POOL, len(self.chunks))
        _, indices = self.index.search(query_vec, pool)
        candidates = [self.chunks[i] for i in indices[0] if i != -1]
        if not candidates:
            return []

        logits = np.asarray(
            self.reranker.predict([[query_text, c.text] for c in candidates]),
            dtype=np.float32,
        )
        relevance = 1.0 / (1.0 + np.exp(-logits))
        order = np.argsort(relevance)[::-1][:top_k]
        return [(candidates[i], float(relevance[i])) for i in order]
