"""
Vector store: turn chunks into vectors and support similarity search over them.

Embedding backend: sentence-transformers (BAAI/bge-small-en-v1.5), local and
free, no API key needed. BGE models are asymmetric: queries are encoded with
an instruction prefix ("Represent this sentence for searching relevant
passages: ") while passage/chunk text is encoded plain -- this is documented
BAAI usage guidance and meaningfully improves retrieval quality over encoding
both sides the same way.

Similarity search backend: FAISS (`IndexFlatIP`), an exact (non-approximate)
inner-product index. Embeddings are L2-normalized before indexing, which makes
inner product mathematically equivalent to cosine similarity -- same scores
as the in-memory sklearn cosine_similarity approach this replaced, just via a
real vector database. Exact search is used rather than an approximate index
(e.g. IndexHNSWFlat) because the corpus size here (~3,300 chunks) is small
enough that exact search is still fast; swap the index type if the corpus
grows to the point where exact search becomes the bottleneck.

Retrieval is a two-stage retrieve-then-rerank pipeline, still with no
lexical/keyword matching layer of any kind (both stages are neural embedding
models, consistent with the brief's "TF-IDF vectors -> Real embeddings"
framing -- a cross-encoder is a *replacement* for lexical matching, not a
BM25/keyword layer bolted back on):

  Stage 1 (recall): the bi-encoder above retrieves a wide candidate pool
  (RERANK_CANDIDATE_POOL chunks) by cosine similarity. Cheap and fast, but a
  bi-encoder embeds query and passage independently, so short/acronym/title
  queries ("What is HNSW?", "What is the Attention Is All You Need paper
  about?") often don't rank the right passage first -- see EVALUATION.md.

  Stage 2 (precision): a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
  re-scores each (query, passage) pair *jointly* -- the query and passage
  attend to each other in one forward pass, which is far more discriminative
  than comparing two independently-produced vectors. This both fixes the
  ordering (the right passage for a title/acronym query now surfaces) and
  yields a relevance score that actually separates in-corpus from
  out-of-corpus queries, which raw bi-encoder cosine does not (measured: raw
  cosine overlaps completely between the two populations, ~0.54-0.65 for
  both, so no cosine threshold can tell them apart; the cross-encoder score
  separates them cleanly -- see EVALUATION.md). query() returns that rerank
  relevance (a sigmoid of the cross-encoder logit, in [0, 1]) as the score,
  which is what app.py's relevance gate and badges now use.

Earlier revisions tried lexical fixes for the bare-acronym gap -- a hardcoded
acronym dictionary, a BM25 hybrid, a narrower exact-acronym-token boost -- and
all three were removed as keyword-matching layers the brief's framing argues
against. Reranking solves most of the same gap without any of that: it stays
purely neural. Two residual hard cases remain (acronyms like FAISS/ScaNN that
barely appear in their own paper's text) and are documented in EVALUATION.md
rather than patched with a keyword layer.

query()'s signature (returns List[(Chunk, score)]) is unchanged; only the
meaning of `score` changed, from bi-encoder cosine to rerank relevance.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F401  (import before faiss)

# faiss and torch (pulled in by sentence_transformers) each bundle their own
# OpenMP runtime. On macOS, importing faiss before torch has been observed to
# segfault during model load -- importing sentence_transformers first avoids it.
import faiss

from .ingest import Chunk

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder reranker (stage 2). Ships inside sentence-transformers (no new
# pip dependency); ~80MB, downloaded once from Hugging Face then cached locally.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Persisted embedding/index cache. The bi-encoder encode pass (~18-27s for
# ~3,300 chunks) is by far the slowest part of startup and is recomputed on
# every Streamlit process restart (@st.cache_resource only helps within one
# running process). We cache the built FAISS index to disk keyed by a
# fingerprint of the embedding model + the exact chunk texts in order, so an
# unchanged corpus loads the index in milliseconds instead of re-embedding.
# Any change to the corpus, chunking, or embedding model changes the
# fingerprint and triggers a clean rebuild -- so the cache can never go stale.
# Gitignored (see .gitignore); safe to delete by hand to force a rebuild.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".index_cache"

# How many bi-encoder candidates to rerank. Wide enough that the correct
# passage for a weak-signal (acronym/title) query is almost always somewhere
# in the pool even when it isn't ranked first by cosine, small enough that
# reranking stays a few tens of ms. Exact search over ~3,300 chunks makes the
# recall stage effectively free, so the pool costs only the extra rerank pairs.
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
        exactly this order -- i.e. index position i still corresponds to
        chunks[i], which is what makes reusing the index safe.
        """
        h = hashlib.sha256()
        h.update(EMBED_MODEL.encode("utf-8"))
        h.update(b"\0")
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and index it in a FAISS IndexFlatIP.

        The built index is cached to disk (keyed by _fingerprint) and reused on
        the next process start if the corpus is unchanged, skipping the slow
        embedding pass entirely.
        """
        self.chunks = chunks
        texts = [c.text for c in chunks]
        cache_file = CACHE_DIR / f"index_{self._fingerprint(texts)}.faiss"

        if cache_file.exists():
            try:
                self.index = faiss.read_index(str(cache_file))
                if self.index.ntotal == len(chunks):
                    return  # cache hit: aligned index loaded, skip embedding
                # Size mismatch (should be impossible given the fingerprint) --
                # fall through and rebuild rather than trust a misaligned index.
            except Exception:
                pass  # unreadable/corrupt cache -> rebuild cleanly

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        ).astype(np.float32)
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(cache_file))
        except Exception:
            pass  # caching is a best-effort speedup; never fail build over it

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, rerank_relevance) pairs for a query string.

        Two stages: (1) bi-encoder cosine retrieves RERANK_CANDIDATE_POOL
        candidates, (2) the cross-encoder re-scores each and the top_k by that
        score are returned. `score` is the rerank relevance -- a sigmoid of the
        cross-encoder logit, in [0, 1] -- not bi-encoder cosine. The query is
        embedded with the BGE instruction prefix for stage 1 (asymmetric
        bi-encoder), but passed *raw* to the cross-encoder in stage 2, which is
        trained on plain (query, passage) pairs with no such prefix.
        """
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before query().")

        # Stage 1: wide bi-encoder recall.
        query_vec = self.model.encode(
            [QUERY_INSTRUCTION + query_text], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)
        pool = min(RERANK_CANDIDATE_POOL, len(self.chunks))
        _, indices = self.index.search(query_vec, pool)
        candidates = [self.chunks[i] for i in indices[0] if i != -1]
        if not candidates:
            return []

        # Stage 2: cross-encoder rerank. predict() returns raw logits for this
        # single-label model; sigmoid maps them to an interpretable [0, 1]
        # relevance the gate and badges in app.py can threshold on.
        logits = np.asarray(
            self.reranker.predict([[query_text, c.text] for c in candidates]),
            dtype=np.float32,
        )
        relevance = 1.0 / (1.0 + np.exp(-logits))
        order = np.argsort(relevance)[::-1][:top_k]
        return [(candidates[i], float(relevance[i])) for i in order]
