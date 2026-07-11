# RAG-Based AI Search System

A Retrieval-Augmented Generation search system over a corpus of AI/ML research
papers: ask a question, get an answer grounded in and cited from the actual
papers, with visible sources and similarity scores. Built for CS382's final
project.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

streamlit run app.py
```

If you want LLM-generated answers ("llm" answer mode), you need a free
Google API key for the Gemini API. The model is selectable from a sidebar
dropdown in LLM mode (default `gemini-2.5-flash`; see the model list below):

```bash
cp .env.example .env   # then edit .env and paste in your key
```

Sign up at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(no credit card required for the free tier). `rag/generate.py` loads `.env`
automatically via `python-dotenv` — no need to `export` it in your shell.
`.env` is gitignored, so your key is never committed. Without a key, the app
still works fully in "extractive" mode (no LLM calls at all); "llm" mode
degrades gracefully to the same extractive output with an explanatory
message if the key is missing/invalid or the API call fails, rather than
crashing.

Open the URL Streamlit prints (usually `http://localhost:8501`) and ask a
question like _"How does the attention mechanism work in Transformers?"_

**First run notes:**

- Downloads the `BAAI/bge-small-en-v1.5` embedding model (~130MB) and the
  `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (~80MB) from Hugging Face —
  needs internet once, then both cached locally. The reranker ships inside
  `sentence-transformers` (already a dependency), so this is a model download,
  not a new pip package.
- Installing `sentence-transformers` pulls in `torch` transitively — a
  noticeably heavier install than a TF-IDF-only baseline (a few hundred MB).
- Building the index (loading ~23 PDFs, chunking, embedding ~3,300 chunks)
  takes well under a minute on first load (embedding alone is ~18s on Apple
  Silicon's MPS backend — see Known Limitations); Streamlit's
  `@st.cache_resource` means this only happens once per running process.
- PDF text is extracted with PyMuPDF (`fitz`), which handles all 23 papers in
  this corpus cleanly (an earlier `pypdf`-based version of this loader
  intermittently failed on `colbert.pdf` on some setups). If a future PDF
  still fails to parse for any reason, it's handled gracefully —
  `rag/ingest.py`'s `load_pdf_documents` catches the exception, logs a
  `[ingest] WARNING: skipping unreadable PDF ...` line, and continues with
  the rest of the corpus rather than crashing ingestion.

## Corpus

23 AI/ML research papers, fetched as PDFs from arXiv's public endpoint (no
auth required) via `scripts/fetch_papers.py`, covering four themes — chosen
so the corpus is literally "papers about the kind of system this project
builds":

- **Foundational**: Attention Is All You Need (1706.03762), BERT (1810.04805), word2vec (1301.3781), Sentence-BERT (1908.10084), RoBERTa (1907.11692), GPT-3 (2005.14165)
- **Retrieval-augmented generation**: RAG (2005.11401), REALM (2002.08909), Dense Passage Retrieval (2004.04906), In-Context RALM (2302.00083), Self-RAG (2310.11511), RAG Survey (2312.10997)
- **Vector search / ANN**: HNSW (1603.09320), FAISS (1702.08734), ColBERT (2004.12832), ScaNN (1908.10396), ANCE (2007.00808)
- **LLM / alignment**: InstructGPT (2203.02155), LLaMA (2302.13971), Chain-of-Thought (2201.11903), Constitutional AI (2212.08073), GPT-4 Technical Report (2303.08774), Lost in the Middle (2307.03172)

Used for coursework/educational purposes only. `data/sample_docs/` (the
original 4-file placeholder corpus) is kept around as a lightweight fallback.

## Architecture

```
PDF/txt files                    data/papers/*.pdf and *.txt
     │
     ▼
Ingest & chunk                   rag/ingest.py: LangChain loaders → RecursiveCharacterTextSplitter
     │                           (~120 words/chunk, 20-word overlap). Dynamically fetches real
     │                           metadata. Creates one synthetic "metadata card" per paper
     │                           (abstract + title + slug) to improve acronym/author matching.
     ▼
Embed                            rag/embed_store.py: BAAI/bge-small-en-v1.5 (local)
     │
     ▼
Vector store                     FAISS IndexFlatIP over L2-normalized embeddings (exact cosine
     │                           similarity search over ~3,300 chunks).
     ▼
Retrieve (recall)                VectorStore.query() Stage 1: pure dense search via FAISS
     │                           fetches a wide candidate pool of 40 chunks.
     ▼
Rerank (precision)               VectorStore.query() Stage 2: ms-marco-MiniLM-L-6-v2 cross-encoder
     │                           re-scores pairs. Fixes bi-encoder ordering and provides absolute
     │                           relevance scores.
     ▼
Relevance filter                 app.py: Drops results below MIN_RELEVANCE_SCORE = 0.10.
     │                           Prevents hallucinations on out-of-corpus queries.
     ▼
Generate                         rag/generate.py: "extractive" or "llm".
     │                           Streams answer with strict system prompt for grounding,
     │                           citations, and jailbreak resistance. Auto-fallbacks on error.
     ▼
Interface                        app.py: Native Streamlit. Live streaming text, expandable source
                                 cards with metadata/relevance badges, and sidebar controls.
```

## Project structure

```
final_project_starter/
├── app.py                    # Streamlit interface
├── .env.example               # copy to .env and add your GOOGLE_API_KEY
├── requirements.txt
├── EVALUATION.md              # 10 test queries + retrieval/generation write-up
├── scripts/
│   └── fetch_papers.py       # one-off: downloads the arXiv corpus
├── data/
│   ├── papers/                # 23 arXiv PDFs (the real corpus used by app.py)
│   │   └── _manifest.json     # filename slug -> arxiv_id, written by fetch_papers.py
│   ├── arxiv_metadata_cache.json  # cached title/authors/year lookups (auto-created)
│   └── sample_docs/           # original 4-file placeholder corpus (fallback)
└── rag/
    ├── ingest.py              # LangChain loaders + RecursiveCharacterTextSplitter
    ├── metadata.py            # dynamic arXiv ID -> title/authors/year lookup + cache
    ├── embed_store.py         # BGE bi-encoder recall + FAISS + cross-encoder rerank (both neural, no keyword layer)
    └── generate.py            # extractive / cloud Gemini answer generation (selectable model), system prompt
```

## Evaluation

See [EVALUATION.md](EVALUATION.md) for the full write-up: 10 test queries
(specific, cross-paper synthesis, and one deliberately out-of-corpus),
retrieved sources and rerank scores, the relevance-threshold calibration
data, LLM-mode groundedness checks, and jailbreak-resistance testing.

## Known limitations

- **Chunking** uses LangChain's `RecursiveCharacterTextSplitter` (paragraph →
  line → word → character backoff), which guarantees every chunk fits the
  ~120-word target but can still cut a boundary mid-sentence.
- **PDF text extraction** (PyMuPDF/`fitz` via LangChain's `PyMuPDFLoader`)
  still degrades on multi-column layouts, inline math, and figures/tables —
  some chunks contain garbled or out-of-order text. Line-wrap hyphenation
  ("the for-\nmer two") is fixed by `rag/ingest.py:_dehyphenate()`; inline
  footnote markers are left as-is (needs layout/bounding-box analysis to fix
  properly, which isn't worth the complexity for a cosmetic issue). A PDF that
  fails to parse is skipped with a logged warning rather than crashing
  ingestion.
- **Persisted embedding/index cache** (`data/.index_cache/`, gitignored) —
  keyed by a fingerprint of the embedding model + every chunk text, so an
  unchanged corpus loads the FAISS index in milliseconds instead of
  re-embedding ~3,300 chunks. Any corpus/chunking/model change invalidates the
  fingerprint and forces a clean rebuild.
- **`MIN_RELEVANCE_SCORE = 0.10`** gates on the **cross-encoder rerank score**
  (a sigmoid of the reranker logit, 0–1) rather than bi-encoder cosine — raw
  cosine cannot separate in-corpus from out-of-corpus queries (both land in
  the same ~0.54–0.65 band), while the rerank score gives a wide, comfortable
  margin (genuinely out-of-corpus queries score ≈0.00, real queries score
  ≥0.6; see EVALUATION.md for the current calibration data). It's still a
  fixed, corpus-calibrated threshold, not an adaptive one. One documented edge
  case survives and is handled a layer deeper: "What is the capital of
  France?" scores ~0.95 because the Self-RAG paper quotes that exact phrase as
  a worked example, so retrieval genuinely contains it — but in "llm" mode the
  grounded system prompt correctly describes what the Self-RAG passage _says_
  rather than answering "Paris" from world knowledge (verified live). This
  threshold is the sole relevance decision in "extractive" mode, which has no
  LLM to make that call.
- **Bare-acronym / paper-title queries are handled by two layers**: the
  cross-encoder reranker (scores a `(query, passage)` pair jointly instead of
  comparing independent vectors) fixes most of them, and a per-document
  **metadata card** (`rag/ingest.py:_metadata_card`, one synthetic chunk per
  paper stating its title, common name, authors, and opening text) covers the
  handful of acronyms that barely appear in their own paper's body text (e.g.
  ScaNN). Both stages stay purely neural — no BM25/keyword layer — consistent
  with the brief's "TF-IDF vectors → Real embeddings" framing. One cosmetic
  residual: for a few papers whose title text overlaps with a related paper
  (e.g. "What is BERT?" vs. Sentence-BERT, "What is FAISS?" vs. a
  FAISS-mentioning DPR chunk), the correct paper lands in the top few results
  but isn't always ranked strictly first — the answer is still correct, just
  not perfectly ordered.
- **Cross-encoder reranking is single-stage, not iterative** — one bi-encoder
  recall pass (`RERANK_CANDIDATE_POOL` cosine candidates) followed by one
  cross-encoder rerank, no multi-hop or query-decomposition step. `app.py`'s
  `LLM_CONTEXT_K = 8` widens the context fed to "llm" mode beyond the UI's
  displayed `top_k` so most 2–3-paper synthesis questions land both papers in
  context, but a question needing 4–5 papers at once can still under-retrieve
  at small `top_k`.
- **`IndexFlatIP` is exact (brute-force) search, not approximate** — the
  right choice at ~3,300 chunks, but doesn't demonstrate approximate-nearest-
  neighbor indexing (e.g. `IndexHNSWFlat`), which a much larger corpus would
  need.
- **The system prompt (`rag/generate.py:SYSTEM_PROMPT`) reduces jailbreak
  susceptibility; it does not eliminate it.** No prompt makes any model
  unjailbreakable in an absolute sense — see EVALUATION.md's "Security
  testing" section for the adversarial prompts tested against it (a mixed
  legitimate-question + jailbreak-attempt prompt sometimes triggers a full
  refusal rather than a partial answer — a safe failure, but a real UX cost).
- **Real metadata is looked up dynamically** (`rag/metadata.py`), not
  hardcoded — `data/papers/_manifest.json` (written by
  `scripts/fetch_papers.py`) is checked first, then arXiv's own watermark ID
  extracted from the PDF's first-page text (only page 1, to avoid picking up
  an ID cited in the paper's own bibliography) via arXiv's public API.
  Results are cached to `data/arxiv_metadata_cache.json`. A PDF with no
  detectable arXiv ID falls back to a filename-derived title with no
  authors/year/arXiv link.
