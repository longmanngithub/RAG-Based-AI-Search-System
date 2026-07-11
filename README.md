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
question like *"How does the attention mechanism work in Transformers?"*

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
PDF/txt files                    data/papers/*.pdf
     │
     ▼
Ingest & chunk         rag/ingest.py: load_documents_any() → build_chunk_records()
     │                 LangChain loaders (PyMuPDFLoader, TextLoader) →
     │                 RecursiveCharacterTextSplitter, ~120 words/chunk
     │                 (word-count length_function), 20-word overlap.
     │                 Real title/authors/year/arXiv ID looked up dynamically
     │                 per document (rag/metadata.py: fetch manifest, then
     │                 the PDF's own arXiv watermark), not a filename guess.
     │                 Plus one synthetic "metadata card" chunk per paper
     │                 (_metadata_card): states the title + common name (the
     │                 filename slug) + authors/year + the paper's opening
     │                 (abstract) text, so "What is ScaNN?" / "who wrote BERT?"
     │                 type queries have an explicit chunk to match when the
     │                 acronym is nearly absent from body text -- and enough
     │                 real content for the LLM to describe the paper, not just
     │                 name it
     ▼
Embed                  rag/embed_store.py: VectorStore.build()
     │                 BAAI/bge-small-en-v1.5 (sentence-transformers, local)
     ▼
Vector store           FAISS IndexFlatIP over L2-normalized embeddings
     │                 (inner product == cosine similarity; exact search,
     │                 fast enough at this scale — ~3,300 chunks)
     ▼
Retrieve (recall)      VectorStore.query() stage 1 — query embedded with the
     │                 BGE query-instruction prefix and searched via FAISS for
     │                 a wide candidate pool (RERANK_CANDIDATE_POOL = 40) by
     │                 cosine similarity. Still pure dense, no keyword/lexical
     │                 layer.
     ▼
Rerank (precision)     VectorStore.query() stage 2 — a cross-encoder
     │                 (ms-marco-MiniLM-L-6-v2) re-scores each (query, passage)
     │                 pair jointly and returns the top_k by that score. Fixes
     │                 the ordering for weak-signal acronym/title queries a
     │                 bi-encoder ranks poorly, and yields a relevance score
     │                 that actually separates in-corpus from out-of-corpus
     │                 queries (raw cosine does not). Both stages are neural —
     │                 no lexical/keyword layer of any kind.
     ▼
Relevance filter       app.py: MIN_RELEVANCE_SCORE = 0.10 (on the rerank
     │                 score, not cosine) drops results below the calibrated
     │                 threshold so an out-of-corpus query gets a refusal, not
     │                 a guess. In "llm" mode the grounded system prompt is the
     │                 real final judge (it refuses off-topic context on its
     │                 own); the gate is the sole relevance decision only in
     │                 "extractive" mode
     ▼
Generate               rag/generate.py: generate_answer_stream()
     │                 "extractive" (no dependencies) or "llm" (a cloud Gemini
     │                 model, chosen from a sidebar dropdown -- default
     │                 gemini-2.5-flash; see AVAILABLE_MODELS -- via the Gemini
     │                 API, streamed token-by-token via generate_content_stream,
     │                 max_output_tokens=1500). In "llm" mode the LLM is fed a
     │                 wider reranked context than the UI lists (LLM_CONTEXT_K
     │                 in app.py) so multi-paper synthesis has all needed
     │                 papers. A fixed SYSTEM_PROMPT sent as system_instruction
     │                 enforces scope, citation format, and resistance to
     │                 prompt injection/jailbreak attempts (see EVALUATION.md);
     │                 grounded strictly in the filtered retrieved chunks,
     │                 retries transient errors and falls back to extractive if
     │                 the key is missing/invalid, the call fails, or the model
     │                 returns no text (never a blank answer)
     ▼
Interface              app.py — Streamlit, no custom CSS (first-party
                        components only -- see Known Limitations): search-bar-
                        style query form, answer shown in a bordered container
                        streamed live via st.write_stream(), sources as
                        bordered cards (title, authors/year, st.badge()
                        relevance indicator, arXiv link, expandable excerpt),
                        sidebar settings (top_k, answer mode, LLM model) plus a
                        system-architecture summary, latency display
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
- **PyMuPDF is AGPL-3.0-licensed** (or needs a commercial license from Artifex
  for closed-source use) — a non-issue for coursework, but worth knowing if
  this codebase is ever reused somewhere that can't accept AGPL. `pypdf`
  (BSD-licensed) is the drop-in alternative in `rag/ingest.py:load_pdf_documents`.
- **Embeddings and reranking run on MPS (Apple Silicon GPU)** — `VectorStore`
  sets `device="mps"`, ~3x faster than CPU on this corpus. MPS shares unified
  memory with the rest of the OS; if `RuntimeError: MPS backend out of memory`
  ever appears, switch to `device="cpu"` in `rag/embed_store.py` (costs about
  a minute of one-time index-build latency).
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
  grounded system prompt correctly describes what the Self-RAG passage *says*
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
- **`faiss` must be imported after `torch`/`sentence-transformers`** in
  `rag/embed_store.py` — on macOS, importing it first has been observed to
  segfault during model load, since both bundle their own OpenMP runtime.
- **LLM mode depends on a cloud API call** (a Gemini model, via the Gemini
  API) — trades "works fully offline, zero marginal cost" for needing a
  `GOOGLE_API_KEY`, the free tier's rate limits, and retrieved paper excerpts
  leaving your machine over the network on every "llm"-mode query. **The
  model is user-selectable** from a sidebar dropdown (`AVAILABLE_MODELS` in
  `rag/generate.py`; default `gemini-2.5-flash`). Only two models are offered
  — other Gemini models (`gemini-3.5-flash`, `gemini-2.5-flash-lite`,
  `gemini-2.0-flash`) were tried and are rate-limited too aggressively on the
  free tier to be usable here:

  | Model | Speed | Grounding | Notes |
  |---|---|---|---|
  | `gemini-2.5-flash` (default) | ~3s | strict | verified: correctly declines to answer "Paris" from world knowledge |
  | `gemini-3.1-flash-lite` | ~1–2s | **weaker** | fastest; answered "Paris" for the capital-of-France artifact from world knowledge |

  The free tier also caps total requests per model per day (20/day observed
  for `gemini-2.5-flash`) — the dropdown lets you switch models if one is rate
  limited. **Robustness** (`rag/generate.py`): transient `ServerError`s
  (including 503 under load) are retried up to twice; `ClientError` (bad key,
  quota exhausted, etc.) is not retried, since retrying an auth/quota failure
  just wastes time. If the model completes with no text at all, that's also
  treated as a transient failure and retried, then falls back to extractive —
  so LLM mode never shows a blank answer.
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
- **The frontend uses no custom CSS** — every visual element is a first-party
  Streamlit component (`st.title`, `st.caption`, `st.badge`,
  `st.container(border=True)`, `st.columns`, etc.), so the app automatically
  follows whichever theme Streamlit itself is configured with. Trade-off: the
  relevance badge colors are limited to `st.badge()`'s fixed palette, and
  there's no way to set a custom max content width.
- **There is intentionally no `.streamlit/config.toml`** — setting even one
  `[theme]` key there disables Streamlit's own light/dark auto-detection and
  locks the app to a single fixed appearance (confirmed by testing). The
  search button's red color is Streamlit's own default `primaryColor`, not
  anything this project sets.
