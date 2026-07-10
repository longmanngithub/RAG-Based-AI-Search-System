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

This project ran generation locally via Ollama earlier in development
specifically to avoid needing an API key at all, but that requires real
local compute — on a machine without much spare RAM, local inference was
slow enough to hurt the live-demo experience. Moving to the cloud trades
"fully offline" for "actually fast regardless of your hardware," at the
cost of needing a key and sending retrieved paper excerpts over the network.

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
retrieved sources and scores, and a discussion of what worked and what
didn't — including a documented retrieval miss (a query naming the FAISS
*paper* doesn't surface it in the top 10 results, despite FAISS itself being
the library powering the vector store) and the relevance-threshold
calibration data.

## Known limitations

- **Chunking uses LangChain's `RecursiveCharacterTextSplitter`**, not a
  from-scratch sentence-aware splitter (an earlier version of this project
  used a hand-rolled regex sentence-splitter instead). The recursive splitter
  is more robust — it guarantees every chunk fits the target size by backing
  off through paragraph → line → word → character separators, which
  eliminated an edge case the old splitter could hit (a single 2,273-word
  chunk observed on reference-list text with no sentence-ending punctuation)
  — but it can still occasionally cut a chunk boundary mid-sentence, unlike
  the old splitter's "never splits mid-sentence" guarantee. A deliberate
  trade-off: fewer size outliers, slightly less semantically clean edges.
- **PDF text extraction** (via PyMuPDF/`fitz`, loaded through LangChain's
  `PyMuPDFLoader`) still degrades on multi-column
  layouts, inline math, and figures/tables — some chunks contain garbled or
  out-of-order text as a result, even though PyMuPDF handles this corpus more
  robustly than the `pypdf`-based loader it replaced (which intermittently
  failed on `colbert.pdf`). Any PDF that still fails to parse is skipped with
  a logged warning rather than crashing ingestion (see First Run Notes above).
  Two specific artifacts were investigated directly: (1) **line-wrap
  hyphenation** ("the for-\nmer two" extracting as "for- mer") is fixed —
  `rag/ingest.py:_dehyphenate()` rejoins a hyphen immediately followed by a
  newline and a lowercase letter, verified against the actual corpus (zero
  such patterns remain post-fix). (2) **inline footnote markers** (e.g. a
  citation URL landing mid-paragraph) are left alone — PyMuPDF has no
  built-in way to separate body text from footnote regions, and doing so
  properly needs bounding-box/font-size layout analysis, which is real added
  complexity for a cosmetic issue that doesn't affect retrieval or generation
  quality. Also tested PyMuPDF's layout-aware `sort=True` extraction mode as
  a possible general fix for column-jumbled text — it made this corpus's
  two-column layout *worse* (interleaved text from both columns mid-sentence
  on at least one page), so it was not adopted.
- **PyMuPDF is AGPL-3.0-licensed** (or requires a commercial license from
  Artifex for closed-source use) — a non-issue for this coursework project,
  but worth knowing if this codebase were ever reused somewhere that couldn't
  accept AGPL. `pypdf` (BSD-licensed, permissive) is the drop-in alternative
  if that ever matters — swap it back into `rag/ingest.py:load_pdf_documents`.
- **Embeddings run on MPS (Apple Silicon GPU), not CPU.** `VectorStore`
  explicitly sets `device="mps"` in `rag/embed_store.py` — measured ~3x
  faster than CPU on this corpus (17.6s vs 52.9s to encode ~3,300 chunks).
  The trade-off: MPS shares unified memory with the rest of the OS, and an
  earlier version of this project forced `device="cpu"` after observing
  `RuntimeError: MPS backend out of memory` on a 16GB Mac with other apps
  open. That error didn't reproduce in later testing, but headroom depends
  on the specific machine and what else is running at the time — if it
  reappears, switch back to `device="cpu"` (costs roughly a minute of
  one-time index-build latency, cached by `@st.cache_resource`, in exchange
  for working regardless of available memory).
- **Persisted embedding/index cache** — the built FAISS index is cached to
  disk (`data/.index_cache/`, gitignored) keyed by a fingerprint of the
  embedding model + the exact chunk texts, so an unchanged corpus loads the
  index in milliseconds on process restart instead of re-embedding ~3,300
  chunks (~18s saved; measured cold 32.7s → warm 15.3s total build, the
  remainder being one-time model loading). Any change to the corpus, chunking,
  or embedding model changes the fingerprint and forces a clean rebuild, so the
  cache can never go stale. `@st.cache_resource` still avoids even this within a
  single running process. (The cross-encoder reranker isn't cached — it's a
  model load, ~1s, not the embedding pass.)
- **Why there's a relevance threshold at all**: the brief's Section 3
  architecture table doesn't list a filtering stage between "Retrieve" and
  "Generate," which raises a fair question — is `MIN_RELEVANCE_SCORE` scope
  creep? It isn't: Section 2's functional requirement #6 is explicit
  ("Graceful failure — when nothing relevant is found, say so instead of
  hallucinating an answer"), independent of what the architecture summary
  table shows. In "extractive" mode there is no LLM to make that judgment —
  the raw top-k chunks are all that's shown, so *something* has to decide
  "nothing relevant" or that mode can never satisfy requirement #6 at all. A
  single float comparison is close to the simplest possible implementation
  of that requirement; the actual complexity this project added (and then
  removed) was the multi-round recalibration and the acronym-specific
  patches chasing edge cases the threshold's existence doesn't itself
  require.
- **`MIN_RELEVANCE_SCORE = 0.10` is a fixed, corpus-calibrated threshold**,
  not an adaptive one — but it now gates on the **cross-encoder rerank score**
  (0–1, a sigmoid of the reranker logit), not on bi-encoder cosine, which is a
  materially more robust signal. History: the cosine-based version was
  recalibrated three times (0.72 → 0.68 → 0.66) chasing a ~0.03-wide gap
  between genuinely-unrelated and genuinely-relevant-but-thin queries, because
  raw cosine simply does not separate those two populations — both land in
  ~0.54–0.65, so *every* cosine threshold either refused real queries ("What
  is HNSW?" at 0.56, "What is the Attention Is All You Need paper about?" at
  0.65) or admitted unrelated ones ("What is the capital of France?" at 0.65).
  Switching the gate to the reranker score replaced that ~0.03 gap with a
  ~0.25-wide one: genuinely out-of-corpus queries score under 0.05, while real
  queries (even weak-signal acronym/title ones) score above ~0.30, so 0.10
  sits in a comfortable margin instead of threading a needle. See EVALUATION.md
  for the full before/after calibration data. One documented edge case
  survives and is *handled a layer deeper*: "what is the capital of France?"
  still scores high (~0.94) because the Self-RAG paper quotes that exact phrase
  as a worked example — so retrieval genuinely does contain it — but in "llm"
  mode the grounded system prompt correctly describes what the Self-RAG passage
  *says about that example* rather than answering "Paris" from world knowledge
  (verified). A gate leak no longer means an ungrounded answer.
- **Bare-acronym / paper-title queries: largely fixed by reranking.** "What is
  HNSW?", "What is DPR?", "What is the paper Attention Is All You Need about?"
  and similar used to score right at or below the old cosine threshold and were
  intermittently refused — a general bi-encoder weakness (short, low-content
  queries embed weakly; confirmed the same gap exists on `all-MiniLM-L6-v2`,
  so a different *embedding* model was never the fix — see EVALUATION.md). The
  cross-encoder rerank stage resolves most of these by scoring the (query,
  passage) pair jointly instead of comparing two independent vectors: across a
  17-query in-corpus battery that previously produced 11 wrong refusals, all 17
  now pass. Earlier lexical fixes for this same gap (a hardcoded
  `ACRONYM_EXPANSIONS` dict, a BM25 hybrid, an exact-acronym-token boost) were
  all removed and deliberately *not* revived — the reranker is a purely neural
  second stage, consistent with the brief's "TF-IDF vectors → Real embeddings"
  framing, and needs no keyword layer. The acronyms that *barely appear in
  their own papers' body text* ("ScaNN" → "Accelerating Large-Scale Inference
  with Anisotropic Vector Quantization"), which even a joint reranker had no
  signal for, are now handled by the **per-document metadata card** (see the
  ingest stage in Architecture): each paper gets one synthetic chunk stating
  its title + common name (the filename slug) + authors, so "What is ScaNN?"
  went from refused (0.000) to answered (0.999, correct paper). The card's
  "common name" is the corpus's own filename slug, not a hand-maintained
  acronym dictionary, so it needs no per-corpus upkeep. One cosmetic residual:
  "What is FAISS?" is answered but a Dense-Passage-Retrieval chunk that
  *mentions* FAISS can still out-rank the FAISS card at position 1 (the card is
  still retrieved within top-k) — it answers correctly, just isn't perfectly
  ordered.
- **Cross-encoder reranking is single-stage, not iterative.** Ranking is a
  bi-encoder recall pass (FAISS cosine over the top `RERANK_CANDIDATE_POOL`)
  followed by one cross-encoder rerank; there is no further multi-hop or
  query-decomposition step, so a synthesis question spanning several papers
  still depends on all the needed papers landing in the reranked top-k. The
  reranker widened this considerably versus pure cosine (e.g. "How does ColBERT
  differ from DPR?" now retrieves both papers' chunks where cosine-top-3
  surfaced only ColBERT), but a question needing 4–5 papers at once can still
  under-retrieve at small `top_k` — raise `top_k` or add multi-query expansion
  if that matters for a given corpus.
- **`IndexFlatIP` is exact (brute-force) search, not approximate** — the
  right choice at ~3,300 chunks, but it doesn't demonstrate the
  approximate-nearest-neighbor indexing (e.g. `IndexHNSWFlat`) that a much
  larger corpus would need; noted here as the natural next upgrade if the
  corpus grows substantially.
- **`faiss` must be imported after `torch`/`sentence-transformers`** in
  `rag/embed_store.py` — on macOS, importing it first has been observed to
  segfault during model load, since both bundle their own OpenMP runtime.
  Already handled (see the import order and comment in that file) — flagged
  here so it isn't accidentally "fixed" by reordering imports later.
- **LLM mode depends on a cloud API call (a Gemini model, via the Gemini
  API).** This trades "works fully offline, zero marginal cost" for "runs on
  Google's hardware" — the real costs are needing a `GOOGLE_API_KEY`, being
  subject to the free tier's rate limits, and retrieved paper excerpts leaving
  your machine over the network on every "llm"-mode query.
  **The model is user-selectable** from a sidebar dropdown in LLM mode
  (`AVAILABLE_MODELS` in `rag/generate.py`; default `gemini-2.5-flash`):

  | Model | Speed | Grounding | Notes |
  |---|---|---|---|
  | `gemini-2.5-flash` (default) | ~3–4s | strict | proven-stable, obeys documents-only rule |
  | `gemini-3.5-flash` | fast when up | strict | strongest, but was intermittently `503`-congested on the free tier at time of writing |
  | `gemini-3.1-flash-lite` | ~1–2s | **weaker** | fastest; answered "Paris" for the capital-of-France artifact from world knowledge |
  | `gemini-2.5-flash-lite` | ~1–2s | **weaker** | same world-knowledge leak observed |
  | `gemini-2.0-flash` | fast | strict | older mainline flash |

  **Grounding vs. speed is a real, measured trade-off** (see EVALUATION.md):
  full flash models obey the strictly-grounded system prompt (on "what is the
  capital of France?", whose phrase appears in the Self-RAG paper but whose
  *answer* does not, they correctly reply "the passages don't state it"),
  while `-lite` models are ~2× faster but answered "Paris" from world
  knowledge — a grounding leak that matters for a documents-only system, which
  is why the default is a full model and the dropdown surfaces the caveat.
  **Why Gemini, not Gemma.** Generation moved local Ollama (`gemma4:e4b`) →
  cloud Gemma 4 (`gemma-4-26b-a4b-it`, MoE) → Gemini Flash. Gemma 4 was dropped
  because it is **unreliable for grounded generation under a token cap**: it
  spends a variable, *uncappable* number of internal reasoning tokens against
  `max_output_tokens` (the API rejects `thinking_budget` for that model with a
  400) and frequently hit the limit before emitting *any* answer text
  (`finish_reason=MAX_TOKENS`, empty output), worst on multi-paper synthesis
  questions — surfacing to the user as a **blank answer**. It was also slow and
  latency-variable (9–70s+) with intermittent `ServerError` 500s.
  **Robustness (`rag/generate.py`):** transient `ServerError`s (incl. 503
  under load) are retried up to twice; `ClientError` (bad key, etc.) is not
  (retrying an auth failure just wastes time). And if the model ever completes
  with *no text* at all, that's now treated like a transient failure —
  retried, then a graceful extractive fallback — so LLM mode **never shows a
  blank answer**. The retries make failures recoverable but can't make a
  congested/rate-limited endpoint fast; a persistently-503 model falls back to
  extractive, which is why other models are one dropdown-click away.
- **The system prompt (`rag/generate.py:SYSTEM_PROMPT`) reduces jailbreak
  susceptibility; it does not eliminate it.** No prompt makes any model
  unjailbreakable in an absolute sense — see EVALUATION.md's "Security
  testing" section for the adversarial prompts actually tried against the
  local Ollama model (all four were resisted, but one showed a real
  trade-off: a mixed legitimate-question + jailbreak-attempt prompt
  sometimes triggers a full refusal rather than a partial answer, a safe
  failure but a UX cost). Re-verify against whichever Gemini model you select
  if that matters for your use case — a different model may behave differently.
- **Real metadata is looked up dynamically (`rag/metadata.py`), not
  hardcoded.** A user pointed out the original approach — a 23-entry table
  keyed by filename — would silently stop working the moment the corpus was
  swapped for different papers. Replaced with: check
  `data/papers/_manifest.json` (written by `scripts/fetch_papers.py`, ground
  truth for exactly which paper each fetched PDF is) first, then fall back to
  extracting arXiv's own watermark ID from the PDF's first-page text and
  querying arXiv's public API. Swap in different arXiv papers (via the fetch
  script or dropped in by hand) and metadata resolves automatically — no
  table to update. Results are cached to `data/arxiv_metadata_cache.json` so
  repeat launches don't re-hit the network. A PDF with no detectable arXiv ID
  (not from arXiv, or `data/sample_docs/`) still falls back to a
  filename-derived title with no authors/year/arXiv link, same as before.
  **Only page 1's text is searched for the watermark, not the whole
  document** — searching everything once picked up an unrelated ID cited in
  a paper's own bibliography instead of its own (found on `hnsw.pdf`, which
  has no watermark on page 1 at all, being a journal-typeset repost).
- **The frontend uses no custom CSS at all** — every visual element is a
  first-party Streamlit component (`st.title`, `st.caption`, `st.badge`,
  `st.container(border=True)`, `st.columns`, etc.), not `st.markdown(...,
  unsafe_allow_html=True)` with hand-written styles. An earlier version did
  use custom CSS (targeting `data-testid` attributes for card shadows and
  button color) — removed on request, since a future Streamlit upgrade could
  rename those internal attributes and silently break styling that depended
  on them, and first-party components don't have that risk. One real
  trade-off: the relevance badge colors are limited to `st.badge()`'s fixed
  palette (`red`/`orange`/`yellow`/`blue`/`green`/`violet`/`gray`/`primary`)
  rather than the exact hex values used before, and there's no way to set a
  custom max content width — the layout is whatever `layout="wide"` gives.
- **There is intentionally no `.streamlit/config.toml`.** Testing showed
  that setting even one `[theme]` key there (tried: just `primaryColor` +
  `font`) disables Streamlit's own light/dark auto-detection and locks the
  whole app to a single fixed appearance — confirmed by forcing the OS color
  scheme to dark and watching the app stay light regardless. The search
  button's red color comes from Streamlit's own default `primaryColor`
  (`#FF4B4B`), not anything this project sets, so the app correctly follows
  whichever mode Streamlit itself is in with zero configuration.
