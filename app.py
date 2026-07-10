"""
RAG-Based AI Search System — Streamlit interface.

Run with:
    streamlit run app.py

Pipeline: LangChain document loaders -> RecursiveCharacterTextSplitter chunking
-> BAAI/bge-small-en-v1.5 embeddings -> FAISS vector search (wide recall) ->
ms-marco-MiniLM-L-6-v2 cross-encoder rerank -> relevance-threshold filter (on
the rerank score) -> extractive or cloud gemini-3.5-flash (Gemini API)
generation -> display.

No custom CSS, no unsafe_allow_html -- every visual element here is a
first-party Streamlit component (st.title, st.caption, st.badge,
st.container(border=True), st.columns, etc.), so the app automatically
follows whichever theme (light/dark, colors) Streamlit itself is
configured with, with no risk of custom styles drifting out of sync with a
future Streamlit version's internal markup. Functional pieces required by
the project brief -- query input + submit, answer panel, expandable sources
panel with doc name + similarity score, an adjustable setting -- are
unchanged.
"""

import time
from pathlib import Path

import streamlit as st

from rag.ingest import load_documents_any, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer_stream, AVAILABLE_MODELS

# Resolved relative to this file, not the process's working directory --
# `streamlit run app.py` from a different cwd (or being launched by tooling
# that doesn't set cwd to the project root) would otherwise raise
# FileNotFoundError on a plain relative path.
DATA_FOLDER = str(Path(__file__).resolve().parent / "data" / "papers")
CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

# Relevance gate, now on the cross-encoder RERANK score (0-1, sigmoid of the
# reranker logit) that VectorStore.query() returns -- NOT bi-encoder cosine.
# The reason for the switch: raw cosine cannot separate in-corpus from
# out-of-corpus queries at all -- both populations land in ~0.54-0.65, so
# every possible cosine threshold either refused real questions (e.g. "What is
# HNSW?" at 0.56, "What is the Attention Is All You Need paper about?" at 0.65)
# or admitted unrelated ones ("What is the capital of France?" at 0.65). The
# cross-encoder score separates them: genuinely out-of-corpus queries score
# well under 0.05, while real (even weak-signal acronym/title) queries score
# above ~0.28. 0.10 sits in that gap. This is a much wider, more robust margin
# than the ~0.03 cosine gap the old 0.66 threshold had to thread. See
# EVALUATION.md for the calibration data.
#
# The gate is deliberately lenient: in "llm" mode the grounded system prompt
# is the real final judge of whether the retrieved context answers the
# question (verified: given genuinely off-topic context it refuses on its
# own). The gate's main job is a cheap first-pass refusal for clearly
# unrelated queries and the sole relevance decision for "extractive" mode,
# which has no LLM to make that call.
MIN_RELEVANCE_SCORE = 0.10

# The LLM is fed up to this many reranked chunks, independent of how many
# sources the UI lists (the top_k slider). Multi-paper synthesis questions
# ("compare REALM and RAG", "how does ColBERT differ from DPR?") need every
# relevant paper in the context, but showing 8 source cards would clutter the
# UI -- so generation gets the wider set while display stays at the user's
# top_k. Only applies to "llm" mode; "extractive" mode shows exactly what it
# lists. Costs nothing extra at retrieval time (the rerank pool is already 40).
LLM_CONTEXT_K = 8

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading, chunking, and embedding documents...")
def load_store():
    docs = load_documents_any(DATA_FOLDER)
    chunks = build_chunk_records(docs, chunk_size=CHUNK_MAX_WORDS, chunk_overlap=CHUNK_OVERLAP_WORDS)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


def relevance_tier(score: float) -> tuple[str, str]:
    """Return (label, st.badge color name) for a rerank-relevance badge.

    Tiers are on the cross-encoder relevance scale (0-1) VectorStore.query()
    now returns, which is bimodal (confident matches saturate near 1.0, weak-
    but-real matches sit in a broad middle band), so the cutoffs differ from
    the old cosine-scale ones.
    """
    if score >= 0.90:
        return "Excellent match", "green"
    if score >= 0.50:
        return "Good match", "blue"
    return "Relevant", "orange"


store, docs, chunks = load_store()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode needs a GOOGLE_API_KEY "
                          "set (see .env.example) -- free at aistudio.google.com/apikey.")
    # Model picker only makes sense in "llm" mode; hidden otherwise so extractive
    # mode stays zero-config. Default is AVAILABLE_MODELS[0] (a full flash model
    # chosen for strict grounding); "-lite" options trade grounding for speed.
    llm_model = AVAILABLE_MODELS[0]
    if mode == "llm":
        llm_model = st.selectbox(
            "LLM model", AVAILABLE_MODELS, index=0,
            help="Full flash models follow the documents-only rule strictly. "
                 "'-lite' models are faster but more likely to answer from general "
                 "knowledge. If a model is unavailable/overloaded, the app retries "
                 "then falls back to extractive rather than failing.",
        )
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents → **{len(chunks)}** chunks")
    st.caption(f"Chunking: RecursiveCharacterTextSplitter, ~{CHUNK_MAX_WORDS} words/chunk, "
               f"{CHUNK_OVERLAP_WORDS}-word overlap")
    with st.expander("Documents in this index"):
        for d in sorted(docs, key=lambda d: d["title"]):
            if d.get("authors") and d.get("year"):
                st.markdown(f"- **{d['title']}** — {d['authors']}, {d['year']}")
            else:
                st.markdown(f"- **{d['title']}**")
    with st.expander("System architecture"):
        st.markdown(
            "- **Document loading:** LangChain (`PyMuPDFLoader`, `TextLoader`)\n"
            "- **Metadata:** dynamic arXiv ID lookup (own watermark + fetch manifest), no hardcoded table\n"
            "- **Chunking:** LangChain `RecursiveCharacterTextSplitter`\n"
            "- **Embeddings:** `BAAI/bge-small-en-v1.5` (sentence-transformers, local)\n"
            "- **Vector store:** FAISS (`IndexFlatIP`, exact cosine search) — wide recall\n"
            "- **Reranker:** `ms-marco-MiniLM-L-6-v2` cross-encoder — precision + relevance gate\n"
            f"- **Generation:** Gemini via the Gemini API (cloud), streamed; model selectable in "
            f"LLM mode (default `{AVAILABLE_MODELS[0]}`)\n"
            "- **Interface:** Streamlit"
        )

st.title("RAG-Based AI Search System")
st.caption("Ask a question about the indexed AI/ML research papers below.")

with st.form("query_form", border=False):
    col_input, col_button = st.columns([6, 1], vertical_alignment="bottom")
    with col_input:
        query = st.text_input("Your question", placeholder="e.g. How does the attention mechanism work in Transformers?",
                               label_visibility="collapsed")
    with col_button:
        search_clicked = st.form_submit_button("Search", type="primary", use_container_width=True)

if search_clicked and query.strip():
    with st.spinner("Retrieving relevant passages..."):
        t0 = time.perf_counter()
        # Retrieve enough for the LLM's wider context even when the user's
        # top_k (what's shown) is small; display is sliced back to top_k below.
        retrieved = store.query(query, top_k=max(top_k, LLM_CONTEXT_K))
        t1 = time.perf_counter()

    relevant = [(c, s) for c, s in retrieved if s >= MIN_RELEVANCE_SCORE]
    display_hits = relevant[:top_k]
    # "llm" mode sees the wider context for synthesis; "extractive" mode shows
    # exactly the passages it lists, so its answer and its sources stay in sync.
    gen_hits = relevant if mode == "llm" else display_hits

    if not relevant:
        st.info("No sufficiently relevant passages were found for that query. "
                 "Try rephrasing, or ask about a topic covered in the indexed papers.")
    else:
        st.subheader("Answer")
        spinner_text = "Generating answer with Gemini..." if mode == "llm" else "Preparing extractive answer..."
        t_gen_start = time.perf_counter()
        with st.spinner(spinner_text):
            answer = st.write_stream(generate_answer_stream(query, gen_hits, mode=mode, model=llm_model))
        t2 = time.perf_counter()

        st.caption(f"Retrieved in {(t1 - t0) * 1000:.0f} ms · Generated in {(t2 - t_gen_start) * 1000:.0f} ms")

        st.subheader("Sources")
        for chunk, score in display_hits:
            with st.container(border=True):
                col1, col2 = st.columns([5, 2])
                with col1:
                    st.markdown(f"**{chunk.doc_title}**")
                    if chunk.authors and chunk.year:
                        st.caption(f"{chunk.authors} · {chunk.year}")
                with col2:
                    label, color = relevance_tier(score)
                    with st.container(horizontal_alignment="right"):
                        st.badge(f"{label} · {score:.0%}", color=color)
                if chunk.arxiv_id:
                    st.caption(f"[arXiv:{chunk.arxiv_id}](https://arxiv.org/abs/{chunk.arxiv_id})")
                with st.expander("View excerpt"):
                    st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")
