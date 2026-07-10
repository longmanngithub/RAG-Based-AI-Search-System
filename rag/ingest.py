"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Loaders: LangChain document loaders (`langchain_community.document_loaders`) --
`PyMuPDFLoader` for PDFs (backed by the same PyMuPDF/fitz engine as before,
just via LangChain's wrapper) and `TextLoader` for plain .txt files.

Chunking: LangChain's `RecursiveCharacterTextSplitter`, configured with a
word-count length_function so chunk_size/chunk_overlap stay in words (not raw
characters) -- matching the ~120-words-per-chunk target used throughout this
project's docs. It recursively tries paragraph -> line -> word -> character
splits, backing off only as far as needed to fit each chunk within the target
size, which avoids the single-oversized-chunk edge case our old regex
sentence splitter could hit on reference-list text with no sentence-ending
punctuation.

Metadata: real title, authors, year, and arXiv ID are looked up dynamically
per document (see rag/metadata.py) by extracting arXiv's own watermark ID
from the PDF's text and querying arXiv's public API -- no hardcoded table, so
swapping in different arXiv papers picks up correct metadata automatically. A
filename-derived title ("hnsw" -> "Hnsw", "colbert" -> "Colbert", getting
every acronym wrong) is only used as a last-resort fallback for a document
with no detectable arXiv ID.

De-hyphenation: PyMuPDF preserves a PDF's line-wrap hyphens literally (e.g.
"the former two" wrapped mid-word extracts as "the for-\nmer two"), which
otherwise surfaces as "for- mer" once chunk text gets joined with spaces for
display. Rejoined by regex right after extraction (`_dehyphenate`), only
when the character after the break is lowercase -- a hyphen followed by a
capital letter is more likely a real compound (e.g. a hyphenated proper
noun) than a line-wrap artifact, so those are left alone.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .metadata import arxiv_id_from_manifest, extract_arxiv_id, lookup_arxiv_metadata

_HYPHEN_LINEWRAP_RE = re.compile(r"(\w)-\n([a-z])")


def _dehyphenate(text: str) -> str:
    """Rejoin a word split by a line-wrap hyphen, e.g. "for-\\nmer" -> "former"."""
    return _HYPHEN_LINEWRAP_RE.sub(r"\1\2", text)


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    authors: Optional[str] = None
    year: Optional[int] = None
    arxiv_id: Optional[str] = None


def _title_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("_", " ").title()


def _metadata_card(doc: dict) -> Optional[str]:
    """A short synthetic 'what is this paper' chunk, one per document.

    Purpose: title/acronym queries like "What is ScaNN?" or "What is the paper
    Attention Is All You Need about?" retrieve poorly against a paper's *body*
    text when the acronym barely appears there (ScaNN's own paper is titled
    "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" --
    the string "ScaNN" is nearly absent). This card gives every paper one chunk
    that states its title and common name explicitly, so those queries have
    something to match. Verified: the card lands in the bi-encoder recall pool
    and wins the cross-encoder rerank for exactly these otherwise-missed queries.

    The "common name" is the corpus's own filename slug (e.g. "scann",
    "attention_is_all_you_need") -- NOT a hand-maintained acronym dictionary.
    It comes from the same source of truth as everything else (the fetched
    file / _manifest.json), so dropping in different papers needs no code change.
    A document with no resolvable title (no slug, no arXiv metadata) gets no
    card rather than a misleading one.
    """
    title = doc.get("title")
    if not title:
        return None
    slug = doc.get("slug", "") or ""
    common = slug.replace("_", " ").strip()
    parts = [f'This document is the research paper titled "{title}".']
    if common and common.lower() != title.lower():
        parts.append(f"It is commonly referred to as {common} ({slug}).")
    byline = ", ".join(str(x) for x in (doc.get("authors"), doc.get("year")) if x)
    if byline:
        parts.append(f"It was written by {byline}.")
    parts.append(f"This paper covers the concepts, methods, and findings of the "
                 f"{common or title} paper.")
    return " ".join(parts)


def _doc_metadata(filename: str, first_page_text: str) -> dict:
    """Look up a document's real metadata: the fetch manifest first (ground
    truth, if this PDF came from scripts/fetch_papers.py), then arXiv's own
    watermark ID extracted from page 1 only (deeper pages risk matching a
    citation's arXiv ID instead of the paper's own -- see rag/metadata.py).
    Falls back to a filename-derived title if neither yields a resolvable ID."""
    slug = os.path.splitext(filename)[0]
    arxiv_id = arxiv_id_from_manifest(slug) or extract_arxiv_id(first_page_text)
    if arxiv_id:
        looked_up = lookup_arxiv_metadata(arxiv_id)
        if looked_up:
            return looked_up
    return {"title": _title_from_filename(filename), "authors": None, "year": None, "arxiv_id": None}


def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder` into document dicts (title/text/metadata)."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        try:
            pages = TextLoader(path, encoding="utf-8").load()
            text = "\n".join(p.page_content for p in pages).strip()
        except Exception as e:
            print(f"[ingest] WARNING: skipping unreadable file {filename!r}: {e}")
            continue
        if not text:
            continue
        docs.append({"text": text, "slug": os.path.splitext(filename)[0],
                     **_doc_metadata(filename, pages[0].page_content)})
    return docs


def load_pdf_documents(folder: str) -> List[dict]:
    """Load every .pdf file in `folder` into document dicts (title/text/metadata).

    A PDF that fails to parse (corrupt file, scanned image with no text layer,
    etc.) is skipped with a warning rather than crashing the whole ingest run.
    """
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(folder, filename)
        try:
            pages = PyMuPDFLoader(path).load()
            text = _dehyphenate("\n".join(p.page_content for p in pages).strip())
        except Exception as e:
            print(f"[ingest] WARNING: skipping unreadable PDF {filename!r}: {e}")
            continue
        if not text:
            print(f"[ingest] WARNING: no extractable text in {filename!r}, skipping")
            continue
        docs.append({"text": text, "slug": os.path.splitext(filename)[0],
                     **_doc_metadata(filename, pages[0].page_content)})
    return docs


def load_documents_any(folder: str) -> List[dict]:
    """Load both .txt and .pdf documents from `folder`."""
    return load_documents(folder) + load_pdf_documents(folder)


def build_chunk_records(docs: List[dict], chunk_size: int = 120, chunk_overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding.

    chunk_size/chunk_overlap are in words, via RecursiveCharacterTextSplitter's
    length_function hook -- not the library's default of raw characters.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(text.split()),
    )
    records = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        # One synthetic metadata card per document, indexed first, so
        # "what is <paper>?" / bare-acronym queries have an explicit chunk to
        # match even when the acronym is nearly absent from the body text. The
        # paper's opening chunk (its title/abstract region) is appended so the
        # card carries enough real content for the LLM to actually *describe*
        # the paper, not just name it -- without this, "What is ScaNN?" matches
        # the card but the model can only echo the title back.
        card = _metadata_card(doc)
        if card:
            if pieces:
                card = f"{card} Overview from the paper: {pieces[0]}"
            records.append(Chunk(
                chunk_id=f"{doc['title']}::card",
                doc_title=doc["title"],
                text=card,
                authors=doc.get("authors"),
                year=doc.get("year"),
                arxiv_id=doc.get("arxiv_id"),
            ))
        for i, piece in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['title']}::{i}",
                doc_title=doc["title"],
                text=piece,
                authors=doc.get("authors"),
                year=doc.get("year"),
                arxiv_id=doc.get("arxiv_id"),
            ))
    return records
