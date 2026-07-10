"""
Dynamic document metadata: derive a real title/authors/year for each PDF
instead of relying on a hand-maintained table.

The prior approach (rag/paper_metadata.py, now removed) hardcoded 23 entries
keyed by filename slug -- correct for the exact corpus it was written
against, but silently useless the moment you swap in different papers: any
new PDF just fell back to a filename-derived title ("hnsw" -> "Hnsw"),
because nothing in the code could look its real metadata up.

Three tiers, most to least reliable:

1. **data/papers/_manifest.json** (filename slug -> arxiv_id), written by
   scripts/fetch_papers.py at fetch time. Ground truth for exactly which
   paper each fetched PDF is, with zero guessing -- checked first.
2. **Extracted-text regex**, for any PDF not in the manifest (added some
   other way). arXiv stamps most of the PDFs it hosts with an
   "arXiv:XXXX.XXXXX" watermark in the page margin; this pulls that ID out
   of the *first page's* text only. Restricting to page 1 matters: searching
   the whole document once picked up an unrelated arXiv ID cited in a
   paper's own bibliography instead of its own ID (confirmed on hnsw.pdf,
   whose text has zero watermark matches on page 1 -- it's a journal-typeset
   repost without one -- but one citation-list match deep in the references).
3. **Filename-derived title**, if neither tier above finds an ID (or the ID
   found doesn't resolve via the API) -- same fallback this project has
   always had for a document with no known metadata.

Either tier 1 or 2 feeds the same arXiv Atom API lookup
(export.arxiv.org/api/query, no API key needed) for the real
title/authors/publication year. Results are cached to disk
(data/arxiv_metadata_cache.json) so repeat app launches don't re-hit the
network for papers already looked up, and a previously-indexed corpus still
loads if the network is briefly unavailable. Swap in any other real arXiv
paper (via fetch_papers.py or dropped in by hand) and tiers 1-2 pick up
correct metadata automatically; nothing needs to be hand-entered.
"""

import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

# arXiv's own watermark, e.g. "arXiv:1706.03762v7  [cs.CL]  2 Aug 2023" --
# present on the first page of most (not all -- see module docstring) real
# arXiv PDFs.
_ARXIV_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "papers" / "_manifest.json"

_manifest: Optional[dict] = None


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        try:
            _manifest = json.loads(_MANIFEST_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _manifest = {}
    return _manifest


def arxiv_id_from_manifest(filename_stem: str) -> Optional[str]:
    """Ground-truth arxiv_id for a PDF fetched by scripts/fetch_papers.py."""
    return _load_manifest().get(filename_stem)

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "arxiv_metadata_cache.json"
_API_TIMEOUT_SECONDS = 10
# arXiv's API usage notes ask callers not to hammer the endpoint -- this only
# costs anything on a cache miss (a genuinely new paper), not on every launch.
_API_COURTESY_DELAY_SECONDS = 1

_cache: Optional[dict] = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_cache, indent=2, sort_keys=True))


def extract_arxiv_id(text: str) -> Optional[str]:
    """Find arXiv's own watermark ID in extracted PDF text, if present."""
    match = _ARXIV_ID_RE.search(text)
    return match.group(1) if match else None


def _format_authors(names: List[str]) -> Optional[str]:
    names = [n for n in names if n]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    last_name = names[0].split()[-1] if names[0].split() else names[0]
    return f"{last_name} et al."


def _fetch_from_arxiv_api(arxiv_id: str) -> Optional[dict]:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=_API_TIMEOUT_SECONDS) as resp:
            xml_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    entry = root.find("atom:entry", _ATOM_NS)
    if entry is None:
        return None

    title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
    title = " ".join(title.split())  # arXiv titles often contain literal newlines
    if not title:
        return None

    authors = [
        (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
        for a in entry.findall("atom:author", _ATOM_NS)
    ]
    published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
    year = int(published[:4]) if published[:4].isdigit() else None

    return {
        "title": title,
        "authors": _format_authors(authors),
        "year": year,
        "arxiv_id": arxiv_id,
    }


def lookup_arxiv_metadata(arxiv_id: str) -> Optional[dict]:
    """Real title/authors/year for `arxiv_id`, from cache or arXiv's API.

    A miss is cached too (as None) so a paper the API can't resolve isn't
    re-queried on every single app launch -- only when the cache file itself
    is deleted or the entry is removed.
    """
    cache = _load_cache()
    if arxiv_id in cache:
        return cache[arxiv_id]

    metadata = _fetch_from_arxiv_api(arxiv_id)
    time.sleep(_API_COURTESY_DELAY_SECONDS)
    cache[arxiv_id] = metadata
    _save_cache()
    return metadata
