"""
Phase 2: Dual-Stream Local Preprocessing — 0 token cost.

Stream A (HTML emails): BS4 strip → markdownify → boilerplate regex → AOIM text.
Stream B (PDF attachments):
  - Metadata stream: Docling pages 1–2 only → AOIM text for Agent 2.
  - RAG stream: full Docling parse → chunked AOIM for vector DB ingestion.

Returns are plain strings / lists of dicts; no Gemini calls happen here.
"""

import io
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

try:
    import markdownify as md_lib
    _HAS_MARKDOWNIFY = True
except ImportError:
    _HAS_MARKDOWNIFY = False

# ── Legal boilerplate patterns ────────────────────────────────────────────────
# Anything matching these is sliced off (everything from the match to end-of-text).
_BOILERPLATE_PATTERNS = [
    re.compile(r"This (?:communication|message|report|email|document) is (?:confidential|proprietary).*", re.I | re.S),
    re.compile(r"IMPORTANT DISCLOSURES?.*", re.I | re.S),
    re.compile(r"Please see (?:our |important )?(?:disclosures?|disclaimers?).*", re.I | re.S),
    re.compile(r"This report has been prepared.*", re.I | re.S),
    re.compile(r"Analyst Certification.*", re.I | re.S),
    re.compile(r"For institutional investors only.*", re.I | re.S),
    re.compile(r"Risk[s]? and Disclaimer[s]?.*", re.I | re.S),
    re.compile(r"Reg(?:ulatory)? Disclosure[s]?.*", re.I | re.S),
    re.compile(r"The information contained herein does not constitute.*", re.I | re.S),
]

# Tracking pixels and short URLs that add no text value
_TRACKING_TAG_RE = re.compile(r"<img[^>]+(?:tracking|pixel|beacon|1x1)[^>]*>", re.I)

# Table tags we want markdownify to render properly
_UNWANTED_TAGS = ["script", "style", "head", "noscript", "iframe", "object", "embed"]


def clean_html_to_aoim(html: str) -> str:
    """
    Strip Chrome/JS/tracking from an HTML email body and return clean AOIM markdown.

    Approximate token savings vs. raw HTML: 60–80%.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags wholesale
    for tag in _UNWANTED_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    # Strip tracking pixels
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if any(kw in src for kw in ("track", "pixel", "beacon", "open.php", "click")):
            img.decompose()

    # Strip all <a> href attributes (keep link text, ditch URLs)
    for a in soup.find_all("a"):
        a.attrs = {}

    raw_html = str(soup)

    if _HAS_MARKDOWNIFY:
        text = md_lib.markdownify(
            raw_html,
            heading_style="ATX",
            bullets="-",
            strip=["a", "img"],
        )
    else:
        # Fallback: naive tag stripping
        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)

    text = _strip_boilerplate(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _strip_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[: m.start()].strip()
    return text


# ── PDF preprocessing ─────────────────────────────────────────────────────────

def _docling_converter():
    """Lazy-import Docling (heavy dependency) only when needed."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    pipeline_opts.do_table_structure = True
    pipeline_opts.generate_page_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        }
    )


def slice_pdf_pages_to_aoim(
    pdf_bytes: bytes,
    page_range: Tuple[int, int] = (1, 2),
) -> Tuple[str, List[dict]]:
    """
    Parse only `page_range` pages from `pdf_bytes` with Docling.

    Returns:
        aoim_text   — clean markdown of those pages (for Agent 2)
        figures     — list of {"page": N, "caption": str, "image": bytes or None}
                      extracted from PictureItem elements on those pages.

    Token cost: 0 (local only).
    """
    from docling.datamodel.document import PictureItem

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        converter = _docling_converter()
        result = converter.convert(tmp_path)
        doc = result.document

        pages_md: List[str] = []
        figures: List[dict] = []
        start, end = page_range

        for item, level in doc.iterate_items():
            page = getattr(item, "page_no", None) or getattr(
                getattr(item, "prov", [None])[0], "page_no", None
            )
            if page is None or not (start <= page <= end):
                continue

            if isinstance(item, PictureItem):
                img_bytes: Optional[bytes] = None
                try:
                    pil_img = item.get_image(doc)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                except Exception:
                    pass
                figures.append({
                    "page": page,
                    "caption": getattr(item, "caption", "") or "",
                    "image": img_bytes,
                })
            else:
                text = getattr(item, "text", None) or ""
                if text.strip():
                    pages_md.append(text)

        aoim_text = _strip_boilerplate("\n\n".join(pages_md))
        return aoim_text, figures
    finally:
        os.unlink(tmp_path)


# Matches report-type badges used in broker digest emails: | Idea |, | Insight |, etc.
_SECTION_BADGE_RE = re.compile(
    r"\|\s*(?:Idea|Insight|Thesis|Note|Alert|Update|Flash|Report|Comment|Strategy|Primer|Initiation)\s*\|",
    re.I,
)

def email_html_to_rag_chunks(
    html: str,
    broker: str = "",
    min_chunk_words: int = 8,
    max_chunk_words: int = 400,
) -> List[dict]:
    """
    Convert a broker digest email's HTML into RAG chunks.

    Used when an email has no PDF attachments but contains rich research
    summaries inline (e.g. Morgan Stanley "Today's Research", DB Digest).

    Strategy:
    1. Clean HTML → markdown.
    2. Strip CSS/VML noise and normalize whitespace.
    3. If badge markers like "| Idea |" are present, split on those (one chunk
       per research note). Otherwise split on paragraph double-newlines.
    4. Merge short blocks, split long ones at sentence boundaries.
    """
    cleaned = clean_html_to_aoim(html)
    if not cleaned.strip():
        return []

    # Strip VML/CSS lines and collapse whitespace
    lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if re.match(r'^[a-z\\*.:#\[\]@\s]+\{[^}]*\}\s*$', s, re.I):
            continue
        lines.append(re.sub(r"[ \t\xa0]{2,}", " ", line))
    # Join and normalize to a single clean string
    cleaned = " ".join(l.strip() for l in lines if l.strip())

    chunks: List[dict] = []
    chunk_idx = 0

    def _flush(text: str) -> None:
        nonlocal chunk_idx
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text or len(text.split()) < min_chunk_words:
            return
        if len(text.split()) > max_chunk_words:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            part: List[str] = []
            for sent in sentences:
                part.append(sent)
                if len(" ".join(part).split()) >= max_chunk_words // 2:
                    chunks.append(_make_chunk(" ".join(part), chunk_idx, broker))
                    chunk_idx += 1
                    part = []
            if part:
                chunks.append(_make_chunk(" ".join(part), chunk_idx, broker))
                chunk_idx += 1
        else:
            chunks.append(_make_chunk(text, chunk_idx, broker))
            chunk_idx += 1

    # Badge-style digest emails: split on | Idea |, | Insight |, etc.
    # Discard everything before the first badge (header/nav boilerplate).
    if _SECTION_BADGE_RE.search(cleaned):
        first = _SECTION_BADGE_RE.search(cleaned).start()
        content = cleaned[first:]
        sections = _SECTION_BADGE_RE.split(content)
        badges = _SECTION_BADGE_RE.findall(content)
        for i, badge in enumerate(badges):
            section_text = badge + " " + sections[i + 1]
            _flush(section_text)
        return chunks

    # Fallback: paragraph split + merge for non-badge emails (DB digest, etc.)
    raw_blocks = [b.strip() for b in re.split(r"\s{3,}", cleaned) if b.strip()]
    _NOISE_RE = re.compile(
        r"^(?:view\s+in\s+browser|unsubscribe|click\s+here|read\s+more|"
        r"not\s+for\s+(?:distribution|redistribution)|"
        r"[|•·—\-]{3,}|\d{1,2}[./]\d{1,2}[./]\d{2,4})$",
        re.I,
    )
    blocks = [b for b in raw_blocks if not _NOISE_RE.match(b) and len(b.split()) >= 3]

    buffer = ""
    for block in blocks:
        if buffer:
            combined = buffer + " " + block
            if len(combined.split()) > max_chunk_words:
                _flush(buffer)
                buffer = block
            else:
                buffer = combined
        else:
            buffer = block
    _flush(buffer)
    return chunks


def _make_chunk(text: str, idx: int, broker: str) -> dict:
    return {
        "raw_content": text,
        "verbalized_summary": None,
        "page_number": idx + 1,
        "metadata": {
            "page_number": idx + 1,
            "level": "email_section",
            "source": "email_body",
            "broker": broker,
        },
    }


def full_pdf_to_rag_chunks(pdf_bytes: bytes, filename: str = "report.pdf") -> List[dict]:
    """
    Full Docling parse of a PDF → list of chunk dicts ready for GeminiRAGPipeline.

    Each chunk: {"raw_content": str, "page_number": int, "metadata": {...}}

    This bypasses the generative LLM completely — embeddings are computed separately
    by GeminiRAGPipeline.backfill_embeddings().
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        converter = _docling_converter()
        result = converter.convert(tmp_path)
        doc = result.document

        chunks: List[dict] = []
        page_texts: dict = {}

        for item, level in doc.iterate_items():
            text = getattr(item, "text", None) or ""
            if not text.strip():
                continue
            page = getattr(item, "page_no", None) or getattr(
                getattr(item, "prov", [None])[0], "page_no", None
            )
            if page is not None:
                page_texts.setdefault(page, []).append(text)

        for page_num in sorted(page_texts):
            raw = "\n\n".join(page_texts[page_num])
            chunks.append({
                "raw_content": raw,
                "verbalized_summary": None,  # populated later by GeminiRAGPipeline
                "page_number": page_num,
                "metadata": {
                    "page_number": page_num,
                    "level": "page",
                    "filename": filename,
                },
            })

        return chunks
    finally:
        os.unlink(tmp_path)
