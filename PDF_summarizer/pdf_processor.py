"""
PDF processing: Docling for layout parsing + Gemini for chart verbalization.

Workflow:
  1. Parse: Docling extracts page layout and page images
  2. Verbalize: Gemini describes every chart/graph/table on each page
  3. Store: One row per page with Text + Chart Summary
"""
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import PictureItem
from google import genai
from google.genai import types

VERBALIZE_MODEL = "models/gemini-2.5-flash"
TEXT_SUMMARY_MODEL = "models/gemini-2.5-flash"
IMAGE_RESOLUTION_SCALE = 2.0


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=key)


def _generate_with_retry(client: genai.Client, model: str, contents, max_attempts: int = 5):
    """Call generate_content with exponential backoff on 429 errors."""
    # Disable thinking for 2.5 Flash — these are straightforward extraction/summarization
    # tasks that don't benefit from chain-of-thought reasoning. Thinking adds several
    # seconds of latency to every call; disabling it gives ~3–5× speedup per call.
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == max_attempts - 1:
                    raise
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                print(f"   [WARNING] Gemini still rate limited despite throttling, retrying in {wait}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise


def _verbalize_page_image(pil_image, model_name: str = VERBALIZE_MODEL) -> str:
    """
    Send page image to Gemini; get plain-text description of charts/graphs/tables.
    (Used for embedding/search. Raw text comes from Docling.)
    """
    # return "GEMINI_DISABLED: This is a placeholder for testing database ingestion."
    client = _get_client()

    prompt = (
        "You are a financial analyst. Summarize this single page of a larger financial report. "
        "For every chart and table, extract the key data points, trends, and legends "
        "into a Markdown table format. Ensure that the visual insights "
        "(e.g., 'Revenue spiked in Q3') are explicitly written as text. "
        "Output the result in clean Markdown."
    )

    # Convert PIL to bytes for Gemini
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    response = _generate_with_retry(
        client, model_name,
        [prompt, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")],
    )
    # response.text can be None if the model response was blocked or empty
    return (response.text or "") if hasattr(response, "text") else str(response)


def _summarize_text_block(text: str, purpose: str) -> str:
    """
    Summarize a (potentially long) text block with Gemini.

    purpose: short description used in the prompt (e.g. 'whole document', 'section')
    """
    if not text.strip():
        return ""

    client = _get_client()

    prompt = (
        "You are a senior equity research analyst.\n"
        f"Summarize the following {purpose} from a financial report.\n"
        "- Capture company, report type, period, and key topics.\n"
        "- Use 3–8 bullet points.\n"
        "- Be factual and avoid speculation.\n"
    )

    # Truncate very long inputs to keep latency and token usage reasonable.
    if len(text) > 20000:
        text = text[:20000]

    response = _generate_with_retry(client, TEXT_SUMMARY_MODEL, [prompt, text])
    # response.text can be None if the model response was blocked or empty
    return (response.text or "") if hasattr(response, "text") else str(response)


_METADATA_DEFAULTS: Dict = {
    "sender_name": None,
    "sender_company": None,
    "sent_date": None,
    "tickers": None,
    "report_type": None,
    "sector": None,
    "asset_class": None,
    "coverage_period_from": None,
    "coverage_period_to": None,
}

_REPORT_TYPES = [
    "equity_research", "technical_analysis", "macro",
    "crypto", "sector_note", "strategy", "other",
]
_ASSET_CLASSES = ["equity", "crypto", "fixed_income", "commodity", "fx", "mixed"]


def _extract_document_metadata(text: str) -> Dict:
    """Extract all document-level metadata from financial document text in one Gemini call.

    Fields extracted:
      sender_name, sender_company, sent_date   — who sent it and when
      tickers          — list of primary ticker/asset symbols covered (e.g. ["BTC", "AAPL"])
      report_type      — one of: equity_research, technical_analysis, macro,
                         crypto, sector_note, strategy, other
      sector           — GICS sector (e.g. "Technology", "Energy") if equity-focused
      asset_class      — one of: equity, crypto, fixed_income, commodity, fx, mixed
      coverage_period_from / _to  — the period being ANALYSED (not when published)
                                    e.g. a Q3 earnings report published in November
                                    has coverage_period = Jul–Sep, sent_date = Nov

    Falls back gracefully: missing or unparseable fields are returned as null.
    Tries the document head first; if key identity fields are still missing,
    retries with the document tail (sender info sometimes lives in footers).
    """
    if not text.strip():
        return _METADATA_DEFAULTS.copy()

    client = _get_client()
    report_types_str = ", ".join(_REPORT_TYPES)
    asset_classes_str = ", ".join(_ASSET_CLASSES)

    def _run_extraction(excerpt: str) -> Dict:
        prompt = (
            "You are a financial document metadata extractor. "
            "Extract the following fields from the document excerpt below. "
            "Return ONLY valid JSON — no markdown, no explanation.\n\n"
            "Required JSON keys (use null for any field not found):\n"
            "{\n"
            '  "sender_name":          "<full name of author or sender, or null>",\n'
            '  "sender_company":       "<publishing firm, bank, or research house, or null>",\n'
            '  "sent_date":            "<YYYY-MM-DD when published/sent, or null>",\n'
            '  "tickers":              ["<exchange ticker symbols, e.g. 9958 TT, AAPL, BTC. Include exchange suffix if present.>"],\n'
            f' "report_type":          "<one of: {report_types_str} — or null>",\n'
            '  "sector":               "<GICS sector — see definitions below — or null if not equity>",\n'
            f' "asset_class":          "<one of: {asset_classes_str} — or null>",\n'
            '  "coverage_period_from": "<YYYY-MM-DD start of the period this report ANALYSES, or null>",\n'
            '  "coverage_period_to":   "<YYYY-MM-DD end   of the period this report ANALYSES, or null>"\n'
            "}\n\n"
            "GICS sector definitions — use EXACTLY these names:\n"
            "• Energy: oil & gas exploration/production, refining, energy equipment, pipelines,\n"
            "  wind/solar/renewable energy developers and operators\n"
            "• Materials: chemicals, metals & mining, steel, paper, packaging, construction materials\n"
            "• Industrials: aerospace & defense, machinery, construction & engineering, transportation,\n"
            "  commercial services, electrical equipment\n"
            "• Consumer Discretionary: retail, automotive, hotels, restaurants, media, gaming, apparel\n"
            "• Consumer Staples: food, beverages, tobacco, household products, personal care\n"
            "• Healthcare: pharmaceuticals, biotech, medical devices, health services\n"
            "• Financials: banks, insurance, asset management, capital markets, real estate finance\n"
            "• Information Technology: software, hardware, semiconductors, IT services, internet\n"
            "• Communication Services: telecom, media, interactive media & services\n"
            "• Utilities: electric utilities, water, gas utilities, independent power producers\n"
            "• Real Estate: REITs, real estate management & development\n\n"
            "Key distinctions:\n"
            "• Wind/solar/renewable energy COMPANIES → Energy (developers/operators) or Utilities\n"
            "  (regulated power producers). NOT Industrials or Materials.\n"
            "• tickers: preserve the full exchange symbol as written (e.g. '9958 TT', '2881 TW').\n"
            "• sent_date vs coverage_period: a Q3 2024 earnings report published in November 2024\n"
            "  has sent_date=2024-11-XX, coverage_period_from=2024-07-01, coverage_period_to=2024-09-30.\n"
            "• report_type=crypto if the PRIMARY focus is cryptocurrency.\n\n"
            f"Document excerpt:\n{excerpt}"
        )
        try:
            response = _generate_with_retry(client, TEXT_SUMMARY_MODEL, prompt)
            raw = (response.text or "").strip() if hasattr(response, "text") else ""
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return {
                "sender_name": data.get("sender_name") or None,
                "sender_company": data.get("sender_company") or None,
                "sent_date": data.get("sent_date") or None,
                "tickers": data.get("tickers") or None,
                "report_type": data.get("report_type") or None,
                "sector": data.get("sector") or None,
                "asset_class": data.get("asset_class") or None,
                "coverage_period_from": data.get("coverage_period_from") or None,
                "coverage_period_to": data.get("coverage_period_to") or None,
            }
        except Exception:
            return _METADATA_DEFAULTS.copy()

    # Use a larger head excerpt — titles, tickers, and coverage dates are usually upfront.
    result = _run_extraction(text[:8000])

    # If any key fields are still missing, retry with the document tail.
    # Sender info sometimes appears only in footers; tickers can appear after a long preamble.
    retry_fields = ("sender_name", "sender_company", "sent_date", "tickers")
    if not all(result.get(k) for k in retry_fields):
        tail = text[-3000:] if len(text) > 8000 else ""
        if tail:
            tail_result = _run_extraction(tail)
            for key in retry_fields:
                if not result.get(key) and tail_result.get(key):
                    result[key] = tail_result[key]

    return result


@dataclass
class SectionInfo:
    section_id: str
    title: str
    level: int
    start_page: int
    end_page: int


class DoclingProcessor:
    """
    Parse PDFs with Docling and verbalize charts with Gemini.
    Produces one verbalized page per row.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, max_workers: int = 4):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        # Max concurrent Gemini calls per PDF (images + section summaries).
        # Keep this low when processing multiple PDFs at once — the effective
        # total concurrency is (directory_workers × max_workers). At 2 PDFs × 4
        # workers = 8 concurrent calls, which is well within Gemini Flash limits.
        self.max_workers = max_workers

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_pdf(self, pdf_path: str) -> tuple:
        """
        Parse PDF with Docling and verbalize each page with Gemini.

        Returns:
            Tuple of (chunks, total_pages, file_size_bytes)
            Each chunk dict: raw_content (Docling markdown), verbalized_summary (Gemini), metadata

        Hierarchical strategy:
            - One document-level chunk with overall summary.
            - One chunk per major section (based on Docling headings).
            - One chunk per page, enriched with section context and sibling info.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_size_bytes = pdf_path.stat().st_size

        # Parse with Docling
        conv_res = self.converter.convert(str(pdf_path))
        doc = conv_res.document
        total_doc_pages = len(doc.pages)

        # Build a map of major sections from Docling's hierarchical structure.
        sections, page_to_section = self._build_section_map(doc)

        chunks: List[dict] = []

        # ----- Document-level chunk -----
        try:
            full_markdown = doc.export_to_markdown()
        except Exception:
            full_markdown = ""

        # Metadata extraction and document summary are independent — run concurrently.
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_meta = ex.submit(_extract_document_metadata, full_markdown)
            fut_doc_summary = ex.submit(_summarize_text_block, full_markdown, "whole document")
            extracted_meta = fut_meta.result()   # tickers, sector, report_type, etc.
            doc_summary = fut_doc_summary.result()

        chunk_doc_metadata = {
            "level": "document",
            "section_id": None,
            "section_title": None,
            "page_number": None,
            "page_span": [1, total_doc_pages] if total_doc_pages else None,
            "file_path": str(pdf_path.absolute()),
        }
        if full_markdown.strip() or doc_summary.strip():
            chunks.append(
                {
                    "raw_content": full_markdown or doc_summary or "[Document summary]",
                    "verbalized_summary": doc_summary or None,
                    "metadata": chunk_doc_metadata,
                }
            )

        # ----- Section-level chunks (parallelized) -----
        # Extract page texts first (fast, CPU-only), then fire all Gemini calls concurrently.
        section_data = [
            (sec, self._get_pages_text(doc, list(range(sec.start_page, sec.end_page + 1))))
            for sec in sections
        ]

        section_summaries: Dict[str, str] = {}
        if section_data:
            workers = min(self.max_workers, len(section_data))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_sec = {
                    ex.submit(_summarize_text_block, text, f"section '{sec.title}'"): sec
                    for sec, text in section_data
                }
                for fut in as_completed(future_to_sec):
                    sec = future_to_sec[fut]
                    try:
                        section_summaries[sec.section_id] = fut.result()
                    except Exception as e:
                        print(f"   [WARNING] Section summary failed for '{sec.title}': {e}")
                        section_summaries[sec.section_id] = ""

        # Assemble section chunks in original document order
        for sec, text in section_data:
            summary = section_summaries.get(sec.section_id, "")
            content_for_storage = text if text.strip() else summary
            if not content_for_storage.strip():
                continue
            metadata = {
                "level": "section",
                "section_id": sec.section_id,
                "section_title": sec.title,
                "section_level": sec.level,
                "page_span": [sec.start_page, sec.end_page],
                "file_path": str(pdf_path.absolute()),
            }
            chunks.append(
                {
                    "raw_content": content_for_storage,
                    "verbalized_summary": summary or None,
                    "metadata": metadata,
                }
            )

        # ----- Image-level chunks (parallelized) -----
        # Collect all images with provenance first, then verbalize all concurrently.
        image_items = []
        for item, _ in doc.iterate_items():
            if not isinstance(item, PictureItem):
                continue
            if item.image is None or item.image.pil_image is None:
                continue
            page_no = item.prov[0].page_no if item.prov else None
            sec = page_to_section.get(page_no) if page_no else None
            image_items.append((item.image.pil_image, page_no, sec))

        if image_items:
            verbalized_results: Dict[int, str] = {}
            workers = min(self.max_workers, len(image_items))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_idx: Dict = {}
                for i, (pil_image, _page_no, _sec) in enumerate(image_items):
                    future_to_idx[ex.submit(_verbalize_page_image, pil_image)] = i
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    try:
                        verbalized_results[i] = fut.result()
                    except Exception as e:
                        print(f"   [WARNING] Image verbalization failed (image {i + 1}): {e}")
                        verbalized_results[i] = ""

            # Assemble image chunks in original document order
            for i, (_pil, page_no, sec) in enumerate(image_items):
                verbalized = verbalized_results.get(i, "")
                if not verbalized.strip():
                    continue
                metadata = {
                    "level": "image",
                    "image_index": i + 1,
                    "page_number": page_no,
                    "section_id": sec.section_id if sec else None,
                    "section_title": sec.title if sec else None,
                    "file_path": str(pdf_path.absolute()),
                }
                chunks.append(
                    {
                        "raw_content": verbalized,
                        "verbalized_summary": verbalized,
                        "metadata": metadata,
                    }
                )

        return chunks, total_doc_pages, file_size_bytes, extracted_meta

    def _get_page_text(self, doc, page_no: int) -> str:
        """Extract text for a single page from Docling document, if available."""
        try:
            if hasattr(doc, "filter"):
                filtered = doc.filter(pages=[page_no])
                return filtered.export_to_markdown()
        except Exception:
            pass
        return ""

    def _get_pages_text(self, doc, pages: List[int]) -> str:
        """Extract text for a list of pages from Docling document."""
        if not pages:
            return ""
        try:
            if hasattr(doc, "filter"):
                filtered = doc.filter(pages=pages)
                return filtered.export_to_markdown()
        except Exception:
            pass
        return ""

    def _build_section_map(self, doc) -> Tuple[List[SectionInfo], Dict[int, SectionInfo]]:
        """
        Identify sections from Docling headings and map each page to its section.

        Strategy:
          - Walk all items; keep those whose type contains 'heading' or 'section'.
          - Use prov[0].page_no (Docling's provenance) to get the heading's page.
          - Each section starts at its heading page and ends just before the next heading.
        """
        sections: List[SectionInfo] = []

        # Collect (page_no, level, title) for every heading item.
        headings: List[Tuple[int, int, str]] = []

        for item, level in doc.iterate_items():
            title = getattr(item, "text", None) or getattr(item, "title", None)
            if not title:
                continue

            item_type = getattr(item, "label", None) or getattr(item, "category", None) or getattr(item, "kind", None)
            type_name = str(item_type).lower() if item_type is not None else ""

            if "heading" not in type_name and "section" not in type_name:
                continue

            # Use provenance to get the page this heading appears on.
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            try:
                page_no = int(prov[0].page_no)
            except (IndexError, AttributeError, TypeError, ValueError):
                continue

            headings.append((page_no, level, title.strip()))

        if not headings:
            return [], {}

        total_pages = len(doc.pages)

        # Each section runs from its heading page to the page before the next heading.
        for i, (start_page, level, title) in enumerate(headings):
            if i + 1 < len(headings):
                end_page = headings[i + 1][0] - 1
            else:
                end_page = total_pages
            end_page = max(end_page, start_page)

            section_id = f"sec_{i + 1}"
            sections.append(
                SectionInfo(
                    section_id=section_id,
                    title=title,
                    level=level,
                    start_page=start_page,
                    end_page=end_page,
                )
            )

        # Sort sections by (start_page, level) so that higher-level (smaller level) sections
        # for the same page range come first.
        sections.sort(key=lambda s: (s.start_page, s.level))

        # Map each page to the most specific section (highest level number) that contains it.
        page_to_section: Dict[int, SectionInfo] = {}
        for sec in sections:
            for p in range(sec.start_page, sec.end_page + 1):
                current = page_to_section.get(p)
                if current is None or sec.level > current.level:
                    page_to_section[p] = sec

        return sections, page_to_section
