"""
Docling inspection script.

Shows exactly what Docling produces for a PDF and how those values
map to what the ingestion pipeline uses:
  - item.label       → used to detect section headings
  - item.text        → used as section title
  - item.prov        → used to get page number
  - PictureItem      → triggers Gemini verbalization per image
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import PictureItem

# ── Config ────────────────────────────────────────────────────────────────────

# Change this to whichever PDF you want to inspect.
PDF_PATH = "/Users/davidfu/Desktop/Rays_Intern/test_pdfs/long_example.pdf"

# ── Docling setup (mirrors pdf_processor.py exactly) ─────────────────────────

pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = False
pipeline_options.generate_picture_images = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print(f"Parsing: {PDF_PATH}")
conv_res = converter.convert(str(PDF_PATH))
doc = conv_res.document
total_pages = len(doc.pages)
print(f"Total pages: {total_pages}\n")

# ── 1. All items — show label, text, prov ────────────────────────────────────
# This is what _build_section_map walks.

print("=" * 70)
print("ALL ITEMS (label | text[:60] | page)")
print("=" * 70)

heading_count = 0
for item, level in doc.iterate_items():
    label   = getattr(item, "label",    None)
    text    = getattr(item, "text",     None) or getattr(item, "title", None) or ""
    prov    = getattr(item, "prov",     None)
    page_no = prov[0].page_no if prov else "?"

    label_str = str(label).lower() if label else "(none)"
    is_heading = "heading" in label_str or "section" in label_str

    marker = "  <<< HEADING" if is_heading else ""
    if is_heading:
        heading_count += 1

    print(f"  [{label_str:30s}] lv={level} pg={page_no}  {text[:70]!r}{marker}")

print(f"\nTotal items that look like headings: {heading_count}")

# ── 2. Section map — what the pipeline actually builds ───────────────────────
# Replicate _build_section_map logic so you can see what sections are created.

print("\n" + "=" * 70)
print("SECTION MAP (what _build_section_map produces)")
print("=" * 70)

headings = []
for item, level in doc.iterate_items():
    text = getattr(item, "text", None) or getattr(item, "title", None)
    if not text:
        continue
    label     = getattr(item, "label", None) or getattr(item, "category", None) or getattr(item, "kind", None)
    type_name = str(label).lower() if label else ""
    if "heading" not in type_name and "section" not in type_name:
        continue
    prov = getattr(item, "prov", None)
    if not prov:
        continue
    try:
        page_no = int(prov[0].page_no)
    except Exception:
        continue
    headings.append((page_no, level, text.strip()))

if not headings:
    print("  No headings detected — all images/pages will show section='unknown'.")
    print("  This is normal for PDFs that don't have machine-readable headings.")
else:
    for i, (start_page, level, title) in enumerate(headings):
        end_page = headings[i + 1][0] - 1 if i + 1 < len(headings) else total_pages
        end_page = max(end_page, start_page)
        print(f"  sec_{i+1}  pages {start_page}-{end_page}  lv={level}  {title!r}")

# ── 3. Images — what triggers Gemini verbalization ───────────────────────────
# Each PictureItem with a pil_image becomes a Gemini call in pdf_processor.py.

print("\n" + "=" * 70)
print("IMAGES (each triggers one Gemini verbalization call)")
print("=" * 70)

img_count = 0
for item, _ in doc.iterate_items():
    if not isinstance(item, PictureItem):
        continue
    if item.image is None or item.image.pil_image is None:
        continue
    page_no = item.prov[0].page_no if item.prov else "?"
    w, h = item.image.pil_image.size
    img_count += 1
    print(f"  image {img_count:3d}  page={page_no}  size={w}x{h}")

print(f"\nTotal images that will be verbalized: {img_count}")

# ── 4. Gemini call estimate ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("GEMINI CALL ESTIMATE FOR THIS PDF")
print("=" * 70)
n_sections = len(headings)
n_sender   = 2  # up to 2 (first chunk + tail fallback)
n_doc_sum  = 1
n_sec_sum  = n_sections
n_images   = img_count
n_embed    = 1 + n_sections + img_count  # one per chunk stored
total_calls = n_sender + n_doc_sum + n_sec_sum + n_images + n_embed
print(f"  Sender extraction : up to {n_sender}")
print(f"  Document summary  : {n_doc_sum}")
print(f"  Section summaries : {n_sec_sum}  ({n_sections} sections)")
print(f"  Image verbalization: {n_images}")
print(f"  Embeddings        : ~{n_embed}  (doc + sections + images)")
print(f"  ─────────────────────────────")
print(f"  TOTAL             : ~{total_calls} Gemini calls")
