"""
Simple test: ingest a single PDF through the full pipeline.

Default behavior: clears the database completely, then runs ingest. All output
is written to test_ingest.txt in this script's directory.

Pipeline:
Docling parse → Gemini verbalization → store chunks in database.

PDF path: set PDF_SUMMARIZER_DEMO_PDF, or the script uses the first PDF
found in test_PDFs/ or research_pdfs/ (relative to this file).
"""

import os
import sys
import uuid as uuid_lib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


try:
    from database import DatabaseManager
    from pipeline import PDFSummarizerPipeline
except ImportError as e:
    print("Missing dependency:", e)
    print("Install with: pip install pgvector psycopg sqlalchemy python-dotenv")
    sys.exit(1)

SKIP_EXISTING = False

DB_URL = os.getenv(
    "PDF_SUMMARIZER_DB_URL",
    "postgresql+psycopg://user:password@localhost/pdf_summarizer",
)


def _find_pdf_path() -> Path:
    """Resolve PDF: env var, or first PDF in test_PDFs/ or research_pdfs/."""
    script_dir = Path(__file__).resolve().parent
    env_path = os.getenv("PDF_SUMMARIZER_DEMO_PDF")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"PDF_SUMMARIZER_DEMO_PDF set but file not found: {p}")
    for base in (script_dir, script_dir.parent):
        for folder in ("test_PDFs", "test_pdfs", "research_pdfs"):
            d = base / folder
            if d.is_dir():
                pdfs = sorted(d.glob("*.pdf"))
                if pdfs:
                    return pdfs[0].resolve()
    raise FileNotFoundError(
        "No PDF found. Set PDF_SUMMARIZER_DEMO_PDF=/path/to/file.pdf "
        "or add a PDF to test_PDFs/ or research_pdfs/ (relative to this script)."
    )

# How many characters of content/summary to show per chunk
SUMMARY_PREVIEW = 5000
CONTENT_PREVIEW = 4000


def _get_chunk_by_id(db: DatabaseManager, chunk_id: str):
    """Resolve chunk by UUID string; return None if invalid or missing."""
    if not chunk_id:
        return None
    try:
        return db.get_chunk_by_id(uuid_lib.UUID(chunk_id))
    except (ValueError, TypeError):
        return None


def _print_chunk(db: DatabaseManager, c, role: str, indent: str = "  ") -> None:
    """Print one chunk's metadata, summary preview, and raw content preview."""
    meta = c.metadata_ or {}
    summary = (c.verbalized_summary or "").strip()
    content = (c.raw_content or "").strip()
    print(f"{indent}{role} chunk_id={c.id}")
    print(f"{indent}  metadata: {meta}")
    print(f"{indent}  summary: {summary[:SUMMARY_PREVIEW]}{'...' if len(summary) > SUMMARY_PREVIEW else ''}")
    print(f"{indent}  content: {content[:CONTENT_PREVIEW]}{'...' if len(content) > CONTENT_PREVIEW else ''}")


def print_ingested_document_and_chunks(db: DatabaseManager, doc_id: int) -> None:
    """Print document summary, then each chunk with its metadata, summary, and parent/siblings."""
    chunks = db.get_chunks_by_document(doc_id)
    if not chunks:
        print("No chunks stored.")
        return

    # Document-level chunk (overall summary)
    doc_chunks = [c for c in chunks if (c.metadata_ or {}).get("level") == "document"]
    if doc_chunks:
        c = doc_chunks[0]
        print("\n" + "=" * 60)
        print("DOCUMENT-LEVEL CHUNK (overall summary)")
        print("=" * 60)
        _print_chunk(db, c, "Document", indent="  ")
        meta = c.metadata_ or {}
        parent_id = meta.get("parent_chunk_id")
        prev_id = meta.get("prev_sibling_chunk_id")
        next_id = meta.get("next_sibling_chunk_id")
        print("  parent:", parent_id or "(none)")
        print("  prev_sibling:", prev_id or "(none)")
        print("  next_sibling:", next_id or "(none)")

    # Section-level chunks
    section_chunks = [c for c in chunks if (c.metadata_ or {}).get("level") == "section"]
    for i, c in enumerate(section_chunks, 1):
        print("\n" + "=" * 60)
        print(f"SECTION CHUNK {i}: {(c.metadata_ or {}).get('section_title', '')}")
        print("=" * 60)
        _print_chunk(db, c, "Section", indent="  ")
        meta = c.metadata_ or {}
        parent = _get_chunk_by_id(db, meta.get("parent_chunk_id"))
        prev_sib = _get_chunk_by_id(db, meta.get("prev_sibling_chunk_id"))
        next_sib = _get_chunk_by_id(db, meta.get("next_sibling_chunk_id"))
        print("  --- Parent chunk ---")
        if parent:
            _print_chunk(db, parent, "Parent", indent="    ")
        else:
            print("    (none)")
        print("  --- Previous sibling chunk ---")
        if prev_sib:
            _print_chunk(db, prev_sib, "Prev sibling", indent="    ")
        else:
            print("    (none)")
        print("  --- Next sibling chunk ---")
        if next_sib:
            _print_chunk(db, next_sib, "Next sibling", indent="    ")
        else:
            print("    (none)")

    # Page-level chunks (each with parent + siblings)
    page_chunks = [c for c in chunks if (c.metadata_ or {}).get("level") == "page"]
    for i, c in enumerate(page_chunks, 1):
        print("\n" + "=" * 60)
        print(f"PAGE CHUNK {i} (page_number={c.page_number})")
        print("=" * 60)
        _print_chunk(db, c, "Page", indent="  ")
        meta = c.metadata_ or {}
        parent = _get_chunk_by_id(db, meta.get("parent_chunk_id"))
        prev_sib = _get_chunk_by_id(db, meta.get("prev_sibling_chunk_id"))
        next_sib = _get_chunk_by_id(db, meta.get("next_sibling_chunk_id"))
        print("  --- Parent chunk ---")
        if parent:
            _print_chunk(db, parent, "Parent", indent="    ")
        else:
            print("    (none)")
        print("  --- Previous sibling chunk ---")
        if prev_sib:
            _print_chunk(db, prev_sib, "Prev sibling", indent="    ")
        else:
            print("    (none)")
        print("  --- Next sibling chunk ---")
        if next_sib:
            _print_chunk(db, next_sib, "Next sibling", indent="    ")
        else:
            print("    (none)")

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(doc_chunks)} doc, {len(section_chunks)} section, {len(page_chunks)} page chunks")
    print("=" * 60)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "test_ingest.txt"
    out_file = open(out_path, "w")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        sys.stdout = out_file
        sys.stderr = out_file

        # Clear database first (test default)
        db = DatabaseManager(database_url=DB_URL)
        # n = db.delete_all_documents()
        # print(f"Cleared database: removed {n} document(s).\n")

        try:
            pdf_path = _find_pdf_path()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        if pdf_path.suffix.lower() != ".pdf":
            print(f"Not a PDF: {pdf_path}")
            sys.exit(1)

        print(f"Database: {DB_URL}")
        print(f"PDF file: {pdf_path}")
        print(f"Output file: {out_path}\n")

        pipeline = PDFSummarizerPipeline(database_url=DB_URL)
        result = pipeline.process_single_pdf(
            str(pdf_path),
            skip_existing=SKIP_EXISTING,
        )

        print("\n--- Pipeline Result ---")
        print(result)

        if result.get("status") in {"success", "skipped"}:
            doc_id = result["document_id"]
            print_ingested_document_and_chunks(db, doc_id)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        out_file.close()


if __name__ == "__main__":
    main()
