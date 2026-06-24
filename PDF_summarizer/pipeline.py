"""
Main pipeline: Docling parse → Gemini verbalize → Store in Postgres.

Process PDFs from research_pdfs (or any directory) and save one verbalized row per page.
Chunk metadata includes parent_chunk_id and prev/next_sibling_chunk_id (UUIDs as strings)
for easy DB lookup in the RAG pipeline.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import uuid as uuid_lib

from database import DatabaseManager
from pdf_processor import DoclingProcessor
from rag_gemini import GeminiRAGPipeline
from utils import get_file_hash

class PDFSummarizerPipeline:
    """Parse PDFs with Docling, verbalize charts with Gemini, store in Postgres."""

    def __init__(self, database_url: str = "sqlite:///pdf_summarizer.db"):
        self._database_url = database_url
        self.db_manager = DatabaseManager(database_url)
        self.processor = DoclingProcessor()
        self.rag = GeminiRAGPipeline(db=self.db_manager)

    def process_single_pdf(
        self,
        pdf_path: str,
        skip_existing: bool = True,
        skip_embeddings: bool = False,
    ) -> dict:
        """
        Process a single PDF: Docling parse → Gemini verbalize → store pages.
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        file_hash = get_file_hash(pdf_path)
        if skip_existing:
            existing = self.db_manager.get_document_by_hash(file_hash)
            if existing:
                print(f" Already processed: {filename}")
                return {
                    "status": "skipped",
                    "filename": filename,
                    "document_id": existing.id,
                    "message": "File already in database",
                }

        try:
            print(f"📄 Processing: {filename}")

            chunks, total_pages, file_size_bytes, doc_metadata = self.processor.process_pdf(str(pdf_path))

            doc = self.db_manager.add_document(
                filename=filename,
                file_path=str(pdf_path.absolute()),
                total_pages=total_pages,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                sender_name=doc_metadata.get("sender_name"),
                sender_company=doc_metadata.get("sender_company"),
                sent_date=doc_metadata.get("sent_date"),
                tickers=doc_metadata.get("tickers"),
                report_type=doc_metadata.get("report_type"),
                sector=doc_metadata.get("sector"),
                asset_class=doc_metadata.get("asset_class"),
                coverage_period_from=doc_metadata.get("coverage_period_from"),
                coverage_period_to=doc_metadata.get("coverage_period_to"),
            )

            chunk_ids = self.db_manager.add_chunks(doc.id, chunks)
            self._set_parent_sibling_metadata(chunk_ids, chunks)

            if not skip_embeddings:
                print(f"   Generating embeddings...")
                self.rag.backfill_embeddings()

            print(f"✅ Done: {filename}")
            print(f"   Pages: {total_pages}")
            print(f"   Document ID: {doc.id}")

            return {
                "status": "success",
                "filename": filename,
                "document_id": doc.id,
                "total_pages": total_pages,
                "file_size_bytes": file_size_bytes,
            }

        except Exception as e:
            print(f"❌ Error: {filename}: {e}")
            return {"status": "error", "filename": filename, "error": str(e)}

    def process_directory(
        self,
        directory_path: str,
        skip_existing: bool = True,
        max_workers: int = 4,
    ) -> dict:
        """Process all PDFs in a directory, up to max_workers PDFs at a time.

        Each worker creates its own pipeline instance (DoclingProcessor is not
        guaranteed thread-safe).  Embeddings are backfilled in a single batch
        after all PDFs are ingested rather than once per file.

        Args:
            max_workers: Number of PDFs to process concurrently.  Defaults to 4.
                         Set to 1 to restore sequential behaviour.
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDFs in {directory_path}")
            return {"status": "no_files", "total_files": 0}

        print(f"📁 Found {len(pdf_files)} PDF(s) in {directory_path} (max_workers={max_workers})\n")

        results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "results": [],
        }

        database_url = self._database_url

        def _process_one(pdf_file: Path) -> dict:
            # Own pipeline instance per thread to avoid DoclingProcessor sharing issues.
            p = PDFSummarizerPipeline(database_url=database_url)
            return p.process_single_pdf(
                str(pdf_file),
                skip_existing=skip_existing,
                skip_embeddings=True,   # single backfill pass at the end
            )

        workers = min(max_workers, len(pdf_files))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_file = {ex.submit(_process_one, f): f for f in pdf_files}
            for fut in as_completed(future_to_file):
                pdf_file = future_to_file[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"status": "error", "filename": pdf_file.name, "error": str(e)}
                results["results"].append(r)
                if r["status"] == "success":
                    results["successful"] += 1
                elif r["status"] == "error":
                    results["failed"] += 1
                    print(f"❌ Error processing {pdf_file.name}: {r.get('error')}")
                elif r["status"] == "skipped":
                    results["skipped"] += 1

        # Single embedding backfill pass after all PDFs have been ingested.
        if results["successful"] > 0:
            print(f"\n📐 Generating embeddings for all new chunks…")
            self.rag.backfill_embeddings()

        print(f"\n📊 Summary: {results['successful']} ok, {results['failed']} failed, {results['skipped']} skipped")
        return results

    def backfill_document_metadata(self, max_workers: int = 4, force: bool = False) -> dict:
        """Re-extract extended metadata (tickers, report_type, sector, asset_class,
        coverage_period) for documents that are missing it, using the already-stored
        document-level chunk text.  No re-ingestion, no Docling, no embeddings —
        just one Gemini Flash call per document.

        Returns a summary dict with counts of updated / skipped / failed documents.
        """
        from pdf_processor import _extract_document_metadata
        from datetime import date as date_cls

        docs = self.db_manager.get_documents_missing_metadata(force=force)
        if not docs:
            print("✅ All documents already have extended metadata — nothing to do.")
            return {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

        print(f"📋 Found {len(docs)} document(s) missing extended metadata.\n")

        results = {"total": len(docs), "updated": 0, "skipped": 0, "failed": 0}

        def _parse_date(value):
            if value is None:
                return None
            if isinstance(value, date_cls):
                return value
            try:
                return date_cls.fromisoformat(str(value))
            except (ValueError, TypeError):
                return None

        def _process_one(doc) -> str:
            """Returns 'updated', 'skipped', or 'failed'."""
            text = self.db_manager.get_document_chunk_text(doc.id)
            if not text.strip():
                print(f"  ⚠️  {doc.filename}: no stored text found — skipping")
                return "skipped"
            try:
                meta = _extract_document_metadata(text)
                self.db_manager.update_document_metadata(
                    document_id=doc.id,
                    tickers=meta.get("tickers"),
                    report_type=meta.get("report_type"),
                    sector=meta.get("sector"),
                    asset_class=meta.get("asset_class"),
                    coverage_period_from=_parse_date(meta.get("coverage_period_from")),
                    coverage_period_to=_parse_date(meta.get("coverage_period_to")),
                    # Also fill sender fields if they were previously missing
                    sender_name=meta.get("sender_name") if not doc.sender_name else None,
                    sender_company=meta.get("sender_company") if not doc.sender_company else None,
                    sent_date=_parse_date(meta.get("sent_date")) if not doc.sent_date else None,
                )
                tickers_str = ", ".join(meta.get("tickers") or []) or "—"
                print(
                    f"  ✅ {doc.filename}\n"
                    f"     tickers={tickers_str}  type={meta.get('report_type')}  "
                    f"class={meta.get('asset_class')}  sector={meta.get('sector')}"
                )
                return "updated"
            except Exception as e:
                print(f"  ❌ {doc.filename}: {e}")
                return "failed"

        workers = min(max_workers, len(docs))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for outcome in ex.map(_process_one, docs):
                results[outcome] += 1

        print(
            f"\n📊 Metadata backfill complete: "
            f"{results['updated']} updated, "
            f"{results['skipped']} skipped, "
            f"{results['failed']} failed"
        )
        return results

    def _set_parent_sibling_metadata(
        self, chunk_ids: List[uuid_lib.UUID], chunks: List[dict]
    ) -> None:
        """
        Set parent_chunk_id, prev_sibling_chunk_id, next_sibling_chunk_id in each chunk's
        metadata for easy DB lookup in RAG. chunk_ids and chunks are in the same order.
        Uses only chunk dicts (no detached ORM objects) to avoid Session errors.
        """
        doc_idx: int | None = None
        section_id_to_idx: dict = {}

        for i, c_dict in enumerate(chunks):
            meta = c_dict.get("metadata") or {}
            level = meta.get("level")
            if level == "document":
                doc_idx = i
            elif level == "section":
                sid = meta.get("section_id")
                if sid is not None:
                    section_id_to_idx[sid] = i

        for i, c_dict in enumerate(chunks):
            meta = dict(c_dict.get("metadata") or {})
            level = meta.get("level")
            parent_id: str | None = None
            prev_sib_id: str | None = None
            next_sib_id: str | None = None

            if level == "section" and doc_idx is not None:
                parent_id = str(chunk_ids[doc_idx])
                if i > doc_idx + 1 and (chunks[i - 1].get("metadata") or {}).get("level") == "section":
                    prev_sib_id = str(chunk_ids[i - 1])
                if i < len(chunks) - 1 and (chunks[i + 1].get("metadata") or {}).get("level") == "section":
                    next_sib_id = str(chunk_ids[i + 1])
            elif level == "image":
                sid = meta.get("section_id")
                if sid and sid in section_id_to_idx:
                    parent_id = str(chunk_ids[section_id_to_idx[sid]])

            meta["parent_chunk_id"] = parent_id
            meta["prev_sibling_chunk_id"] = prev_sib_id
            meta["next_sibling_chunk_id"] = next_sib_id
            self.db_manager.update_chunk_metadata(chunk_ids[i], meta)

    def get_document_info(self, document_id: int) -> Optional[dict]:
        """Get info about a processed document."""
        session = self.db_manager.get_session()
        try:
            from database import PDFDocument

            doc = session.query(PDFDocument).filter_by(id=document_id).first()
            if not doc:
                return None

            chunks = self.db_manager.get_chunks_by_document(document_id)
            return {
                "id": doc.id,
                "filename": doc.filename,
                "file_path": doc.file_path,
                "total_pages": doc.total_pages,
                "total_chunks_stored": len(chunks),
                "uploaded_at": doc.uploaded_at.isoformat(),
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "chunks": [
                    {
                        "id": str(c.id),
                        "page_number": c.page_number,
                        "raw_preview": (c.raw_content or "")[:200] + "...",
                    }
                    for c in chunks[:10]
                ],
            }
        finally:
            session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF pipeline: Docling + Gemini verbalization")
    parser.add_argument(
        "input", nargs="?",
        help="PDF file or directory to ingest (omit when using --backfill-metadata)",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("PDF_SUMMARIZER_DB_URL", "postgresql+psycopg://user:password@localhost/pdf_summarizer"),
        help="Database URL (Postgres + pgvector required)",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess existing files")
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Max concurrent PDFs when processing a directory (default: 4)",
    )
    parser.add_argument(
        "--backfill-metadata", action="store_true",
        help="Re-extract tickers / report_type / sector / asset_class / coverage_period "
             "for all existing documents that are missing these fields. "
             "No re-ingestion — uses already-stored chunk text.",
    )

    args = parser.parse_args()
    pipeline = PDFSummarizerPipeline(database_url=args.db_url)

    if args.backfill_metadata:
        pipeline.backfill_document_metadata(max_workers=args.max_workers)
        return

    if not args.input:
        parser.error("positional argument 'input' is required unless --backfill-metadata is set")

    path = Path(args.input)
    if path.is_file() and path.suffix.lower() == ".pdf":
        pipeline.process_single_pdf(str(path), skip_existing=not args.no_skip_existing)
    elif path.is_dir():
        pipeline.process_directory(
            str(path),
            skip_existing=not args.no_skip_existing,
            max_workers=args.max_workers,
        )
    else:
        print(f"❌ Invalid input: {args.input} (must be PDF file or directory)")
        sys.exit(1)


if __name__ == "__main__":
    main()
