#!/usr/bin/env python
"""
Backfill extended metadata (tickers, report_type, sector, asset_class,
coverage_period) for documents already in the database.

No re-ingestion needed — reads the already-stored document text and runs
one Gemini Flash call per document.

Usage (from the PDF_summarizer/ directory):
    python backfill_metadata.py
    python backfill_metadata.py --max-workers 8
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure PDF_summarizer/ is on the path when called from the project root
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import PDFSummarizerPipeline

parser = argparse.ArgumentParser(
    description="Backfill extended metadata for existing documents (no re-ingestion)."
)
parser.add_argument(
    "--db-url",
    default=os.getenv(
        "PDF_SUMMARIZER_DB_URL",
        "postgresql+psycopg://user:password@localhost/pdf_summarizer",
    ),
    help="PostgreSQL connection string (defaults to $PDF_SUMMARIZER_DB_URL)",
)
parser.add_argument(
    "--max-workers",
    type=int,
    default=4,
    help="Concurrent Gemini calls (default: 4). Use 8+ on a paid API key.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-extract metadata for ALL documents, even those already populated. "
         "Use this after improving the extraction prompt.",
)
args = parser.parse_args()

print("Connecting to database…")
pipeline = PDFSummarizerPipeline(database_url=args.db_url)
pipeline.backfill_document_metadata(max_workers=args.max_workers, force=args.force)
