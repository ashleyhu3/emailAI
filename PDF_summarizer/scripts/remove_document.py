"""
Remove a document (and all its chunks) from the database.

Usage:
  # By file path (uses file hash to find the document)
  python remove_document.py /path/to/file.pdf

  # By document ID
  python remove_document.py --doc-id 5

  # By filename (as stored in DB)
  python remove_document.py --filename "report.pdf"
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from database import DatabaseManager
from utils import get_file_hash

DB_URL = os.getenv(
    "PDF_SUMMARIZER_DB_URL",
    "postgresql+psycopg://user:password@localhost/pdf_summarizer",
)


def main() -> None:
    db = DatabaseManager(database_url=DB_URL)
    doc = None

    if "--doc-id" in sys.argv:
        try:
            i = sys.argv.index("--doc-id")
            doc_id = int(sys.argv[i + 1])
        except (IndexError, ValueError):
            print("Usage: python remove_document.py --doc-id <integer>")
            sys.exit(1)
        session = db.get_session()
        try:
            from database import PDFDocument
            doc = session.query(PDFDocument).filter_by(id=doc_id).first()
        finally:
            session.close()
    elif "--filename" in sys.argv:
        try:
            i = sys.argv.index("--filename")
            filename = sys.argv[i + 1]
        except IndexError:
            print("Usage: python remove_document.py --filename \"report.pdf\"")
            sys.exit(1)
        doc = db.get_document_by_filename(filename)
    else:
        if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
            print(__doc__)
            sys.exit(1)
        pdf_path = Path(sys.argv[1]).expanduser().resolve()
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            sys.exit(1)
        file_hash = get_file_hash(pdf_path)
        doc = db.get_document_by_hash(file_hash)

    if not doc:
        print("Document not found in database.")
        sys.exit(1)

    doc_id = doc.id
    filename = doc.filename
    deleted = db.delete_document(doc_id)
    if deleted:
        print(f"Removed document id={doc_id} ({filename}) and all its chunks.")
    else:
        print("Delete failed (document may already be gone).")
        sys.exit(1)


if __name__ == "__main__":
    main()
