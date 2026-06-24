"""
Example usage of the Docling + Gemini verbalization pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()
from pipeline import PDFSummarizerPipeline
from database import DatabaseManager

DB_URL = os.getenv("PDF_SUMMARIZER_DB_URL")


def example_single_pdf():
    """Process a single PDF from research_pdfs."""
    print("=" * 60)
    print("Example 1: Process a Single PDF")
    print("=" * 60)

    pipeline = PDFSummarizerPipeline(database_url=DB_URL)

    result = pipeline.process_single_pdf(
        "test_pdfs/short_example.pdf",
        skip_existing=True,
    )

    print(f"\nResult: {result}")
    return result


def example_directory():
    """Process all PDFs in research_pdfs."""
    print("\n" + "=" * 60)
    print("Example 2: Process test_PDFs Directory")
    print("=" * 60)

    pipeline = PDFSummarizerPipeline(database_url=DB_URL)

    results = pipeline.process_directory(
        "test_PDFs/",
        skip_existing=True,
    )

    print(f"\nSummary: {results}")
    return results


def example_query_database():
    """Query the database for processed documents and chunks."""
    print("\n" + "=" * 60)
    print("Example 3: Query Database")
    print("=" * 60)

    db = DatabaseManager(database_url=DB_URL)

    doc = db.get_document_by_filename("example_report.pdf")
    if doc:
        print(f"\nDocument: {doc.filename}")
        print(f"  ID: {doc.id}")
        print(f"  Pages: {doc.total_pages}")

        chunks = db.get_chunks_by_document(doc.id)
        print(f"  Stored chunks: {len(chunks)}")

        if chunks:
            c = chunks[0]
            print(f"\n  First chunk (page {c.page_number}) preview:")
            print(f"    Raw content: {(c.raw_content or '')[:200]}...")
            print(f"    Verbalized summary: {(c.verbalized_summary or '')[:200]}...")


if __name__ == "__main__":
    print("PDF Summarizer - Docling + Gemini verbalization")
    print("\nSet GEMINI_API_KEY before running.")
    print("Uncomment examples below to run.\n")

    example_single_pdf()
    # example_directory()
    # example_query_database()
    

    print("Done.")
