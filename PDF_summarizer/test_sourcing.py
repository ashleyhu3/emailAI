"""
Tests that source citations (filename + page) flow correctly through the RAG pipeline.
No Gemini API calls are made — all LLM interactions are mocked.

Run from PDF_summarizer/:
    python test_sourcing.py
"""
import sys
import uuid
from unittest.mock import MagicMock, patch

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(filename: str, page: int, content: str, summary: str,
               doc_id: int = 1, level: str = "page") -> MagicMock:
    chunk = MagicMock()
    chunk.id = uuid.uuid4()
    chunk.document_id = doc_id
    chunk.page_number = page
    chunk.raw_content = content
    chunk.verbalized_summary = summary
    chunk.metadata_ = {"level": level}
    chunk.document = MagicMock()
    chunk.document.filename = filename
    return chunk


def build_pipeline(db: MagicMock) -> object:
    """Construct a GeminiRAGPipeline with a mocked client and the given db."""
    with patch("rag_gemini._get_client", return_value=MagicMock()):
        from rag_gemini import GeminiRAGPipeline
        pipeline = GeminiRAGPipeline.__new__(GeminiRAGPipeline)
        pipeline.client = MagicMock()
        pipeline.db = db
    return pipeline


def passed(name: str) -> None:
    print(f"  PASS  {name}")


def failed(name: str, detail: str) -> None:
    print(f"  FAIL  {name}: {detail}")
    sys.exit(1)


# ── Test 1: _build_context includes filename and page ─────────────────────────

def test_context_contains_source():
    db = MagicMock()
    db.get_chunk_by_id.return_value = None
    pipeline = build_pipeline(db)

    from rag_gemini import GeminiRAGPipeline
    chunk = make_chunk("annual_report_2024.pdf", 7, "Revenue was $100M", "Revenue summary")
    context, _ = pipeline._build_context([chunk])

    if "annual_report_2024.pdf" not in context:
        failed("context contains filename", f"filename missing from context:\n{context}")
    if "page 7" not in context:
        failed("context contains page", f"page number missing from context:\n{context}")
    passed("context contains filename and page")


# ── Test 2: chunks_used metadata is correct ───────────────────────────────────

def test_chunks_used_metadata():
    db = MagicMock()
    db.get_chunk_by_id.return_value = None

    chunk = make_chunk("sinopac_q3.pdf", 12, "Net income $50M", "Profit summary")
    db.semantic_search_chunks.return_value = [chunk]

    pipeline = build_pipeline(db)
    pipeline.client.models.generate_content.return_value = MagicMock(
        text="Net income was $50M (sinopac_q3.pdf page 12)"
    )

    with patch("rag_gemini.embed_text", return_value=[0.1] * 768):
        result = pipeline.answer_question("What was net income?")

    used = result["chunks_used"]
    if len(used) != 1:
        failed("chunks_used length", f"expected 1, got {len(used)}")
    if used[0]["page_number"] != 12:
        failed("chunks_used page_number", f"expected 12, got {used[0]['page_number']}")
    if used[0]["document_id"] != 1:
        failed("chunks_used document_id", f"expected 1, got {used[0]['document_id']}")
    passed("chunks_used has correct page_number and document_id")


# ── Test 3: no relevant chunks returns the fallback message ───────────────────

def test_no_relevant_chunks_fallback():
    db = MagicMock()
    db.semantic_search_chunks.return_value = []

    pipeline = build_pipeline(db)

    with patch("rag_gemini.embed_text", return_value=[0.1] * 768):
        result = pipeline.answer_question("What is the meaning of life?")

    if result["chunks_used"]:
        failed("empty chunks_used on no match", f"expected [], got {result['chunks_used']}")
    if "No sufficiently relevant" not in result["answer"]:
        failed("fallback answer text", f"unexpected answer: {result['answer']}")
    passed("returns fallback message when no relevant chunks found")


# ── Test 4: multiple chunks — all sources appear in context ───────────────────

def test_multiple_sources_in_context():
    db = MagicMock()
    db.get_chunk_by_id.return_value = None
    pipeline = build_pipeline(db)

    chunks = [
        make_chunk("report_a.pdf", 3, "Revenue A", "Summary A", doc_id=1),
        make_chunk("report_b.pdf", 9, "Revenue B", "Summary B", doc_id=2),
    ]
    context, _ = pipeline._build_context(chunks)

    for filename, page in [("report_a.pdf", 3), ("report_b.pdf", 9)]:
        if filename not in context:
            failed("multiple sources in context", f"{filename} missing")
        if f"page {page}" not in context:
            failed("multiple sources in context", f"page {page} missing")
    passed("context includes all filenames and pages for multiple chunks")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running sourcing tests (no Gemini API calls)...\n")
    test_context_contains_source()
    test_chunks_used_metadata()
    test_no_relevant_chunks_fallback()
    test_multiple_sources_in_context()
    print("\nAll tests passed.")
