"""Document management routes: upload, list, delete."""
import os
import tempfile
from typing import List, Optional
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from dependencies import get_db, get_pipeline, get_rag
from database import DatabaseManager, PDFDocument, PDFChunk
from pipeline import PDFSummarizerPipeline
from rag_gemini import GeminiRAGPipeline
from models import DocumentOut, UploadResult


class DocumentContent(BaseModel):
    id: int
    filename: str
    broker: Optional[str] = None
    sender_company: Optional[str] = None
    written_date: Optional[date] = None
    rating: Optional[str] = None
    target_price: Optional[float] = None
    tickers: Optional[List[str]] = None
    sector: Optional[str] = None
    report_type: Optional[str] = None
    dense_summary: Optional[str] = None
    pages: List[dict]  # [{page_number, content}]

router = APIRouter()


@router.get("", response_model=list[DocumentOut])
def list_documents(db: DatabaseManager = Depends(get_db)):
    session = db.get_session()
    try:
        docs = session.query(PDFDocument).order_by(PDFDocument.uploaded_at.desc()).all()
        return [DocumentOut.model_validate(d) for d in docs]
    finally:
        session.close()


@router.post("/upload", response_model=UploadResult)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: PDFSummarizerPipeline = Depends(get_pipeline),
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()

    # Write to a temp file named after the original so pipeline stores the right filename
    tmp_dir = tempfile.gettempdir()
    named_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(named_path, "wb") as f:
            f.write(contents)

        result = await run_in_threadpool(
            pipeline.process_single_pdf, named_path, True
        )

        if result["status"] in ("success", "skipped"):
            await run_in_threadpool(rag.backfill_embeddings, 64)

        return UploadResult(
            status=result["status"],
            filename=result["filename"],
            document_id=result.get("document_id"),
            total_pages=result.get("total_pages"),
            message=result.get("message"),
        )
    finally:
        if os.path.exists(named_path):
            os.remove(named_path)


@router.get("/{document_id}/content", response_model=DocumentContent)
def get_document_content(
    document_id: int,
    db: DatabaseManager = Depends(get_db),
):
    session = db.get_session()
    try:
        doc = session.get(PDFDocument, document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        chunks = (
            session.query(PDFChunk)
            .filter(PDFChunk.document_id == document_id)
            .order_by(PDFChunk.page_number.asc().nullslast())
            .all()
        )

        pages = []
        seen_pages = set()
        for c in chunks:
            pg = c.page_number
            if pg in seen_pages:
                continue
            seen_pages.add(pg)
            content = (c.raw_content or "").strip()
            if content:
                pages.append({"page_number": pg, "content": content})

        return DocumentContent(
            id=doc.id,
            filename=doc.filename,
            broker=doc.broker,
            sender_company=doc.sender_company,
            written_date=doc.written_date,
            rating=doc.rating,
            target_price=doc.target_price,
            tickers=doc.tickers,
            sector=doc.sector,
            report_type=doc.report_type,
            dense_summary=doc.dense_summary,
            pages=pages,
        )
    finally:
        session.close()


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: DatabaseManager = Depends(get_db),
):
    deleted = db.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return {"deleted": True, "document_id": document_id}
