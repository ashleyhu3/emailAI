"""Document management routes: upload, list, delete."""
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool

from dependencies import get_db, get_pipeline, get_rag
from database import DatabaseManager, PDFDocument
from pipeline import PDFSummarizerPipeline
from rag_gemini import GeminiRAGPipeline
from models import DocumentOut, UploadResult

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


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: DatabaseManager = Depends(get_db),
):
    deleted = db.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return {"deleted": True, "document_id": document_id}
