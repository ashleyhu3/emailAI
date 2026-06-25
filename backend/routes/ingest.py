"""
Ingest router — manual trigger + status endpoints.

POST /ingest/upload-eml — test pipeline with a single .eml file
POST /ingest/trigger    — kick off a synchronous email ingest run
GET  /ingest/status     — last run stats from the DB
GET  /ingest/test-email — verify IMAP credentials
POST /ingest/route      — Agent 4 query routing
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

# Resolve PDF_summarizer on the path (mirrors dependencies.py).
# All heavy imports (DatabaseManager singletons) are deferred into function
# bodies so importing this module never triggers a DB connection at reload time.
_backend_dir = Path(__file__).resolve().parent.parent
_pdf_summarizer_dir = _backend_dir.parent / "PDF_summarizer"
if str(_pdf_summarizer_dir) not in sys.path:
    sys.path.insert(0, str(_pdf_summarizer_dir))

from models import (
    IngestTriggerRequest,
    IngestTriggerResponse,
    IngestStatusResponse,
    RouteQueryRequest,
    RouteQueryResponse,
)

router = APIRouter()


# ── GET /ingest/gmail-setup ───────────────────────────────────────────────────

@router.get("/gmail-setup")
def gmail_setup():
    """
    Check Gmail OAuth2 status. Returns instructions if not yet authorized.
    Run the one-time setup from the terminal:
      python -m ingest.gmail_fetcher --setup
    """
    try:
        from ingest.gmail_fetcher import is_available, _credentials_path, _token_path
        creds_path = _credentials_path()
        token_path = _token_path()
        authorized = is_available()
        return {
            "authorized": authorized,
            "credentials_file": str(creds_path) if creds_path else None,
            "token_file": str(token_path),
            "token_exists": token_path.exists(),
            "instructions": (
                None if authorized else
                "1. Set GMAIL_CREDENTIALS_FILE in .env pointing to your OAuth client JSON. "
                "2. Run: python -m ingest.gmail_fetcher --setup (opens browser once). "
                "3. Restart the server."
            ),
        }
    except Exception as e:
        return {"authorized": False, "error": str(e)}


# ── GET /ingest/test-email ────────────────────────────────────────────────────

@router.get("/test-email")
def test_email_connection():
    """
    Verify the IMAP credentials in .env and preview broker-matching messages.
    Fetches only headers (no body download) — safe to call at any time.
    """
    from ingest.email_fetcher import test_connection
    return test_connection()


# ── POST /ingest/upload-eml ───────────────────────────────────────────────────

@router.post("/upload-eml")
async def upload_eml(
    file: UploadFile = File(..., description=".eml file exported from any email client"),
    broker: Optional[str] = Form(None, description="Override broker name (optional)"),
):
    """
    Test the ingest pipeline by uploading a single .eml file.

    Runs the same pipeline as the scheduled IMAP fetch:
    HTML cleaning → Docling PDF slicing (pages 1–2) → Agent 1 (charts) →
    Agent 2 (matrix extractor) → Agent 3 fallback → PostgreSQL commit.

    Returns status='skipped' if the same email hash is already in the DB.
    """
    if not file.filename or not file.filename.endswith(".eml"):
        raise HTTPException(status_code=400, detail="File must be a .eml (RFC 822) file")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        from ingest.email_fetcher import parse_eml_bytes
        payload = parse_eml_bytes(raw, broker_override=broker)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse .eml: {e}")

    try:
        from ingest.worker import _process_email
        # Run the blocking pipeline in a thread so the async event loop stays free
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_email, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        **result,
        "filename": file.filename,
        "sender": payload.sender,
        "broker_detected": payload.broker,
        "report_date": payload.report_date.isoformat() if payload.report_date else None,
        "pdf_attachments": [fn for fn, _ in payload.pdf_attachments],
    }


# ── POST /ingest/upload-pdf ──────────────────────────────────────────────────

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    broker: str = Form(..., description="Broker name (e.g. 'Daiwa Capital Markets')"),
    sender: Optional[str] = Form(None, description="Sender email address (optional)"),
):
    """
    Ingest a PDF directly — useful for PDFs downloaded manually from broker portals.

    Runs the same pipeline as the email path but skips email parsing:
    Docling PDF slicing → Agent 1 (charts) → Agent 2 (metadata) → PostgreSQL commit.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a .pdf")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if pdf_bytes[:4] != b"%PDF":
        raise HTTPException(status_code=422, detail="File does not appear to be a valid PDF")

    from datetime import datetime, timezone
    from ingest.email_fetcher import EmailPayload
    import hashlib

    payload = EmailPayload(
        message_id=hashlib.sha256(pdf_bytes).hexdigest(),
        sender=sender or f"manual-upload@{broker.lower().replace(' ', '')}.upload",
        sender_domain="manual-upload",
        broker=broker,
        report_date=datetime.now(timezone.utc),
        html_body=None,
        text_body=None,
        pdf_attachments=[(file.filename, pdf_bytes)],
    )

    try:
        from ingest.worker import _process_email
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_email, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        **result,
        "filename": file.filename,
        "broker_detected": broker,
    }


# ── POST /ingest/trigger ──────────────────────────────────────────────────────

@router.post("/trigger", response_model=IngestTriggerResponse)
async def trigger_ingest(req: IngestTriggerRequest):
    """Manually kick off the email ingest pipeline (no Celery required)."""
    try:
        from ingest.worker import run_ingest_now
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, run_ingest_now, req.max_emails)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestTriggerResponse(
        total_fetched=len(results),
        ingested=sum(1 for r in results if r.get("status") == "success"),
        skipped=sum(1 for r in results if r.get("status") == "skipped"),
        errors=sum(1 for r in results if r.get("status") == "error"),
        results=results,
        triggered_at=datetime.utcnow().isoformat(),
    )


# ── GET /ingest/status ────────────────────────────────────────────────────────

@router.get("/status", response_model=IngestStatusResponse)
def ingest_status():
    """Return counts and latest ingestion metadata from the database."""
    from dependencies import get_db
    from sqlalchemy import text as sa_text

    db = get_db()
    session = db.get_session()
    try:
        row = session.execute(sa_text("""
            SELECT
                COUNT(*) FILTER (WHERE ingest_source = 'email') AS email_count,
                COUNT(*) FILTER (WHERE ingest_source != 'email' OR ingest_source IS NULL) AS upload_count,
                COUNT(*) AS total_count,
                MAX(uploaded_at) AS last_ingested_at
            FROM pdf_documents
        """)).fetchone()

        recent = session.execute(sa_text("""
            SELECT broker, broker_action, rating, target_price, uploaded_at
            FROM pdf_documents
            WHERE ingest_source = 'email'
            ORDER BY uploaded_at DESC
            LIMIT 10
        """)).fetchall()
    finally:
        session.close()

    return IngestStatusResponse(
        total_documents=row[2] or 0,
        email_ingested=row[0] or 0,
        manual_uploads=row[1] or 0,
        last_ingested_at=row[3].isoformat() if row[3] else None,
        recent=[
            {"broker": r[0], "broker_action": r[1], "rating": r[2],
             "target_price": r[3], "uploaded_at": r[4].isoformat() if r[4] else None}
            for r in recent
        ],
    )


# ── POST /ingest/route ────────────────────────────────────────────────────────

def _format_chunks_for_synthesis(chunks: List[Any], doc_meta: Dict[int, Any]) -> str:
    """Format retrieved PDFChunk objects into a labelled context block for Agent 6."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = doc_meta.get(chunk.document_id, {})
        header = (
            f"[Source {i} | Doc {chunk.document_id} | "
            f"{meta.get('broker', '?')} | {meta.get('date', '?')} | "
            f"{meta.get('filename', '?')}]"
        )
        page = f"(Page {chunk.page_number})" if chunk.page_number else ""
        content = (chunk.raw_content or "").strip()
        parts.append(f"{header}\n{page}\n{content}" if page else f"{header}\n{content}")
    return "\n\n".join(parts)


@router.post("/route", response_model=RouteQueryResponse)
async def route_query(req: RouteQueryRequest):
    """
    Multi-Agent RAG Cluster — 3-phase interactive query pipeline.

    Phase 1  Agent 4 Gateway classifies intent (NEW_SEARCH / FOLLOWUP_Q / FILE_RETRIEVAL).
    Phase 2  For NEW_SEARCH: Agent 5 builds SQL + semantic params → pgvector retrieval.
             For FOLLOWUP_Q: skip global scan, restrict retrieval to active document IDs.
             For FILE_RETRIEVAL: return document metadata directly.
    Phase 3  Agent 6 synthesizes a cited answer from retrieved chunks using gemini-2.5-pro.
    """
    from dependencies import get_db, get_rag
    from ingest.query_agents import run_agent4_gateway, run_agent5_query_composer, run_agent6_synthesis
    from rag_gemini import RetrievalFilters
    from sqlalchemy import text as sa_text

    # Build plain-dict history for agent calls (Pydantic → dict)
    history_dicts = [
        {"role": m.role, "content": m.content, "document_ids": m.document_ids or []}
        for m in req.history
    ]

    # ── Phase 1: Intent classification (Agent 4 Gateway) ─────────────────────
    loop = asyncio.get_event_loop()
    try:
        gateway = await loop.run_in_executor(
            None, run_agent4_gateway, req.question, history_dicts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 4 gateway failed: {e}")

    db = get_db()

    # ── Phase 2A: FILE_RETRIEVAL — return metadata directly ──────────────────
    if gateway.intent == "FILE_RETRIEVAL" and gateway.active_context_document_ids:
        session = db.get_session()
        try:
            rows = session.execute(sa_text("""
                SELECT id, broker, filename, sent_date, total_pages, file_size_bytes
                FROM pdf_documents WHERE id = ANY(:ids)
            """), {"ids": gateway.active_context_document_ids}).fetchall()
        finally:
            session.close()

        docs_info = [
            {
                "document_id": r[0], "broker": r[1], "filename": r[2],
                "date": r[3].isoformat() if r[3] else None,
                "total_pages": r[4], "file_size_bytes": r[5],
            }
            for r in rows
        ]
        import json as _json
        return RouteQueryResponse(
            question=req.question,
            intent=gateway.intent,
            core_user_ask=gateway.core_user_ask,
            candidate_doc_ids=gateway.active_context_document_ids,
            answer=_json.dumps(docs_info, indent=2),
            referenced_document_ids=gateway.active_context_document_ids,
        )

    # ── Phase 2B: Retrieval ───────────────────────────────────────────────────
    chunks: List[Any] = []
    sql_filters_out: Dict[str, Any] = {}
    semantic_query = gateway.core_user_ask or req.question

    rag = get_rag()

    if gateway.intent == "FOLLOWUP_Q" and gateway.active_context_document_ids:
        # Bypass full-corpus scan — restrict to the already-discussed document IDs.
        filters = RetrievalFilters(document_ids=list(gateway.active_context_document_ids))
        try:
            from rag_gemini import GeminiRAGPipeline
            chunks = await loop.run_in_executor(
                None,
                lambda: rag.retrieve_relevant_chunks(
                    query=semantic_query,
                    top_k=req.top_k,
                    filters=filters,
                    similarity_threshold=GeminiRAGPipeline.FOLLOWUP_SIMILARITY_THRESHOLD
                    if hasattr(GeminiRAGPipeline, "FOLLOWUP_SIMILARITY_THRESHOLD")
                    else 0.0,
                ),
            )
        except Exception as e:
            chunks = []
            print(f"[route] FOLLOWUP_Q retrieval failed: {e}")

    else:
        # NEW_SEARCH — Agent 5 builds SQL filters + semantic query
        gateway.intent = "NEW_SEARCH"
        try:
            routing = await loop.run_in_executor(
                None, run_agent5_query_composer, req.question
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent 5 query composer failed: {e}")

        semantic_query = routing.semantic_query
        sql_filters_out = {k: v for k, v in {
            "broker": routing.sql_broker,
            "broker_action": routing.sql_broker_action,
            "rating": routing.sql_rating,
            "date_from": routing.sql_date_from,
            "date_to": routing.sql_date_to,
        }.items() if v is not None}

        # SQL filter pass on pdf_documents
        session = db.get_session()
        try:
            clauses: List[str] = []
            params: Dict[str, Any] = {}
            if routing.sql_broker:
                clauses.append("broker ILIKE :broker")
                params["broker"] = f"%{routing.sql_broker}%"
            if routing.sql_broker_action:
                clauses.append("broker_action = :broker_action")
                params["broker_action"] = routing.sql_broker_action
            if routing.sql_rating:
                clauses.append("rating = :rating")
                params["rating"] = routing.sql_rating
            if routing.sql_target_price_min is not None:
                clauses.append("target_price >= :tp_min")
                params["tp_min"] = routing.sql_target_price_min
            if routing.sql_target_price_max is not None:
                clauses.append("target_price <= :tp_max")
                params["tp_max"] = routing.sql_target_price_max
            if routing.sql_date_from:
                clauses.append("sent_date >= :date_from")
                params["date_from"] = routing.sql_date_from
            if routing.sql_date_to:
                clauses.append("sent_date <= :date_to")
                params["date_to"] = routing.sql_date_to

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            filtered = session.execute(sa_text(f"""
                SELECT id, dense_summary FROM pdf_documents {where}
                ORDER BY sent_date DESC LIMIT 100
            """), params).fetchall()
        finally:
            session.close()

        # Lightweight semantic pre-filter on dense_summary before pgvector scan
        kws = routing.semantic_query.lower().split()
        candidate_ids = [
            r[0] for r in filtered
            if not r[1] or any(kw in (r[1] or "").lower() for kw in kws)
        ]

        if not candidate_ids:
            return RouteQueryResponse(
                question=req.question,
                intent=gateway.intent,
                core_user_ask=gateway.core_user_ask,
                sql_filters=sql_filters_out,
                semantic_query=semantic_query,
                answer="No documents matched your query filters.",
                referenced_document_ids=[],
            )

        if routing.requires_vector_search:
            filters = RetrievalFilters(document_ids=candidate_ids)
            try:
                chunks = await loop.run_in_executor(
                    None,
                    lambda: rag.retrieve_relevant_chunks(
                        query=routing.semantic_query,
                        top_k=req.top_k,
                        filters=filters,
                    ),
                )
            except Exception as e:
                chunks = []
                print(f"[route] NEW_SEARCH retrieval failed: {e}")

    # ── Phase 3: Agent 6 synthesis ────────────────────────────────────────────
    if not chunks:
        return RouteQueryResponse(
            question=req.question,
            intent=gateway.intent,
            core_user_ask=gateway.core_user_ask,
            sql_filters=sql_filters_out,
            semantic_query=semantic_query,
            answer="No relevant content found for your query.",
            referenced_document_ids=[],
        )

    # Batch-fetch document metadata for chunk formatting
    doc_ids_needed = list({c.document_id for c in chunks})
    session = db.get_session()
    try:
        doc_rows = session.execute(sa_text("""
            SELECT id, broker, filename, sent_date FROM pdf_documents
            WHERE id = ANY(:ids)
        """), {"ids": doc_ids_needed}).fetchall()
    finally:
        session.close()

    doc_meta: Dict[int, Any] = {
        r[0]: {
            "broker": r[1],
            "filename": r[2],
            "date": r[3].isoformat() if r[3] else None,
        }
        for r in doc_rows
    }

    formatted_context = _format_chunks_for_synthesis(chunks, doc_meta)
    chunks_used_out = [
        {"chunk_id": str(c.id), "document_id": c.document_id, "page_number": c.page_number}
        for c in chunks
    ]
    referenced_ids = doc_ids_needed

    try:
        answer = await loop.run_in_executor(
            None,
            lambda: run_agent6_synthesis(
                user_query=req.question,
                formatted_context=formatted_context,
                history=history_dicts,
            ),
        )
    except Exception as e:
        answer = f"Synthesis error: {e}"

    return RouteQueryResponse(
        question=req.question,
        intent=gateway.intent,
        core_user_ask=gateway.core_user_ask,
        sql_filters=sql_filters_out,
        semantic_query=semantic_query,
        candidate_doc_ids=doc_ids_needed,
        answer=answer,
        chunks_used=chunks_used_out,
        referenced_document_ids=referenced_ids,
    )
