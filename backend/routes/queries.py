"""RAG query routes."""
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from dependencies import get_rag
from rag_gemini import GeminiRAGPipeline, RetrievalFilters
from models import AskRequest, AskResponse, BackfillResponse, ChunkRef, HistoryMessage

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    filters = RetrievalFilters(
        document_ids=req.document_ids,
        filenames=req.filenames,
        page_min=req.page_min,
        page_max=req.page_max,
        sender_names=req.sender_names,
        sender_companies=req.sender_companies,
        written_date_from=req.written_date_from,
        written_date_to=req.written_date_to,
        tickers=req.tickers,
        report_type=req.report_type,
        sector=req.sector,
        asset_class=req.asset_class,
        coverage_period_from=req.coverage_period_from,
        coverage_period_to=req.coverage_period_to,
    )
    history = [m.model_dump() for m in req.history] if req.history else None
    result = await run_in_threadpool(
        rag.answer_question, req.question, top_k=req.top_k, filters=filters,
        history=history,
    )
    return AskResponse(
        answer=result["answer"],
        chunks_used=[ChunkRef(**c) for c in result["chunks_used"]],
        inferred_filters=result.get("inferred_filters"),
        query_type=result.get("query_type", "rag"),
        is_enumeration=result.get("is_enumeration", False),
    )


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_embeddings(
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    count = await run_in_threadpool(
        rag.backfill_embeddings, batch_size=batch_size, max_batches=max_batches
    )
    return BackfillResponse(embedded_count=count)
