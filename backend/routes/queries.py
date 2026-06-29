"""RAG query routes."""
import re
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from dependencies import get_rag
from rag_gemini import GeminiRAGPipeline, RetrievalFilters
from models import AskRequest, AskResponse, BackfillResponse, ChunkRef, HistoryMessage

router = APIRouter()

# Patterns that trigger a Morgan Stanley chart response
_MS_CHART_RE = re.compile(
    r"(?:show|get|give|list|display|chart|graph|plot|visuali[sz]e).*"
    r"(?:morgan\s*stanley|ms|m\.s\.).*"
    r"(?:report|research|coverage|note|idea|initiat|upgrade|downgrade|rating)",
    re.IGNORECASE,
)
_WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12}


def _parse_days(question: str) -> int:
    """Return the requested lookback window in days."""
    q = question.lower()
    # weeks
    if re.search(r"(?:past|last|this)\s+week", q):
        return 7
    m = re.search(r"(?:past|last)\s+(\d+)\s+weeks?", q)
    if m:
        return int(m.group(1)) * 7
    # days
    m = re.search(r"(?:past|last)\s+(\d+)\s+days?", q)
    if m:
        return int(m.group(1))
    # months
    m = re.search(r"(?:past|last)\s+(one|two|three|six|twelve|\d+)\s+months?", q)
    if m:
        raw = m.group(1)
        n = _WORD_TO_NUM.get(raw, int(raw) if raw.isdigit() else 3)
        return n * 30
    if re.search(r"\bmonth\b", q):
        return 30
    return 90  # default 3 months


def _days_to_label(days: int) -> str:
    if days <= 7:
        return "past week"
    if days <= 14:
        return "past 2 weeks"
    if days % 30 == 0:
        n = days // 30
        return f"past {n} month{'s' if n != 1 else ''}"
    return f"past {days} days"


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    # Chart shortcut — detect "show me MS reports" style queries
    if _MS_CHART_RE.search(req.question):
        days = _parse_days(req.question)
        try:
            from charts_util import generate_ms_research_chart
            chart_html, n_companies = await run_in_threadpool(
                generate_ms_research_chart, days
            )
            time_label = _days_to_label(days)
            if n_companies == 0:
                answer = f"No Morgan Stanley research found for the {time_label}. Try running an ingest first."
                chart_html = None
            else:
                answer = (
                    f"Here is the Morgan Stanley research risk-reward chart for the {time_label} "
                    f"({n_companies} compan{'ies' if n_companies != 1 else 'y'})."
                )
            return AskResponse(
                answer=answer,
                chunks_used=[],
                query_type="chart",
                chart_html=chart_html,
            )
        except Exception as e:
            pass  # fall through to normal RAG on chart failure

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
