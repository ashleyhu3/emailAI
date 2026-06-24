"""Agentic research routes."""
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from dependencies import get_rag
from rag_gemini import GeminiRAGPipeline, RetrievalFilters
from agent import ResearchAgent
from models import AgentRunRequest, AgentResult, SubQueryResult, ChunkRef

router = APIRouter()


@router.post("/run", response_model=AgentResult)
async def run_agent(
    req: AgentRunRequest,
    rag: GeminiRAGPipeline = Depends(get_rag),
):
    filters = RetrievalFilters(
        document_ids=req.document_ids,
        filenames=req.filenames,
        page_min=req.page_min,
        page_max=req.page_max,
    )

    research_agent = ResearchAgent(rag=rag)
    raw = await run_in_threadpool(
        research_agent.run, req.goal, top_k=req.top_k, filters=filters
    )

    return AgentResult(
        goal=raw["goal"],
        sub_queries=[
            SubQueryResult(
                question=sq["question"],
                answer=sq["answer"],
                chunks_used=[ChunkRef(**c) for c in sq["chunks_used"]],
            )
            for sq in raw["sub_queries"]
        ],
        synthesis=raw["synthesis"],
    )
