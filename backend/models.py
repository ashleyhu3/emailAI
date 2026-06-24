"""Pydantic request/response schemas for the FastAPI backend."""
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel


# ── Documents ──────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: int
    filename: str
    total_pages: int
    file_size_bytes: int
    sender_name: Optional[str] = None
    sender_company: Optional[str] = None
    sent_date: Optional[date] = None
    written_date: Optional[date] = None
    tickers: Optional[List[str]] = None
    report_type: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    coverage_period_from: Optional[date] = None
    coverage_period_to: Optional[date] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UploadResult(BaseModel):
    status: str          # "success" | "skipped" | "error"
    filename: str
    document_id: Optional[int] = None
    total_pages: Optional[int] = None
    message: Optional[str] = None


# ── Queries ────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    history: Optional[List[HistoryMessage]] = None
    document_ids: Optional[List[int]] = None
    filenames: Optional[List[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    sender_names: Optional[List[str]] = None
    sender_companies: Optional[List[str]] = None
    written_date_from: Optional[date] = None
    written_date_to: Optional[date] = None
    tickers: Optional[List[str]] = None
    report_type: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    coverage_period_from: Optional[date] = None
    coverage_period_to: Optional[date] = None


class ChunkRef(BaseModel):
    chunk_id: str
    document_id: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    chunks_used: List[ChunkRef]
    inferred_filters: Optional[Dict[str, Any]] = None
    query_type: str = "rag"
    is_enumeration: bool = False


class BackfillResponse(BaseModel):
    embedded_count: int


# ── Agent ──────────────────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str
    top_k: int = 3
    document_ids: Optional[List[int]] = None
    filenames: Optional[List[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class SubQueryResult(BaseModel):
    question: str
    answer: str
    chunks_used: List[ChunkRef]


class AgentResult(BaseModel):
    goal: str
    sub_queries: List[SubQueryResult]
    synthesis: str


# ── Canvas ─────────────────────────────────────────────────────────────────

class CanvasMeta(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CanvasCreateRequest(BaseModel):
    name: str = "Untitled Canvas"


class CanvasSaveRequest(BaseModel):
    name: Optional[str] = None
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


class CanvasDetail(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []


# ── Ingest pipeline ────────────────────────────────────────────────────────────

class IngestTriggerRequest(BaseModel):
    max_emails: int = 50


class IngestTriggerResponse(BaseModel):
    total_fetched: int
    ingested: int
    skipped: int
    errors: int
    results: List[Dict[str, Any]] = []
    triggered_at: str


class IngestStatusResponse(BaseModel):
    total_documents: int
    email_ingested: int
    manual_uploads: int
    last_ingested_at: Optional[str] = None
    recent: List[Dict[str, Any]] = []


# ── RAG Cluster: Query Routing ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """One turn in a conversation. Assistant turns should include document_ids
    from the prior RouteQueryResponse.referenced_document_ids so Agent 4 can
    detect follow-up intent and skip unnecessary full-corpus scans."""
    role: Literal["user", "assistant"]
    content: str
    document_ids: Optional[List[int]] = None


class RouteQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    history: List[ChatMessage] = []


class RouteQueryResponse(BaseModel):
    question: str
    intent: str = "NEW_SEARCH"
    core_user_ask: str = ""
    sql_filters: Dict[str, Any] = {}
    semantic_query: str = ""
    candidate_doc_ids: List[int] = []
    answer: str = ""
    chunks_used: List[Any] = []
    referenced_document_ids: List[int] = []
