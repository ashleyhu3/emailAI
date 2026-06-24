"""
Phase 3: Multi-Agent Orchestration Layer.

Ingestion Team (async):
  Agent 1 — Spatial & Graph Deconstructor  (gemini-2.5-flash, multimodal)
  Agent 2 — Core Financial Matrix Extractor (gemini-2.5-flash + context cache)
  Agent 3 — QA Audit Guard                  (gemini-2.5-pro, failsafe only)

RAG Team (interactive queries):
  Agent 4 — Gateway Intent Router    (gemini-2.5-flash, classifies intent)
  Agent 5 — Structured Query Composer (gemini-2.5-flash, SQL + semantic params)
  Agent 6 — Answer Synthesis Engine   (gemini-2.5-pro, cited answer generation)

All agents use the existing google-genai SDK pattern already in pdf_processor.py.
"""

import hashlib
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from google import genai
from google.genai import types

# Resolve sibling imports when running as a standalone module
_PDF_SUMMARIZER = Path(__file__).resolve().parent.parent
if str(_PDF_SUMMARIZER) not in sys.path:
    sys.path.insert(0, str(_PDF_SUMMARIZER))

from broker_cache import BrokerContextCache

# ── Model constants ───────────────────────────────────────────────────────────
_FLASH      = "models/gemini-2.5-flash"
_FLASH_LITE = "models/gemini-2.5-flash"   # swap to gemini-2.5-flash-lite when available
_PRO        = "models/gemini-2.5-pro"

# Disable thinking budget for extraction tasks (straightforward, not reasoning-heavy)
_NO_THINK = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)


def _client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


# ── Shared Pydantic schema ────────────────────────────────────────────────────

class EpsPeData(BaseModel):
    """EPS and P/E estimates extracted from the summary table. All fields optional."""
    eps_fy1: Optional[float] = Field(None, description="Current fiscal year EPS estimate")
    eps_fy2: Optional[float] = Field(None, description="Next fiscal year EPS estimate")
    pe_fy1: Optional[float] = Field(None, description="Current fiscal year P/E multiple")
    pe_fy2: Optional[float] = Field(None, description="Next fiscal year P/E multiple")


class FinancialReportMetadata(BaseModel):
    """
    Structured output schema for Agent 2 (and Agent 3 fallback).
    Maps 1:1 to the new broker-pipeline columns in pdf_documents.
    """
    broker: str
    report_date: Optional[date] = None
    broker_action: Literal["u", "d", "id", "m"]
    rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    target_price: Optional[float] = None
    eps_pe: Optional[EpsPeData] = None
    dense_summary: str

    @field_validator("broker_action", mode="before")
    @classmethod
    def normalize_action(cls, v):
        mapping = {
            "upgrade": "u", "upgraded": "u", "up": "u",
            "downgrade": "d", "downgraded": "d", "down": "d",
            "initiation": "id", "initiate": "id", "initiation of coverage": "id",
            "maintenance": "m", "maintain": "m", "reiterate": "m", "neutral": "m",
        }
        return mapping.get(str(v).lower(), v)


class SQLFilterParams(BaseModel):
    """Structured output for Agent 5a — SQL WHERE clause parameters only."""
    sql_broker: Optional[str] = None
    sql_broker_action: Optional[Literal["u", "d", "id", "m"]] = None
    sql_rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    sql_target_price_min: Optional[float] = None
    sql_target_price_max: Optional[float] = None
    sql_date_from: Optional[str] = None   # YYYY-MM-DD
    sql_date_to: Optional[str] = None     # YYYY-MM-DD
    requires_vector_search: bool = True


class QueryRoutingParams(BaseModel):
    """Combined output for Agent 5 — SQL filters + semantic search query."""
    sql_broker: Optional[str] = None
    sql_broker_action: Optional[Literal["u", "d", "id", "m"]] = None
    sql_rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    sql_target_price_min: Optional[float] = None
    sql_target_price_max: Optional[float] = None
    sql_date_from: Optional[str] = None   # YYYY-MM-DD
    sql_date_to: Optional[str] = None     # YYYY-MM-DD
    semantic_query: str
    requires_vector_search: bool = True


# ── Agent-5 routing cache (in-process, 6-hour TTL) ───────────────────────────

_ROUTE_CACHE: Dict[str, Any] = {}
_ROUTE_CACHE_TTL = 6 * 3600


def _routing_cache_key(query: str) -> str:
    return hashlib.md5(" ".join(query.lower().split()).encode()).hexdigest()


def _get_cached_routing(query: str) -> Optional[QueryRoutingParams]:
    key = _routing_cache_key(query)
    entry = _ROUTE_CACHE.get(key)
    if entry:
        ts, params = entry
        if time.time() - ts < _ROUTE_CACHE_TTL:
            return params
        _ROUTE_CACHE.pop(key, None)
    return None


def _set_cached_routing(query: str, params: QueryRoutingParams) -> None:
    _ROUTE_CACHE[_routing_cache_key(query)] = (time.time(), params)


class GatewayIntentParams(BaseModel):
    """Structured output for Agent 4 Gateway — classifies user intent."""
    intent: Literal["NEW_SEARCH", "FOLLOWUP_Q", "FILE_RETRIEVAL"]
    core_user_ask: str
    active_context_document_ids: List[int] = Field(default_factory=list)


# ── Agent 1: Spatial & Graph Deconstructor ────────────────────────────────────

_AGENT1_SYSTEM = """\
You are an expert financial data layout engine.

TASK: Convert the provided visual asset (chart, graph, or table image) into a dense,
token-optimized Markdown table or data structure.

EXECUTION BOUNDARIES:
1. Reconstruct x-axis / y-axis intersections precisely into a Markdown grid.
2. Prepend spatial indicators for trend data:
   [Top-Legend: Bear Case = $120], [Base Case Line: Sloping upward $140→$180]
3. Zero conversational boilerplate. Output raw, dense Markdown string only.
4. If the image is illegible or contains no structured data, return exactly: [NO_DATA]
"""


def run_agent1(image_bytes: bytes, context_text: str = "", api_key: Optional[str] = None) -> str:
    """
    Agent 1: Convert a cropped chart/graph image to a Markdown data table.

    Args:
        image_bytes: PNG/JPEG bytes of the cropped bounding box from Docling.
        context_text: Surrounding page text snippet for spatial context.

    Returns:
        Markdown string (or "[NO_DATA]" if the image has no structured content).
    """
    c = _client(api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    prompt = f"Spatial context:\n{context_text}\n\nConvert the chart/table above to Markdown:" if context_text else "Convert the chart/table to Markdown:"

    response = c.models.generate_content(
        model=_FLASH,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=_AGENT1_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "[NO_DATA]").strip()


# ── Agent 2: Core Financial Matrix Extractor ──────────────────────────────────

def _run_agent2_raw(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> str:
    """Internal: call Agent 2 and return the raw JSON string before validation."""
    if cache is None:
        cache = BrokerContextCache(api_key=api_key)

    c = cache.get_client()

    try:
        cache_name = cache.get_or_create(broker)
        config = types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=FinancialReportMetadata,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    except Exception:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinancialReportMetadata,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    response = c.models.generate_content(
        model=_FLASH,
        contents=f"Extract metadata from this broker report:\n\n{aoim_text}",
        config=config,
    )
    return response.text or ""


def run_agent2(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """
    Agent 2: Extract structured broker report metadata from AOIM text.

    Uses Gemini Context Caching for a ~90% reduction in instruction tokens.
    Raises ValidationError if the model output fails schema validation.
    Callers should use extract_metadata() which handles retries automatically.
    """
    raw = _run_agent2_raw(aoim_text, broker, cache, api_key)
    return FinancialReportMetadata.model_validate_json(raw)


def _run_agent2_repair(
    failed_json: str,
    error_message: str,
    aoim_text: str,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """
    Flash self-correction: feed the validation error back to Flash to fix it.

    Called after the rule-engine repair fails. Consumes ~1 Flash call instead of
    escalating to the expensive Agent 3 (Pro). Handles ~90% of validation errors
    that reach this stage (simple field formatting issues, enum mismatches, etc.).
    """
    c = _client(api_key)
    prompt = (
        "The JSON output below failed schema validation. "
        "Fix ONLY the invalid fields — do not change correct ones.\n\n"
        f"Validation error:\n{error_message[:400]}\n\n"
        f"Failed output:\n{failed_json}\n\n"
        f"Source (for reference):\n{aoim_text[:1200]}\n\n"
        "Return the corrected JSON only:"
    )
    response = c.models.generate_content(
        model=_FLASH,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinancialReportMetadata,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return FinancialReportMetadata.model_validate_json(response.text)


# ── Agent 3: QA Audit Guard (failsafe only) ───────────────────────────────────

_AGENT3_SYSTEM = """\
You are an elite financial data auditor and deep-reasoning engine.

A lower-tier parsing model has broken validation schema constraints or returned ambiguous
data while processing a broker research report. Your task:

1. Analyze the source text carefully.
2. Pinpoint the exact field(s) that caused the schema violation.
3. Generate a flawless JSON payload that corrects every constraint failure.

Be meticulous. Every field must conform to the schema contract.
"""


def run_agent3(
    aoim_text: str,
    failed_json: str,
    error_message: str,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """
    Agent 3: QA fallback — re-extract with gemini-2.5-pro after Agent 2 fails validation.

    This runs ONLY when Agent 2 throws a ValidationError or Exception.
    """
    c = _client(api_key)
    prompt = (
        f"CRITICAL FIX REQUIRED.\n\n"
        f"Validation error:\n{error_message}\n\n"
        f"Failed JSON attempt:\n{failed_json}\n\n"
        f"Source data (AOIM):\n{aoim_text}\n\n"
        f"Generate a corrected, schema-compliant JSON object:"
    )
    response = c.models.generate_content(
        model=_PRO,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_AGENT3_SYSTEM,
            response_mime_type="application/json",
            response_schema=FinancialReportMetadata,
        ),
    )
    return FinancialReportMetadata.model_validate_json(response.text)


# ── Agent 4: Gateway Intent Router ───────────────────────────────────────────

_AGENT4_GATEWAY_SYSTEM = """\
You are the Gateway Intent Router for a financial research RAG system.

Analyze the CURRENT QUERY against the CONVERSATION HISTORY and classify the user's intent
as exactly one of three values:

  NEW_SEARCH      — The user wants to find new documents or data not yet discussed.
  FOLLOWUP_Q      — The user is asking a follow-up about documents already cited in
                    the conversation. The answer should come from those same docs.
  FILE_RETRIEVAL  — The user wants a download link, source file, or raw metadata for
                    a document already referenced (e.g. "give me that PDF", "send the
                    Morgan Stanley report").

FOLLOW-UP DETECTION RULES:
- Pronouns referencing prior subjects ("their", "its", "that report", "the same company")
  are strong follow-up signals.
- A question that extends or drills into a topic from the previous assistant turn is a
  follow-up even if it introduces a new specific angle.
- A question about a NEW company/ticker not discussed yet is always NEW_SEARCH.

OUTPUT:
- intent: one of the three literals above.
- core_user_ask: a concise rephrasing of the underlying business question (resolve all
  pronouns — "their gross margins" → "Apple gross margins discussed in prior answer").
- active_context_document_ids: for FOLLOWUP_Q or FILE_RETRIEVAL, extract the integer
  document IDs that were cited in previous assistant turns (visible in the history as
  "docs:<id>,<id>"). Return [] for NEW_SEARCH.
"""


def run_agent4_gateway(
    user_query: str,
    history: Optional[List[dict]] = None,
    api_key: Optional[str] = None,
) -> GatewayIntentParams:
    """
    Agent 4: Classify user intent and extract active document context from history.

    Args:
        user_query: The user's current message.
        history: List of {"role": str, "content": str, "document_ids": list[int]}.
                 Assistant turns should include document_ids from the prior response.
    """
    c = _client(api_key)

    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:  # last 3 exchanges
            role = msg.get("role", "user").upper()
            doc_ids = msg.get("document_ids") or []
            doc_suffix = f" | docs:{','.join(str(d) for d in doc_ids)}" if doc_ids else ""
            lines.append(f"[{role}{doc_suffix}]: {msg['content']}")
        history_block = "=== CONVERSATION HISTORY ===\n" + "\n".join(lines) + "\n=== END HISTORY ===\n\n"

    prompt = f"{history_block}CURRENT QUERY: {user_query}"

    response = c.models.generate_content(
        model=_FLASH_LITE,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_AGENT4_GATEWAY_SYSTEM,
            response_mime_type="application/json",
            response_schema=GatewayIntentParams,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return GatewayIntentParams.model_validate_json(response.text)


# ── Agent 5a: SQL Filter Generator ───────────────────────────────────────────

_AGENT5A_SYSTEM = """\
You are a financial database filter agent.

Translate the user's query into SQL WHERE clause parameters for the pdf_documents table.
Populate ONLY the fields that are explicitly or implicitly requested.

SCHEMA:
  broker         (String)  — authoring institution name
  broker_action  (Enum)    — 'u' Upgrade | 'd' Downgrade | 'id' Initiation | 'm' Maintenance
  rating         (Enum)    — 'Overweight' | 'Equal-weight' | 'Underweight'
  target_price   (Float)   — numeric range via sql_target_price_min / sql_target_price_max
  sent_date      (Date)    — ISO 8601 range via sql_date_from / sql_date_to

Set requires_vector_search=false ONLY if the question needs no content search
(e.g. "list all upgrades this week" — answered by SQL alone).
"""


def run_agent5a_sql_filter(
    user_query: str,
    api_key: Optional[str] = None,
) -> SQLFilterParams:
    """Agent 5a: Generate SQL filter parameters from a user query."""
    c = _client(api_key)
    response = c.models.generate_content(
        model=_FLASH,
        contents=f"Generate SQL filters for: {user_query}",
        config=types.GenerateContentConfig(
            system_instruction=_AGENT5A_SYSTEM,
            response_mime_type="application/json",
            response_schema=SQLFilterParams,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return SQLFilterParams.model_validate_json(response.text)


# ── Agent 5b: Semantic Query Refiner ─────────────────────────────────────────

_AGENT5B_SYSTEM = """\
You are a semantic search query optimizer for a financial research database.

Rewrite the user's natural language query into a concise, keyword-dense string
optimized for vector similarity search over document content.

Rules:
- Resolve all pronouns to their referents (e.g. "their gross margins" → "Apple gross margins").
- Remove filler words (is, the, what, tell me about).
- Expand common abbreviations (TP → target price, EPS → earnings per share).
- Preserve company names, tickers, sector terms, and analyst jargon exactly.
- Output ONLY the optimized query string — no explanation, no punctuation.
"""


def run_agent5b_semantic_refiner(
    user_query: str,
    api_key: Optional[str] = None,
) -> str:
    """Agent 5b: Rewrite a user query into a keyword-dense vector search string."""
    c = _client(api_key)
    response = c.models.generate_content(
        model=_FLASH_LITE,
        contents=f"Optimize for vector search: {user_query}",
        config=types.GenerateContentConfig(
            system_instruction=_AGENT5B_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or user_query).strip()


# ── Agent 5: Combined Query Composer (5a + 5b in parallel) ───────────────────

def run_agent5_query_composer(
    user_query: str,
    api_key: Optional[str] = None,
) -> QueryRoutingParams:
    """
    Agent 5: Translate a natural language query into SQL filters + semantic search string.

    Runs Agent 5a (SQL filter generator) and Agent 5b (semantic query refiner) in
    parallel via a ThreadPoolExecutor, then merges their outputs. Results are cached
    in-process for 6 hours so repeated identical queries skip both LLM calls entirely.

    Only called on NEW_SEARCH intent from the Gateway (Agent 4).
    """
    import concurrent.futures

    # Fast path: return cached result for identical (or near-identical) query text
    cached = _get_cached_routing(user_query)
    if cached is not None:
        print("[Agent5] Routing cache hit — skipping LLM calls")
        return cached

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_sql = pool.submit(run_agent5a_sql_filter, user_query, api_key)
        f_sem = pool.submit(run_agent5b_semantic_refiner, user_query, api_key)

        try:
            sql_params = f_sql.result(timeout=30)
        except Exception as e:
            print(f"[Agent5a] Failed: {e} — using empty SQL filters")
            sql_params = SQLFilterParams()

        try:
            semantic_query = f_sem.result(timeout=30)
        except Exception as e:
            print(f"[Agent5b] Failed: {e} — using original query as semantic query")
            semantic_query = user_query

    result = QueryRoutingParams(
        sql_broker=sql_params.sql_broker,
        sql_broker_action=sql_params.sql_broker_action,
        sql_rating=sql_params.sql_rating,
        sql_target_price_min=sql_params.sql_target_price_min,
        sql_target_price_max=sql_params.sql_target_price_max,
        sql_date_from=sql_params.sql_date_from,
        sql_date_to=sql_params.sql_date_to,
        semantic_query=semantic_query,
        requires_vector_search=sql_params.requires_vector_search,
    )

    _set_cached_routing(user_query, result)
    return result


# Backward-compat alias — existing callers that import run_agent4 continue to work
run_agent4 = run_agent5_query_composer


# ── Agent 6: Answer Synthesis Engine ─────────────────────────────────────────

_AGENT6_SYSTEM = """\
You are an elite financial research synthesis engine.

You receive a set of source excerpts from broker research reports and a user question.
Your job is to synthesize a precise, well-structured answer that draws ONLY from the
provided sources.

CITATION RULES (strictly enforced):
- Organize the answer into short focused paragraphs.
- Do NOT scatter citations after every sentence. Write the prose cleanly first, then
  place citations at the END of each paragraph on their own line:
    Sources: (filename.pdf, p.3), (filename.pdf, p.7)
- Cite ONLY the page that directly supports a specific claim in that paragraph.
- Do not cite neighboring context pages unless you actually used a fact from them.
- The citation set per paragraph should be the MINIMAL set that backs your stated facts.

ANSWER RULES:
- If the user asks for a numbered list, format as:
    1) Claim here.
       Source: (filename.pdf, p.N)
- Never fabricate facts not present in the source excerpts.
- If no source excerpt addresses the question, say so explicitly.
- For follow-up questions, use the conversation history for context but still cite the
  source excerpts for every factual claim.
"""


def run_agent6_synthesis(
    user_query: str,
    formatted_context: str,
    history: Optional[List[dict]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Agent 6: Synthesize a cited answer from retrieved document chunks.

    Args:
        user_query: The user's current question (pronouns already resolved by Agent 4).
        formatted_context: Pre-formatted source blocks, one per chunk.
        history: Recent chat turns for conversational context (not used for retrieval).
    """
    c = _client(api_key)

    history_block = ""
    if history:
        lines = [
            f"[{msg.get('role', 'user').upper()}]: {msg['content']}"
            for msg in history[-4:]
        ]
        history_block = "=== RECENT CONVERSATION ===\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"{history_block}"
        f"=== SOURCE EXCERPTS ===\n{formatted_context}\n\n"
        f"=== QUESTION ===\n{user_query}"
    )

    response = c.models.generate_content(
        model=_PRO,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_AGENT6_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    return (response.text or "").strip()


# ── Orchestrated extraction (4-stage repair chain) ────────────────────────────

def extract_metadata(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """
    Extract structured metadata using a 4-stage repair chain.

    Stage 1 — Agent 2 (Flash, cached): normal extraction path.
    Stage 2 — Rule engine: fix obvious formatting errors without any LLM token cost.
    Stage 3 — Agent 2 Flash self-correction: feed the error back to Flash to fix it.
    Stage 4 — Agent 3 (Pro): last resort, called <2% of the time.
    """
    import json as _json

    raw_json = ""
    error_msg = ""

    # Stage 1: Agent 2 normal extraction
    try:
        raw_json = _run_agent2_raw(aoim_text, broker, cache=cache, api_key=api_key)
        return FinancialReportMetadata.model_validate_json(raw_json)
    except Exception as e:
        error_msg = str(e)
        print(f"[Agent2] Stage 1 failed ({broker}): {error_msg[:80]}")

    # Stage 2: Rule-engine repair (zero LLM cost — fixes ~50% of failures)
    if raw_json:
        try:
            from ingest.extractor import repair_metadata_fields
            data = _json.loads(raw_json)
            data = repair_metadata_fields(data)
            result = FinancialReportMetadata.model_validate(data)
            print("[Agent2] Stage 2 rule-engine repair succeeded")
            return result
        except Exception:
            pass

    # Stage 3: Flash self-correction (~40% of remaining failures)
    try:
        result = _run_agent2_repair(raw_json, error_msg, aoim_text, api_key=api_key)
        print("[Agent2] Stage 3 Flash self-correction succeeded")
        return result
    except Exception as e2:
        error_msg = str(e2)
        print(f"[Agent2] Stage 3 Flash repair failed: {error_msg[:80]} — escalating to Agent 3")

    # Stage 4: Agent 3 Pro (true last resort, <2% of emails)
    return run_agent3(aoim_text, raw_json, error_msg, api_key=api_key)
