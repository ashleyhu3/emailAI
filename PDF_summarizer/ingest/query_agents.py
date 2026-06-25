"""
RAG query agents (Agents 4–6) — interactive query team.

  Agent 4 — Gateway Intent Router     (gemini-2.5-flash, classifies NEW_SEARCH / FOLLOWUP_Q / FILE_RETRIEVAL)
  Agent 5 — Structured Query Composer (gemini-2.5-flash, SQL filters + semantic query, parallel 5a+5b)
  Agent 6 — Answer Synthesis Engine   (gemini-2.5-pro, cited answer generation)
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

_PDF_SUMMARIZER = Path(__file__).resolve().parent.parent
if str(_PDF_SUMMARIZER) not in sys.path:
    sys.path.insert(0, str(_PDF_SUMMARIZER))

_FLASH      = "models/gemini-2.5-flash"
_FLASH_LITE = "models/gemini-2.5-flash"
_PRO        = "models/gemini-2.5-pro"


def _client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GatewayIntentParams(BaseModel):
    intent: Literal["NEW_SEARCH", "FOLLOWUP_Q", "FILE_RETRIEVAL"]
    core_user_ask: str
    active_context_document_ids: List[int] = Field(default_factory=list)


class SQLFilterParams(BaseModel):
    sql_broker: Optional[str] = None
    sql_broker_action: Optional[Literal["u", "d", "id", "m"]] = None
    sql_rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    sql_target_price_min: Optional[float] = None
    sql_target_price_max: Optional[float] = None
    sql_date_from: Optional[str] = None
    sql_date_to: Optional[str] = None
    requires_vector_search: bool = True


class QueryRoutingParams(BaseModel):
    sql_broker: Optional[str] = None
    sql_broker_action: Optional[Literal["u", "d", "id", "m"]] = None
    sql_rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    sql_target_price_min: Optional[float] = None
    sql_target_price_max: Optional[float] = None
    sql_date_from: Optional[str] = None
    sql_date_to: Optional[str] = None
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
    """Agent 4: Classify user intent and extract active document context from history."""
    c = _client(api_key)

    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            doc_ids = msg.get("document_ids") or []
            doc_suffix = f" | docs:{','.join(str(d) for d in doc_ids)}" if doc_ids else ""
            lines.append(f"[{role}{doc_suffix}]: {msg['content']}")
        history_block = "=== CONVERSATION HISTORY ===\n" + "\n".join(lines) + "\n=== END HISTORY ===\n\n"

    response = c.models.generate_content(
        model=_FLASH_LITE,
        contents=f"{history_block}CURRENT QUERY: {user_query}",
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
    Agent 5: SQL filters + semantic search string, composed from 5a and 5b in parallel.
    Results are cached in-process for 6 hours (identical queries skip both LLM calls).
    """
    import concurrent.futures

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
            print(f"[Agent5b] Failed: {e} — using original query")
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


# Backward-compat alias
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
    """Agent 6: Synthesize a cited answer from retrieved document chunks."""
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
