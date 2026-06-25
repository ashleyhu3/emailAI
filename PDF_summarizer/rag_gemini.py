"""
RAG pipeline: search on verbalized_summary, answer from raw_content.

Flow: Return top 3 most relevant chunks; for each, include metadata + summary of the chunk,
its parent, and its siblings, then repeat for the other two (~9 chunks in context).
"""
import calendar
import json
import os
import re
import time
import uuid as uuid_lib
from dataclasses import dataclass
from datetime import date as date_cls
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import google.api_core.exceptions
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from database import DatabaseManager, PDFChunk, PDFDocument
from embeddings import (
    EMBEDDING_MODEL, EMBEDDING_DIMS,
    embed_text, embed_texts_batch, EmbeddingCache, _get_embedding_cache,
)
from reranker import get_reranker
from user_memory import get_user_memory

GENERATION_MODEL = "models/gemini-3.5-flash"
HISTORY_WINDOW = 14

load_dotenv()


def _is_rate_limit(exc: Exception) -> bool:
    """True if an exception is a 429/quota error from either the google-genai SDK
    (ClientError code 429) or the older google-api-core (ResourceExhausted)."""
    if isinstance(exc, google.api_core.exceptions.ResourceExhausted):
        return True
    if isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429:
        return True
    return "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)


# ── Deterministic period extraction ────────────────────────────────────────────
# Detects an explicitly-named period in a question so coverage_period can be set
# reliably, regardless of the LLM's (inconsistent) judgement of whether a query is
# "scoped". This is what makes "Q1 2025" auto-filter every time, not just sometimes.
_QUARTER_RE = re.compile(r"\bQ\s*([1-4])\s*[-/,]?\s*((?:19|20)\d{2})\b", re.IGNORECASE)
_QUARTER_RE_YEAR_FIRST = re.compile(r"\b((?:19|20)\d{2})\s*[-/,]?\s*Q\s*([1-4])\b", re.IGNORECASE)
_HALF_RE = re.compile(r"\bH\s*([12])\s*[-/,]?\s*((?:19|20)\d{2})\b", re.IGNORECASE)
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_QUARTER_RANGES = {
    1: ((1, 1), (3, 31)),
    2: ((4, 1), (6, 30)),
    3: ((7, 1), (9, 30)),
    4: ((10, 1), (12, 31)),
}
_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_coverage_period(question: str) -> Optional[Tuple[date_cls, date_cls]]:
    """Extract an explicitly-named period (quarter, half-year, month, or year) from the
    question and return (from_date, to_date). Most specific match wins; returns None if
    no period is named."""
    m = _QUARTER_RE.search(question)
    if m:
        q, yr = int(m.group(1)), int(m.group(2))
        (sm, sd), (em, ed) = _QUARTER_RANGES[q]
        return date_cls(yr, sm, sd), date_cls(yr, em, ed)
    m = _QUARTER_RE_YEAR_FIRST.search(question)
    if m:
        yr, q = int(m.group(1)), int(m.group(2))
        (sm, sd), (em, ed) = _QUARTER_RANGES[q]
        return date_cls(yr, sm, sd), date_cls(yr, em, ed)
    m = _HALF_RE.search(question)
    if m:
        h, yr = int(m.group(1)), int(m.group(2))
        return (date_cls(yr, 1, 1), date_cls(yr, 6, 30)) if h == 1 \
            else (date_cls(yr, 7, 1), date_cls(yr, 12, 31))
    m = _MONTH_RE.search(question)
    if m:
        mon = _MONTH_NUM[m.group(1)[:3].lower()]
        yr = int(m.group(2))
        last = calendar.monthrange(yr, mon)[1]
        return date_cls(yr, mon, 1), date_cls(yr, mon, last)
    m = _YEAR_RE.search(question)
    if m:
        yr = int(m.group(1))
        return date_cls(yr, 1, 1), date_cls(yr, 12, 31)
    return None


# Nouns that, when enumerated, signal CONTENT (a RAG analysis) rather than a file inventory.
_CONTENT_LIST_NOUNS = frozenset({
    "trend", "trends", "risk", "risks", "takeaway", "takeaways", "theme", "themes",
    "point", "points", "idea", "ideas", "finding", "findings", "reason", "reasons",
    "factor", "factors", "driver", "drivers", "highlight", "highlights", "insight",
    "insights", "observation", "observations", "catalyst", "catalysts", "call", "calls",
    "recommendation", "recommendations", "conclusion", "conclusions", "thing", "things",
})
# Nouns that signal the user wants the FILES themselves (a document inventory).
_DOCUMENT_NOUNS = frozenset({
    "document", "documents", "file", "files", "report", "reports", "doc", "docs",
    "note", "notes", "pdf", "pdfs",
})


def _is_content_enumeration(question: str) -> bool:
    """True if the question asks to enumerate CONTENT (trends, risks, takeaways, ...) rather
    than documents — i.e. a RAG analysis that happens to be list-shaped. Used to correct a
    classifier that over-eagerly routes any 'list ...' query to the document inventory."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words & _DOCUMENT_NOUNS:
        return False  # explicitly about files/reports → let the inventory path handle it
    return bool(words & _CONTENT_LIST_NOUNS)


def _apply_period_safety_net(result: dict, question: str) -> dict:
    """Fill coverage_period_from/to deterministically when the question names a period and
    the LLM didn't already set it, so auto-filtering is consistent run-to-run."""
    hf = result.get("hard_filters") or {}
    result["hard_filters"] = hf
    if not hf.get("coverage_period_from") and not hf.get("coverage_period_to"):
        period = _extract_coverage_period(question)
        if period:
            hf["coverage_period_from"] = period[0].isoformat()
            hf["coverage_period_to"] = period[1].isoformat()
    return result


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


@dataclass
class RetrievalFilters:
    document_ids: Optional[Sequence[int]] = None
    filenames: Optional[Sequence[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    sender_names: Optional[Sequence[str]] = None
    sender_companies: Optional[Sequence[str]] = None
    written_date_from: Optional[str] = None
    written_date_to: Optional[str] = None
    # Extended metadata filters
    tickers: Optional[Sequence[str]] = None
    report_type: Optional[str] = None
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    coverage_period_from: Optional[str] = None
    coverage_period_to: Optional[str] = None


def _record_memory(memory, analysis: dict) -> None:
    """Update user preference memory from a completed query's analysis."""
    try:
        hf = analysis.get("hard_filters") or {}
        tickers = hf.get("tickers") or []
        brokers = hf.get("sender_companies") or []
        sector = hf.get("sector") or None
        if tickers or brokers or sector:
            memory.record(tickers=tickers, brokers=brokers, sector=sector)
    except Exception:
        pass  # memory update failures must never break the answer pipeline


class GeminiRAGPipeline:
    """RAG over verbalized pages (text + chart descriptions)."""

    def __init__(self, database_url: str = None, db: DatabaseManager = None):
        self.client = _get_client()
        if db is not None:
            self.db = db
        elif database_url is not None:
            self.db = DatabaseManager(database_url)
        else:
            raise ValueError("Either database_url or db must be provided")

    def backfill_embeddings(
        self,
        batch_size: int = 64,
        max_batches: Optional[int] = None,
        reset: bool = False,
        sleep: float = 0.0,
    ) -> int:
        """Generate embeddings for chunks that don't have them yet.

        Embeds an entire batch in a single API call instead of one call per chunk,
        cutting embedding time from O(n) round-trips to O(n/batch_size) round-trips.

        reset=True first NULLs every existing embedding, so the whole corpus is re-embedded
        — use this when switching EMBEDDING_MODEL (vectors from different models aren't
        comparable). Each batch retries on rate limits, and a dimension guard aborts before
        writing if the model's output doesn't match the Vector(EMBEDDING_DIMS) schema.
        """
        if reset:
            # Probe the model's output dimension BEFORE wiping anything, so a model that
            # can't emit EMBEDDING_DIMS fails here rather than leaving an empty index.
            probe = embed_text("dimension probe")
            if probe and len(probe) != EMBEDDING_DIMS:
                raise ValueError(
                    f"{EMBEDDING_MODEL} produced {len(probe)}-dim vectors but the schema is "
                    f"Vector({EMBEDDING_DIMS}). Refusing to clear existing embeddings. Pick a "
                    f"model that supports {EMBEDDING_DIMS}-dim output, or migrate the "
                    f"pdf_chunks.embedding column + index to {len(probe)} dims first."
                )
            cleared = self.db.clear_all_embeddings()
            print(f"[RESET] cleared {cleared} existing embedding(s) for a full re-embed")

        total = 0
        batches = 0
        cache_hits = 0
        emb_cache = _get_embedding_cache()

        while True:
            if max_batches is not None and batches >= max_batches:
                break

            chunks = self.db.get_chunks_without_embedding(limit=batch_size)
            if not chunks:
                break

            texts = [
                (
                    (chunk.raw_content or "") +
                    "\n\n" +
                    (chunk.verbalized_summary or "")
                ).strip()
                for chunk in chunks
            ]

            # Check the boilerplate cache before hitting the API.
            # Identical text (same legal disclaimer, same page header) that was embedded
            # in a previous run is served from disk — zero API call needed.
            cached_vecs: List[Optional[List[float]]] = [emb_cache.get(t) for t in texts]
            uncached_pairs = [(i, t) for i, (t, c) in enumerate(zip(texts, cached_vecs)) if c is None]
            batch_cache_hits = len(chunks) - len(uncached_pairs)
            cache_hits += batch_cache_hits
            if batch_cache_hits:
                print(f"[BACKFILL] cache: {batch_cache_hits} hit(s), {len(uncached_pairs)} API call(s) needed")

            if uncached_pairs:
                uncached_indices, uncached_texts = zip(*uncached_pairs)
                # Retry on rate limits so a full re-embed doesn't abort mid-run.
                for attempt in range(6):
                    try:
                        new_embeddings = embed_texts_batch(list(uncached_texts))
                        break
                    except Exception as exc:
                        if not _is_rate_limit(exc) or attempt == 5:
                            raise
                        wait = min(60, 10 * (2 ** attempt))  # 10,20,40,60,60s
                        print(f"[WARNING] embedding rate limited, retrying in {wait}s "
                              f"(attempt {attempt + 1}/6)")
                        time.sleep(wait)
                # Slot new embeddings back into the full-batch list and save to cache.
                for orig_idx, emb in zip(uncached_indices, new_embeddings):
                    cached_vecs[orig_idx] = emb
                    if emb:
                        emb_cache.put(texts[orig_idx], emb)
                emb_cache.flush()

            for chunk, emb in zip(chunks, cached_vecs):
                if not emb:
                    continue
                if len(emb) != EMBEDDING_DIMS:
                    raise ValueError(
                        f"{EMBEDDING_MODEL} returned {len(emb)}-dim vectors, but the schema is "
                        f"Vector({EMBEDDING_DIMS}). This model may not support "
                        f"{EMBEDDING_DIMS}-dim output. Aborting before writing — either pick a "
                        f"model that supports {EMBEDDING_DIMS} dims, or migrate the "
                        f"pdf_chunks.embedding column + index to the new size."
                    )
                self.db.upsert_chunk_embedding(chunk.id, emb)
                total += 1

            batches += 1
            print(f"[BACKFILL] embedded {total} chunk(s) so far (cache hits: {cache_hits})...")
            if sleep > 0:
                time.sleep(sleep)  # proactively throttle to stay under rate/token limits

        return total

    # Minimum cosine similarity for a chunk to be considered relevant.
    # Cosine similarity ranges 0–1; chunks below this are discarded rather than
    # passed as context, preventing the model from hallucinating from unrelated content.
    SIMILARITY_THRESHOLD = 0.40

    # For conversational follow-ups (e.g. "expand on that"), skip the similarity
    # threshold entirely. The vector search still runs and returns the top-k closest
    # chunks — we just don't discard any of them. The model has the conversation
    # history to understand it should elaborate, not introduce new topics.
    FOLLOWUP_SIMILARITY_THRESHOLD = 0.0

    # When 1–2 hard metadata filters are active (company, ticker, date, etc.) the
    # document pool is already meaningfully narrowed, so we can relax the threshold.
    SIMILARITY_THRESHOLD_FEW_FILTERS = 0.30

    # When 3+ hard metadata filters are active the pool is tightly scoped; relax further.
    SIMILARITY_THRESHOLD_MANY_FILTERS = 0.20

    # Retrieval is driven by the similarity threshold, not a small top-k cap: every chunk
    # above the threshold is used, so a broad question can pull in as much relevant context
    # as exists. This ceiling is only a runaway guard against a very broad, low-threshold
    # query (e.g. a company filter at the 0.20 tier) flooding the prompt with chunks.
    RETRIEVAL_CEILING = 40

    # Enumeration queries ("10 trends across SinoPac Q1") want BREADTH across documents,
    # not depth in the one or two that best match a vague "trends" query. So we relax the
    # threshold (the metadata filter already guarantees relevance) and cap how many chunks
    # any single document can contribute, so retrieval spreads across many docs and the
    # model has a real, citable page for each distinct item instead of fabricating one.
    ENUMERATION_SIMILARITY_THRESHOLD = 0.05
    ENUMERATION_PER_DOC_CAP = 3
    # Candidate pool fetched from the DB before per-document diversification. Must be well
    # above RETRIEVAL_CEILING so docs that rank below the top-40 by raw similarity still
    # enter the pool and can be picked by the round-robin (otherwise diversity has nothing
    # extra to spread to).
    ENUMERATION_CANDIDATE_POOL = 200

    # Cross-encoder reranking: how many candidates to keep after CE rescoring.
    # Applied after hybrid RRF, before diversification. When the reranker is
    # unavailable (sentence-transformers not installed), this is a no-op.
    RERANKER_TOP_K = 25

    # Parent chunk expansion: for the top-N retrieved page chunks, fetch their
    # parent section chunk (if any) and add it to the retrieval pool. Gives the
    # model broader section-level context alongside the precise page match.
    # Set to 0 to disable.
    PARENT_EXPANSION_TOP_N = 10

    # Citation verification: when True, also strip citations to pages that were only
    # neighbouring context (parent/sibling), not independently retrieved as relevant.
    # Default False strips only fabricated pages (never in context at all) — the safe check.
    STRICT_CITATIONS = False

    # Matches an in-text citation like "(report.pdf, p.3)", "(report.pdf, pages 9, 11)", or
    # document-level "(report.pdf)" with no page. The filename may contain one level of nested
    # parens (e.g. "91APP (6741).pdf"); requiring ".pdf" keeps prose parens like "(see below)"
    # from matching. The page spec is optional so document-level citations parse too.
    _CITATION_BODY = (
        r"\((?:[^()]|\([^()]*\))*?\.pdf(?:[\s,]+(?:pp?\.?|pages?)?\s*\d[\d,\s\-–]*)?\)"
    )
    # Capturing version: group 1 = filename, group 2 = page spec (optional; None if absent).
    _CITATION_RE = re.compile(
        r"\(((?:[^()]|\([^()]*\))*?\.pdf)"
        r"(?:[\s,]+((?:pp?\.?|pages?)?\s*\d[\d,\s\-–]*))?\)",
        re.IGNORECASE,
    )
    # A run of adjacent citations separated only by commas/semicolons (a "Sources:" list).
    # Used to merge repeated same-document citations within one list.
    _CITATION_RUN_RE = re.compile(
        rf"{_CITATION_BODY}(?:\s*[;,]\s*{_CITATION_BODY})*", re.IGNORECASE
    )

    def _analyze_query(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        user_hint: str = "",
    ) -> dict:
        """Single Gemini Flash call that simultaneously:
          1. Extracts hard filters (company, date range, pages) from the question + history.
          2. Rewrites the question into a self-contained standalone_query for vector search
             (strips filter noise, resolves history references like "their" / "that").
          3. Classifies whether this is a conversational follow-up (expand, clarify, continue)
             vs a genuinely new information search.
          4. Classifies the query_type: "list_documents" (user wants an inventory of files)
             vs "rag" (user wants an answer drawn from document content).

        Returns:
            {
                "hard_filters": {
                    "sender_companies": [...] | null,
                    "written_date_from": "YYYY-MM-DD" | null,
                    "written_date_to":   "YYYY-MM-DD" | null,
                    "page_min": int | null,
                    "page_max": int | null,
                },
                "standalone_query": str,  # used for embedding / vector search
                "is_followup": bool,      # true → relax similarity threshold
                "query_type": str,        # "rag" | "list_documents"
            }

        Falls back to {"hard_filters": {}, "standalone_query": question, "is_followup": false, "query_type": "rag"}
        on any error so the pipeline degrades gracefully.
        """
        # Fetch known companies and filenames to ground the model's extraction.
        session = self.db.get_session()
        try:
            known_companies = [
                r[0] for r in session.query(PDFDocument.sender_company).distinct().all()
                if r[0]
            ]
            known_filenames = [
                r[0] for r in session.query(PDFDocument.filename).distinct().all()
                if r[0]
            ]
        finally:
            session.close()

        # Format history as readable lines (oldest first).
        history_text = "(none)"
        if history:
            lines = []
            for msg in history[-HISTORY_WINDOW:]:
                label = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{label}: {msg['content']}")
            history_text = "\n".join(lines)

        today = date_cls.today().isoformat()
        user_hint_block = (user_hint.strip() + "\n\n") if user_hint.strip() else ""

        prompt = f"""You are a query analysis assistant for a financial document RAG system.
Today's date: {today}
Known companies in the database: {json.dumps(known_companies)}
Known filenames: {json.dumps(known_filenames)}

Given the conversation history and current question, return a JSON object with exactly five fields:

1. "hard_filters": deterministic constraints to narrow the document search.
   Extract only what is explicitly stated or clearly implied:
   - "sender_companies": list of company/firm names matching a known company above, or null.
     Use the SHORTEST distinctive root shared across variants of the same firm — e.g. prefer
     "SinoPac" over "SinoPac Securities" so every SinoPac-authored document is matched, not
     just one naming variant. Matching is fuzzy/substring, so the root form is safest.
   - "written_date_from": ISO date (YYYY-MM-DD) for when the document was PUBLISHED, or null
   - "written_date_to":   ISO date (YYYY-MM-DD) for when the document was PUBLISHED, or null
   - "page_min": integer page lower bound if explicitly mentioned, or null
   - "page_max": integer page upper bound if explicitly mentioned, or null
   - "tickers": list of ticker symbols mentioned (e.g. ["BTC", "AAPL"]), or null.
     Use standard exchange symbols, not full names.
   - "report_type": type of analysis if specified — one of: equity_research,
     technical_analysis, macro, crypto, sector_note, strategy, other — or null
   - "sector": GICS sector if mentioned (e.g. "Technology", "Energy"), or null
   - "asset_class": asset class if specified — one of: equity, crypto, fixed_income,
     commodity, fx, mixed — or null
   - "coverage_period_from": ISO date for the START of the period being ANALYSED
     (not when published). Whenever the user explicitly names a period — a quarter,
     half-year, month, or year (e.g. "Q1 2025", "H2 2023", "March 2024", "in 2025",
     "what did SinoPac say about Q1 2025") — set this to the START of that period.
     e.g. "SinoPac Q3 2024 earnings" → 2024-07-01. Or null if no period is named.
   - "coverage_period_to": ISO date for the END of that named period
     (e.g. Q3 2024 → 2024-09-30; "in 2025" → 2025-12-31). Or null.

   Quarter mapping: Q1=Jan 1–Mar 31, Q2=Apr 1–Jun 30, Q3=Jul 1–Sep 30, Q4=Oct 1–Dec 31.
   If a quarter is mentioned without a year, infer from context or use today's year.
   Distinguish written_date (when published) from coverage_period (what period analysed):
   a Q3 earnings report published in November uses written_date=Nov, coverage_period=Jul–Sep.

2. "standalone_query": rewrite the question as a self-contained semantic search string.
   - Resolve all pronouns / references using history (e.g. "their" → company name)
   - Remove information already captured in hard_filters (ticker, date, company, report type)
   - Keep only the semantic/reasoning intent (what the user actually wants to know)
   - If the question is already self-contained, keep it as-is
   - NEVER invent a topic. If removing the filters leaves no real topic (e.g. "what has
     SinoPac said in Q1 2025?"), set standalone_query to just the company/period that
     remains (or the original question) — do NOT fabricate themes like "earnings, forecasts".

3. "is_followup": true if the question asks the assistant to elaborate, clarify, continue,
   or reason further about something already discussed (e.g. "can you expand on that",
   "tell me more", "what do you mean by X", "go deeper on the second point", "why is that").
   Set to false if the question requests genuinely new information from the documents,
   even if it references prior context (e.g. "what about their bond holdings?").

4. "query_type": classify the user's intent as one of two strings. The deciding factor is
   WHAT the user wants enumerated — the documents themselves, or content from inside them.
   The word "list" alone does NOT mean "list_documents".
   - "list_documents": the user wants an inventory of matching FILES/REPORTS/DOCUMENTS —
     the documents are the answer. The thing being listed is documents. Triggered by:
     "show me all files about X", "what documents do you have from Y",
     "I want all files relevant to Z", "list all reports on W",
     "how many files mention ticker T", "do you have any reports on X",
     "what do you have about X", "give me a list of documents that cover X".
   - "rag": the user wants CONTENT drawn from inside the documents — analysis, numbers,
     quotes, summaries, comparisons, OR an enumeration of ideas/findings/themes. This
     INCLUDES list-shaped requests whose items are content rather than documents, e.g.
     "list 10 trends SinoPac talked about in Q1 2025", "give me 5 key risks",
     "what are the main takeaways", "list the top themes this quarter". Here the items
     are trends/risks/takeaways/themes (content), not files — so it is "rag".
   Rule of thumb: if the items would be DOCUMENT NAMES → "list_documents"; if the items
   would be SENTENCES/FACTS/IDEAS → "rag". When in doubt, use "rag".

5. "is_underspecified": true ONLY when the question is scoped by a company/ticker/sector/date
   (i.e. hard_filters above are non-empty) but names NO specific topic, stock, metric, event,
   or answerable question — a broad "what did X say" with nothing to focus retrieval on.
   - true:  "what has SinoPac said in Q1 2025?", "what's the latest from SinoPac?",
            "anything from Apple recently?"  (filter + no topic)
   - false: "what did SinoPac say about China Mobile?" (names a stock),
            "give me an overview of SinoPac Q1 2025" / "summarize SinoPac's quarter"
            (a summary/overview IS the intent — answer it, don't ask to clarify),
            "what were the key risks?" (names a topic),
            any question with no hard_filters at all (nothing to anchor a clarification).

6. "keywords": a compact list (≤8 items) of the most specific searchable terms from the question.
   Include: exact ticker symbols (e.g. "TSMC", "3405.JP"), company/broker names, financial action
   words (upgrade, downgrade, overweight, underweight, initiate), metric names (EPS, PE, margin,
   revenue, target price). Exclude: generic words, pronouns, stop words, dates (already in filters).
   Used for keyword search — precision matters more than recall. Return [] if no specific terms.
   Example: ["TSMC", "upgrade", "overweight", "semiconductor"]

7. "response_track": classify the most appropriate answer strategy as one of four strings.
   - "point_fact": The user wants a specific number, rating, or status — a lookup answer, not an
     essay. Triggered by: "what is X's target price?", "what rating does Goldman have on TSMC?",
     "did JPMorgan upgrade Apple?", "what is the current consensus rating?", "is TSMC overweight?"
     The answer can be expressed in 1–3 sentences from structured metadata (no essay needed).
   - "comparison": The user explicitly wants a side-by-side comparison across ≥2 brokers, ≥2 time
     periods, or ≥2 companies. Triggered by: "compare Goldman vs JPMorgan on TSMC", "how has the
     consensus changed over Q1 vs Q2?", "what do different brokers say about Apple?", "contrast
     SinoPac and Nomura's views".
   - "sector_sweep": The user wants broad thematic analysis across many documents/companies —
     trends, themes, or signals across a whole sector/market. Triggered by: "what are the main
     themes in semiconductor research this quarter?", "summarize 10 key risks across Asia equities",
     "what macro trends appear across all the recent reports?".
   - "deep_dive": Default for analytical questions about one company or topic that require reading
     and synthesizing document content. Use this when none of the above fit: earnings analysis,
     valuation discussion, competitive positioning, anything requiring prose synthesis.
   When uncertain, prefer "deep_dive".

Conversation history (oldest first):
{history_text}

{user_hint_block}Current question: {question}

Return ONLY a valid JSON object. No markdown, no explanation."""

        try:
            response = self.client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            text = (response.text or "").strip()
            # Strip markdown code fences if present despite response_mime_type
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            result = json.loads(text)
            if "hard_filters" not in result or "standalone_query" not in result:
                raise ValueError("Missing required keys in analysis response")
            # Ensure is_followup is always a bool
            result.setdefault("is_followup", False)
            result["is_followup"] = bool(result["is_followup"])
            # Ensure query_type is valid
            result.setdefault("query_type", "rag")
            if result["query_type"] not in ("list_documents", "rag"):
                result["query_type"] = "rag"
            # Deterministic guard: enumerating content (trends/risks/takeaways...) is a RAG
            # analysis, not a file inventory — even when the user says "list".
            if result["query_type"] == "list_documents" and _is_content_enumeration(question):
                print("[DEBUG] query_type override: list_documents → rag (content enumeration)")
                result["query_type"] = "rag"
            # Ensure is_underspecified is always a bool
            result.setdefault("is_underspecified", False)
            result["is_underspecified"] = bool(result["is_underspecified"])
            # Ensure keywords is always a list
            result.setdefault("keywords", [])
            if not isinstance(result.get("keywords"), list):
                result["keywords"] = []
            # Ensure response_track is a valid value
            _valid_tracks = ("point_fact", "comparison", "sector_sweep", "deep_dive")
            result.setdefault("response_track", "deep_dive")
            if result["response_track"] not in _valid_tracks:
                result["response_track"] = "deep_dive"
            return _apply_period_safety_net(result, question)
        except Exception as exc:
            print(f"[WARNING] _analyze_query failed ({exc}); falling back to original question")
            return _apply_period_safety_net(
                {"hard_filters": {}, "standalone_query": question,
                 "is_followup": False, "query_type": "rag", "keywords": [],
                 "response_track": "deep_dive"},
                question,
            )

    @staticmethod
    def _parse_date(value) -> Optional[date_cls]:
        """Convert an ISO date string from the LLM ('YYYY-MM-DD') to a datetime.date.
        Returns None if the value is missing or unparseable.
        """
        if value is None:
            return None
        if isinstance(value, date_cls):
            return value
        try:
            return date_cls.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    def _merge_filters(
        self, explicit: RetrievalFilters, analysis: dict
    ) -> RetrievalFilters:
        """Merge LLM-inferred hard filters with explicit caller-supplied filters.
        Explicit (sidebar) values always win — inferred values only fill in None fields.
        Date strings from the LLM are converted to datetime.date so SQLAlchemy
        can compare them against the DATE column without a type error.
        """
        hf = analysis.get("hard_filters") or {}
        return RetrievalFilters(
            document_ids=explicit.document_ids,
            filenames=explicit.filenames,
            page_min=explicit.page_min if explicit.page_min is not None else hf.get("page_min"),
            page_max=explicit.page_max if explicit.page_max is not None else hf.get("page_max"),
            sender_names=explicit.sender_names,
            sender_companies=explicit.sender_companies or hf.get("sender_companies") or None,
            written_date_from=explicit.written_date_from or self._parse_date(hf.get("written_date_from")),
            written_date_to=explicit.written_date_to or self._parse_date(hf.get("written_date_to")),
            tickers=explicit.tickers or hf.get("tickers") or None,
            report_type=explicit.report_type or hf.get("report_type") or None,
            sector=explicit.sector or hf.get("sector") or None,
            asset_class=explicit.asset_class or hf.get("asset_class") or None,
            coverage_period_from=explicit.coverage_period_from or self._parse_date(hf.get("coverage_period_from")),
            coverage_period_to=explicit.coverage_period_to or self._parse_date(hf.get("coverage_period_to")),
        )

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
        similarity_threshold: Optional[float] = None,
        per_document_cap: Optional[int] = None,
        bm25_keywords: Optional[List[str]] = None,
    ) -> List[PDFChunk]:
        """Hybrid vector + BM25 retrieval over pdf_chunks.

        Expects `query` to already be the standalone/cleaned query from _analyze_query
        and `filters` to already be the merged result of _merge_filters.
        `bm25_keywords` (from _analyze_query "keywords" field) are joined into a compact
        keyword string for the BM25 leg; falls back to `query` if not provided.
        Returns chunks ordered by RRF score (up to RETRIEVAL_CEILING).

        Args:
            top_k: Retained for API compatibility but no longer caps retrieval; the
                threshold and RETRIEVAL_CEILING govern how many chunks are returned.
            similarity_threshold: Override the default SIMILARITY_THRESHOLD.
                Pass FOLLOWUP_SIMILARITY_THRESHOLD for conversational follow-ups.
            bm25_keywords: Precision terms for BM25 (tickers, action words, company names).
        """
        if filters is None:
            filters = RetrievalFilters()
        if similarity_threshold is None:
            similarity_threshold = self.SIMILARITY_THRESHOLD

        query_emb = embed_text(query)
        final_limit = max(top_k, self.RETRIEVAL_CEILING)
        # For breadth (enumeration), fetch a much larger candidate pool so diversify-by-
        # document can reach docs that rank well below the top-40 by raw similarity.
        candidate_limit = self.ENUMERATION_CANDIDATE_POOL if per_document_cap else final_limit
        bm25_query = " ".join(bm25_keywords) if bm25_keywords else query
        raw_chunks = self.db.hybrid_search_chunks(
            query_embedding=query_emb,
            query_text=bm25_query,
            limit=candidate_limit,
            similarity_threshold=similarity_threshold,
            candidate_k=candidate_limit,
            document_ids=filters.document_ids,
            filenames=filters.filenames,
            page_min=filters.page_min,
            page_max=filters.page_max,
            sender_names=filters.sender_names,
            sender_companies=filters.sender_companies,
            written_date_from=filters.written_date_from,
            written_date_to=filters.written_date_to,
            tickers=filters.tickers,
            report_type=filters.report_type,
            sector=filters.sector,
            asset_class=filters.asset_class,
            coverage_period_from=filters.coverage_period_from,
            coverage_period_to=filters.coverage_period_to,
        )

        if not raw_chunks:
            return []

        # ── Cascade cross-encoder reranking ──────────────────────────────────
        # The bi-encoder (vector) and BM25 scores are fast but approximate; the
        # cross-encoder jointly encodes (query, passage) and re-scores the top
        # candidates with much higher precision. Only runs when sentence-transformers
        # is installed; degrades to the original RRF order otherwise.
        reranker = get_reranker()
        if reranker.is_available() and not per_document_cap:
            # Skip reranking for enumeration queries (already broadened for diversity)
            rerank_k = max(self.RERANKER_TOP_K, final_limit)
            raw_chunks = reranker.rerank(query, raw_chunks, top_k=rerank_k)

        # ── Parent chunk expansion (small-to-big) ────────────────────────────
        # For the top-N page-level results, fetch the parent section chunk and
        # add it to the pool. This gives the model section-level context for the
        # most relevant passages without requiring a second search call.
        if self.PARENT_EXPANSION_TOP_N > 0 and not per_document_cap:
            raw_chunks = self._expand_to_section_parents(raw_chunks)

        # per_document_cap (enumeration/overview): spread retrieval across many documents so
        # the model has a citable page for each distinct item. Otherwise reorder for
        # hierarchy/section context (depth within the most relevant docs).
        if per_document_cap:
            diversified = self._diversify_by_document(
                raw_chunks, top_k=final_limit, per_doc_cap=per_document_cap
            )
            print(f"[DEBUG] retrieved {len(diversified)} chunks across "
                  f"{len({c.document_id for c in diversified})} documents "
                  f"(from {len(raw_chunks)} candidates, per-doc cap {per_document_cap}, "
                  f"threshold {similarity_threshold})")
        else:
            diversified = self._diversify_chunks(raw_chunks, top_k=final_limit)
            print(f"[DEBUG] retrieved {len(diversified)} hybrid chunks (threshold {similarity_threshold})")
        return diversified

    def _expand_to_section_parents(
        self, chunks: List[PDFChunk]
    ) -> List[PDFChunk]:
        """Small-to-big parent expansion: add the section chunk for the top-N page hits.

        Page chunks are retrieved for precision (short, dense text scores better in
        embedding space). Their parent section chunk has broader context (often 2–5 pages
        of narrative) that helps the model answer questions that span a section. By
        appending the section to the pool, _build_context will render it alongside the
        precise page match without duplicating text (the `seen` set deduplicates).

        Only the top-PARENT_EXPANSION_TOP_N page chunks are expanded to avoid inflating
        the pool with section chunks from every retrieved page.
        """
        seen_ids: set = {c.id for c in chunks}
        parents: List[PDFChunk] = []

        for chunk in chunks[: self.PARENT_EXPANSION_TOP_N]:
            meta = chunk.metadata_ or {}
            level = meta.get("level", "")
            if level not in ("page", "image"):
                continue  # already a section or document chunk
            parent_id_str = meta.get("parent_chunk_id")
            if not parent_id_str:
                continue
            try:
                parent_id = uuid_lib.UUID(parent_id_str)
            except (ValueError, TypeError):
                continue
            if parent_id in seen_ids:
                continue
            parent = self.db.get_chunk_by_id(parent_id)
            if parent is not None:
                parents.append(parent)
                seen_ids.add(parent_id)

        if parents:
            print(f"[parent-expansion] appended {len(parents)} section parent(s) to pool")

        return chunks + parents

    def _get_chunk_family(
        self, chunk: PDFChunk
    ) -> Tuple[Optional[PDFChunk], Optional[PDFChunk], Optional[PDFChunk]]:
        """Return (parent, prev_sibling, next_sibling) for a chunk using metadata IDs."""
        meta = chunk.metadata_ or {}
        parent, prev_sib, next_sib = None, None, None
        try:
            pid = meta.get("parent_chunk_id")
            if pid:
                parent = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        try:
            pid = meta.get("prev_sibling_chunk_id")
            if pid:
                prev_sib = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        try:
            pid = meta.get("next_sibling_chunk_id")
            if pid:
                next_sib = self.db.get_chunk_by_id(uuid_lib.UUID(pid))
        except (ValueError, TypeError):
            pass
        return parent, prev_sib, next_sib

    def _build_context(self, top_chunks: List[PDFChunk]) -> Tuple[str, set]:
        """
        Build context from the retrieved chunks: for each chunk include its metadata +
        summary + content, then the same for its parent and sibling chunks.

        Chunk content is deduplicated across the whole context. A chunk's full content is
        emitted only once; if it reappears (because two adjacent retrieved chunks are each
        other's siblings, or many chunks share one section-level parent), later occurrences
        show a short breadcrumb reference instead of repeating the text. This keeps the
        prompt compact when many chunks are retrieved.

        Returns (context_string, context_refs) where context_refs is the set of
        (filename_lower, page_number) pairs for every chunk actually shown to the model —
        used afterwards to verify the citations in the answer.
        """
        # Seed with all primary retrieved chunk IDs so family expansion never re-emits a
        # chunk that is itself a primary (e.g. two adjacent retrieved pages).
        seen: set = {chunk.id for chunk in top_chunks}
        context_refs: set = set()
        parts: List[str] = []
        for n, chunk in enumerate(top_chunks, 1):
            parent, prev_sib, next_sib = self._get_chunk_family(chunk)
            block = self._format_chunk_block(
                chunk, parent, prev_sib, next_sib, seen, context_refs, label=f"Retrieved chunk {n}"
            )
            parts.append(block)
        return "\n\n".join(parts), context_refs

    def _format_chunk_block(
        self,
        chunk: PDFChunk,
        parent: Optional[PDFChunk],
        prev_sibling: Optional[PDFChunk],
        next_sibling: Optional[PDFChunk],
        seen: set,
        context_refs: set,
        label: str = "Chunk",
    ) -> str:
        """Format one retrieved chunk plus its parent and siblings (metadata + summary + content).

        `seen` is the set of chunk IDs already emitted in full anywhere in the context; it is
        mutated here. Family members already in `seen` are rendered as a breadcrumb reference
        rather than having their content repeated. `context_refs` collects the
        (filename_lower, page_number) of every chunk shown, for later citation verification.
        """
        lines: List[str] = [f"=== {label} ==="]

        def append_chunk(c: PDFChunk, role: str) -> None:
            meta = c.metadata_ or {}
            summary = (c.verbalized_summary or "").strip()
            content = (c.raw_content or "").strip()
            filename = c.document.filename if c.document else f"document_id={c.document_id}"
            page = f" page {c.page_number}" if c.page_number is not None else ""
            if c.document and c.page_number is not None:
                context_refs.add((c.document.filename.lower(), c.page_number))
            lines.append(f"  {role} source: {filename}{page}")
            lines.append(f"  {role} metadata: {meta}")
            lines.append(f"  {role} summary: {summary[:1500]}{'...' if len(summary) > 1500 else ''}")
            lines.append(f"  {role} content: {content[:4000]}{'...' if len(content) > 4000 else ''}")

        def append_family(c: Optional[PDFChunk], role: str) -> None:
            if c is None:
                lines.append(f"  {role}: (none)")
            elif c.id in seen:
                # Already shown in full elsewhere — reference it instead of repeating.
                filename = c.document.filename if c.document else f"document_id={c.document_id}"
                page = f" page {c.page_number}" if c.page_number is not None else ""
                lines.append(f"  {role}: (already shown above — {filename}{page})")
            else:
                seen.add(c.id)
                append_chunk(c, role)

        # The primary chunk is always emitted in full (its ID is already in `seen`).
        append_chunk(chunk, "Chunk")
        append_family(parent, "Parent")
        append_family(prev_sibling, "Previous sibling")
        append_family(next_sibling, "Next sibling")

        return "\n".join(lines)

    @staticmethod
    def _expand_pages(spec: Optional[str]) -> List[int]:
        """Extract page numbers from a citation page spec, expanding ranges.
        'pages 9, 11, 12' → [9, 11, 12];  'pp. 3-5' → [3, 4, 5];  None/'' → []."""
        if not spec:
            return []
        pages: List[int] = []
        for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", spec):
            lo, hi = int(a), int(b)
            if 0 < hi - lo < 100:  # guard against absurd ranges
                pages.extend(range(lo, hi + 1))
        spec_wo_ranges = re.sub(r"\d+\s*[-–]\s*\d+", " ", spec)
        pages.extend(int(x) for x in re.findall(r"\d+", spec_wo_ranges))
        return sorted(set(pages))

    def _verify_citations(
        self, answer: str, chunks: List[PDFChunk], context_refs: set
    ) -> str:
        """Document-level citation check (no LLM call). The model reliably gets the *document*
        right (it sees the whole-doc summary) but not always the *page* (it only holds 1–3
        real pages), so we verify at the document level:
          - document was retrieved + the exact cited page was in context → keep '(doc, p.N)'
          - document was retrieved but the page wasn't                  → keep '(doc)'  (page
            unverifiable, but the document is right)
          - document was never retrieved                                → strip (real fabrication)
        This stops 'right document, wrong page' from being discarded, which is what was
        leaving list items unsourced.
        """
        retrieved_files = {
            c.document.filename.lower() for c in chunks if c.document
        }
        in_context = set(context_refs) | {
            (c.document.filename.lower(), c.page_number)
            for c in chunks
            if c.document and c.page_number is not None
        }
        stats = {"page": 0, "doc_level": 0, "fabricated": 0}

        def repl(m: "re.Match") -> str:
            filename, spec = m.group(1), m.group(2)
            fn_lower = filename.lower()
            if fn_lower not in retrieved_files:
                stats["fabricated"] += 1
                return ""  # wrong document entirely → genuine fabrication, strip
            kept = [p for p in self._expand_pages(spec) if (fn_lower, p) in in_context]
            if kept:
                stats["page"] += 1
                if len(kept) == 1:
                    return f"({filename}, p.{kept[0]})"
                return f"({filename}, pp. {', '.join(str(p) for p in kept)})"
            stats["doc_level"] += 1
            return f"({filename})"  # right document, unverifiable page → cite document only

        cleaned = self._CITATION_RE.sub(repl, answer)

        # Tidy separators left by any stripped citations.
        cleaned = re.sub(r",\s*,", ", ", cleaned)
        cleaned = re.sub(r"(Sources:)\s*,\s*", r"\1 ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Sources:\s*(?=\n|$)", "", cleaned, flags=re.IGNORECASE)

        print(
            f"[CITATION CHECK] page-level={stats['page']} doc-level={stats['doc_level']} "
            f"fabricated(stripped)={stats['fabricated']}"
        )
        return cleaned

    def _consolidate_citations(self, answer: str) -> str:
        """Merge adjacent citations to the same document into one (no LLM call):
        '(doc.pdf, p.1), (doc.pdf, p.2), (doc.pdf, p.3)' → '(doc.pdf, pp. 1, 2, 3)'.
        Only citations within one comma-separated run (a Sources: list) are merged, so
        citations supporting different paragraphs stay separate.
        """
        def merge_run(m: "re.Match") -> str:
            run = m.group(0)
            cites = list(self._CITATION_RE.finditer(run))
            if len(cites) <= 1:
                return run  # single citation — nothing to merge
            order: List[str] = []          # filename keys in first-seen order
            pages_by: dict = {}            # key -> list of page numbers
            display: dict = {}             # key -> original filename for output
            for cm in cites:
                fname, spec = cm.group(1), cm.group(2)
                key = fname.lower()
                if key not in pages_by:
                    pages_by[key], display[key] = [], fname
                    order.append(key)
                pages_by[key].extend(self._expand_pages(spec))
            parts: List[str] = []
            for key in order:
                pages = sorted(set(pages_by[key]))
                fname = display[key]
                if not pages:
                    parts.append(f"({fname})")  # document-level citation, no page
                elif len(pages) == 1:
                    parts.append(f"({fname}, p.{pages[0]})")
                else:
                    parts.append(f"({fname}, pp. {', '.join(str(p) for p in pages)})")
            return ", ".join(parts)

        return self._CITATION_RUN_RE.sub(merge_run, answer)

    def _enforce_sourced_items(self, answer: str) -> str:
        """Guarantee every numbered list item carries a citation (no LLM call). Run AFTER
        verification/consolidation: any item whose citation was stripped (the model cited a
        document that wasn't retrieved) is dropped, and the survivors are renumbered. So a
        list never shows an unsourced item.

        Returns a fallback message if every item was dropped (nothing was attributable)."""
        item_start = re.compile(r"^\s*\d+[).]\s")
        lines = answer.split("\n")

        preamble: List[str] = []
        items: List[List[str]] = []
        cur: Optional[List[str]] = None
        for line in lines:
            if item_start.match(line):
                if cur is not None:
                    items.append(cur)
                cur = [line]
            elif cur is not None:
                cur.append(line)
            else:
                preamble.append(line)
        if cur is not None:
            items.append(cur)

        if not items:
            return answer  # not a numbered list — nothing to enforce

        kept = [b for b in items if self._CITATION_RE.search("\n".join(b))]
        dropped = len(items) - len(kept)
        if dropped:
            print(f"[SOURCING] dropped {dropped} list item(s) with no valid citation")

        if not kept:
            return (
                "I couldn't find any items in the documents that I can attribute to a "
                "specific source, so I'm not listing unsupported claims. Try narrowing the "
                "question (a specific company, sector, or topic)."
            )

        out = list(preamble)
        for i, block in enumerate(kept, 1):
            out.append(re.sub(r"^(\s*)\d+[).]", rf"\g<1>{i})", block[0], count=1))
            out.extend(block[1:])
        return "\n".join(out)

    def _select_cited_chunks(
        self, answer: str, chunks: List[PDFChunk]
    ) -> List[PDFChunk]:
        """Return only the chunks the answer actually cites — not the whole retrieved
        pool. Sources surfaced to the user should reflect what the model used, not every
        low-similarity chunk that happened to be fetched as context.

        A chunk is kept if its (filename, page) appears in a parsed citation. As a
        fallback for filenames whose page spec didn't parse (e.g. an unclosed paren in
        the name), a chunk is also kept if its filename is cited but none of that file's
        pages parsed — so the source still surfaces rather than vanishing.
        """
        cited_pages: set = set()       # (filename_lower, page)
        cited_files: set = set()       # filename_lower
        for m in self._CITATION_RE.finditer(answer):
            fn = m.group(1).lower()
            cited_files.add(fn)
            for p in self._expand_pages(m.group(2)):
                cited_pages.add((fn, p))
        files_with_pages = {fn for fn, _ in cited_pages}

        selected: List[PDFChunk] = []
        for c in chunks:
            if not c.document:
                continue
            fn = c.document.filename.lower()
            if (fn, c.page_number) in cited_pages:
                selected.append(c)
            elif fn in cited_files and fn not in files_with_pages:
                selected.append(c)  # filename cited but page spec unparseable
        print(
            f"[SOURCES] retrieved={len(chunks)} cited={len(selected)} "
            f"(documents: {len({c.document_id for c in selected})})"
        )
        return selected

    # Inventory/list-related words that carry no semantic-search value. Used to decide
    # whether a list query has a real *concept* to rank by, or is pure metadata.
    _LIST_STOPWORDS = frozenset({
        "list", "lists", "all", "show", "me", "give", "giving", "document", "documents",
        "doc", "docs", "report", "reports", "file", "files", "what", "which", "do",
        "does", "you", "your", "have", "having", "from", "about", "any", "the", "a",
        "an", "of", "on", "for", "please", "i", "want", "wanted", "see", "related",
        "relevant", "to", "associated", "with", "by", "in", "is", "are", "and", "or",
        "that", "talk", "talks", "talking", "cover", "covers", "covering", "mention",
        "mentions", "mentioning", "how", "many", "there", "find", "get", "got", "us",
    })

    def _has_semantic_intent(
        self, standalone_query: str, merged_filters: RetrievalFilters
    ) -> bool:
        """True if the query carries a concept worth semantic ranking, beyond the
        metadata filters already extracted. 'all SinoPac documents' → False (pure
        metadata); 'SinoPac docs about supply chain' → True (concept present)."""
        text = (standalone_query or "").lower()
        for name in (merged_filters.sender_companies or []):
            text = text.replace(name.lower(), " ")
        for tk in (merged_filters.tickers or []):
            text = text.replace(tk.lower(), " ")
        if merged_filters.sector:
            text = text.replace(merged_filters.sector.lower(), " ")
        tokens = [t for t in re.findall(r"[a-z0-9]+", text) if t not in self._LIST_STOPWORDS]
        return len(tokens) > 0

    # Vague verbs/time words that carry no specific topic. A query made up of only these
    # (plus a company/period filter) is an underspecified "what did X say" request. Note:
    # "overview"/"summary"/"all"/"everything" are deliberately NOT here — those ARE the
    # user's intent (a broad summary), so they should pass through, not trigger a re-ask.
    _BROAD_STOPWORDS = _LIST_STOPWORDS | frozenset({
        "has", "had", "said", "say", "says", "saying", "tell", "tells", "told", "telling",
        "think", "thinks", "thought", "thinking", "discuss", "discussed", "discusses",
        "comment", "comments", "commented", "commentary", "note", "noted", "notes",
        "view", "views", "stance", "been", "was", "were", "did", "during", "recent",
        "recently", "latest", "quarter", "quarters", "half", "year", "years", "their",
        "they", "it", "its", "anything",
    })

    @staticmethod
    def _has_active_filter(merged_filters: RetrievalFilters) -> bool:
        """True if any metadata filter scopes the search (company/ticker/sector/date/etc.)."""
        return any([
            merged_filters.sender_companies, merged_filters.tickers, merged_filters.sector,
            merged_filters.report_type, merged_filters.asset_class,
            merged_filters.coverage_period_from, merged_filters.coverage_period_to,
            merged_filters.written_date_from, merged_filters.written_date_to,
        ])

    def _is_underspecified(self, question: str, merged_filters: RetrievalFilters) -> bool:
        """Deterministic fallback for the model's is_underspecified judgment (used only if
        the analysis call errored). True when a metadata filter scopes the search but the
        question names NO concrete topic — e.g. 'what has SinoPac said in Q1 2025?'."""
        if not self._has_active_filter(merged_filters):
            return False  # no scope to anchor a clarification — answer normally
        text = (question or "").lower()
        for name in (merged_filters.sender_companies or []):
            text = text.replace(name.lower(), " ")
        for tk in (merged_filters.tickers or []):
            text = text.replace(tk.lower(), " ")
        if merged_filters.sector:
            text = text.replace(merged_filters.sector.lower(), " ")
        # Strip explicit period mentions (Q1, H2, months, years) — they're the filter, not a topic.
        text = re.sub(r"\bq[1-4]\b|\bh[12]\b|\b(?:19|20)\d{2}\b", " ", text)
        text = re.sub(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", " ", text
        )
        # len > 1 drops stray single chars left by possessives/contractions ("what's" → "s").
        tokens = [
            t for t in re.findall(r"[a-z0-9]+", text)
            if len(t) > 1 and t not in self._BROAD_STOPWORDS
        ]
        return len(tokens) == 0

    def _clarify_message(self, merged_filters: RetrievalFilters, analysis: dict) -> dict:
        """Build a clarification response for an underspecified, filter-scoped query: state
        the scope + how many documents match, and offer ways to narrow. No generation call."""
        docs = self.db.list_documents_filtered(
            document_ids=merged_filters.document_ids,
            filenames=merged_filters.filenames,
            sender_names=merged_filters.sender_names,
            sender_companies=merged_filters.sender_companies,
            written_date_from=merged_filters.written_date_from,
            written_date_to=merged_filters.written_date_to,
            tickers=merged_filters.tickers,
            report_type=merged_filters.report_type,
            sector=merged_filters.sector,
            asset_class=merged_filters.asset_class,
            coverage_period_from=merged_filters.coverage_period_from,
            coverage_period_to=merged_filters.coverage_period_to,
        )
        scope_bits = []
        if merged_filters.sender_companies:
            scope_bits.append(", ".join(merged_filters.sender_companies))
        if merged_filters.coverage_period_from or merged_filters.coverage_period_to:
            scope_bits.append(
                f"covering {merged_filters.coverage_period_from or '…'} to "
                f"{merged_filters.coverage_period_to or '…'}"
            )
        scope = " ".join(scope_bits) if scope_bits else "your filters"
        n = len(docs)
        answer = (
            f"That's a broad question — I found **{n}** matching document{'s' if n != 1 else ''} "
            f"for {scope}. To give you a focused, well-sourced answer, what would you like?\n"
            f"1) An overall summary of the key themes across all of them\n"
            f"2) A specific stock, ticker, or sector (e.g. \"China Mobile\", \"offshore wind\")\n"
            f"3) A particular topic (e.g. earnings, technical setups, macro outlook)\n"
            f"Reply with one and I'll dig into the details."
        )
        return {
            "answer": answer,
            "chunks_used": [],
            "inferred_filters": analysis.get("hard_filters") or {},
            "query_type": "clarify",
            "is_enumeration": False,
        }

    def _format_document_list(
        self,
        docs: List[PDFDocument],
        doc_pages: Optional[Dict[int, List[int]]] = None,
    ) -> str:
        """Deterministically format a document inventory as markdown — no LLM call.

        Each line is the document name plus the relevant page numbers. When no specific
        pages are supplied for a document (doc_pages is None or empty for it), the whole
        document is relevant, so pages are shown as "All".
        """
        count = len(docs)
        lines = [f"Found **{count}** matching document{'s' if count != 1 else ''}:\n"]
        for i, doc in enumerate(docs, 1):
            pages = (doc_pages or {}).get(doc.id) if doc_pages else None
            if pages:
                pages_str = ", ".join(str(p) for p in sorted(set(pages)))
            else:
                pages_str = "All"
            # Numbered item is the document; its page reference (the source pointer) sits
            # directly underneath so each entry carries its own reference.
            lines.append(f"{i}) **{doc.filename}**")
            lines.append(f"   Source: Pages {pages_str}")
        return "\n".join(lines)

    def _answer_list_query(
        self,
        question: str,
        standalone_query: str,
        merged_filters: RetrievalFilters,
        analysis: dict,
    ) -> dict:
        """Handle list-type queries: return all matching documents as an organised inventory.

        Output is always a deterministic markdown list of document names + relevant pages
        (no generation LLM call). Two retrieval paths feed that formatter:
          * Pure metadata ("all SinoPac documents") → single SQL query, every doc's pages
            shown as "All". No embedding call — near-instant.
          * Concept / hybrid ("docs about supply chain") → semantic ranking pass; each
            doc lists the specific page numbers whose chunks matched.
        """
        # Step 1: metadata lookup — all docs matching hard filters, no limit.
        meta_docs = self.db.list_documents_filtered(
            document_ids=merged_filters.document_ids,
            filenames=merged_filters.filenames,
            sender_names=merged_filters.sender_names,
            sender_companies=merged_filters.sender_companies,
            written_date_from=merged_filters.written_date_from,
            written_date_to=merged_filters.written_date_to,
            tickers=merged_filters.tickers,
            report_type=merged_filters.report_type,
            sector=merged_filters.sector,
            asset_class=merged_filters.asset_class,
            coverage_period_from=merged_filters.coverage_period_from,
            coverage_period_to=merged_filters.coverage_period_to,
        )

        # ── Fast path: pure-metadata listing (no concept to rank by) ──────────
        # Skip the embedding + generation Gemini calls entirely; format in Python.
        if not self._has_semantic_intent(standalone_query, merged_filters):
            print("[DEBUG] list_documents fast path (pure metadata, no LLM calls)")
            if not meta_docs:
                return {
                    "answer": "No documents matching your criteria were found in the database.",
                    "chunks_used": [],
                    "inferred_filters": analysis.get("hard_filters") or {},
                    "query_type": "list_documents",
                }
            return {
                "answer": self._format_document_list(meta_docs),
                "chunks_used": [],
                "inferred_filters": analysis.get("hard_filters") or {},
                "query_type": "list_documents",
            }

        # Step 2: hybrid ranking — same metadata filters, high limit, very low threshold.
        query_emb = embed_text(standalone_query)
        bm25_kws = analysis.get("keywords") or []
        bm25_q = " ".join(bm25_kws) if bm25_kws else standalone_query
        semantic_chunks = self.db.hybrid_search_chunks(
            query_embedding=query_emb,
            query_text=bm25_q,
            limit=50,
            similarity_threshold=0.05,
            candidate_k=100,
            document_ids=merged_filters.document_ids,
            filenames=merged_filters.filenames,
            sender_names=merged_filters.sender_names,
            sender_companies=merged_filters.sender_companies,
            written_date_from=merged_filters.written_date_from,
            written_date_to=merged_filters.written_date_to,
            tickers=merged_filters.tickers,
            report_type=merged_filters.report_type,
            sector=merged_filters.sector,
            asset_class=merged_filters.asset_class,
            coverage_period_from=merged_filters.coverage_period_from,
            coverage_period_to=merged_filters.coverage_period_to,
        )

        # Build doc_id → RRF rank proxy from hybrid results (earlier = better).
        doc_score: dict[int, float] = {}
        for chunk in semantic_chunks:
            if chunk.document_id not in doc_score:
                # We don't have the raw distance here, so use insertion order as proxy.
                # Assign descending scores so earlier (more relevant) chunks rank higher.
                doc_score[chunk.document_id] = len(semantic_chunks) - len(doc_score)

        # Step 3: merge — metadata docs ranked by semantic score, then by date.
        if not meta_docs and not semantic_chunks:
            return {
                "answer": "No documents matching your criteria were found in the database.",
                "chunks_used": [],
                "inferred_filters": analysis.get("hard_filters") or {},
                "query_type": "list_documents",
            }

        # If no metadata filters were applied, fall back to semantic-only results.
        if not meta_docs and semantic_chunks:
            # Deduplicate semantic chunks by document, preserving relevance order.
            seen: set[int] = set()
            ranked_doc_ids: list[int] = []
            for chunk in semantic_chunks:
                if chunk.document_id not in seen:
                    seen.add(chunk.document_id)
                    ranked_doc_ids.append(chunk.document_id)
            # Fetch full PDFDocument objects for these IDs.
            all_docs = self.db.list_documents_filtered(document_ids=ranked_doc_ids)
            doc_map = {d.id: d for d in all_docs}
            ranked_docs = [doc_map[did] for did in ranked_doc_ids if did in doc_map]
        else:
            # Sort metadata docs: those with a semantic score first (by score desc),
            # then remaining docs by date.
            ranked_docs = sorted(
                meta_docs,
                key=lambda d: (-doc_score.get(d.id, -1),
                               -(d.written_date.toordinal() if d.written_date else 0)),
            )

        # Step 4: collect the relevant page numbers per document from the matching chunks.
        # Only page-level chunks carry a page_number; doc/section-level matches (page=None)
        # imply the whole document is relevant, so those docs fall back to "All".
        doc_pages: Dict[int, List[int]] = {}
        for chunk in semantic_chunks:
            if chunk.page_number is not None:
                doc_pages.setdefault(chunk.document_id, []).append(chunk.page_number)

        return {
            "answer": self._format_document_list(ranked_docs, doc_pages),
            "chunks_used": [],
            "inferred_filters": analysis.get("hard_filters") or {},
            "query_type": "list_documents",
        }

    # ── Specialized response tracks ────────────────────────────────────────────

    def _handle_point_fact(
        self,
        question: str,
        merged_filters: "RetrievalFilters",
        analysis: dict,
    ) -> dict:
        """Fast SQL metadata lookup for point-fact queries (no embedding, no generation).

        Answers questions like "What's TSMC's current target price?" or "Did Goldman
        upgrade Apple?" by reading structured columns from pdf_documents directly.
        Falls back to the full RAG path if no useful metadata rows are found.
        """
        rows = self.db.get_metadata_for_point_fact(
            sender_companies=merged_filters.sender_companies,
            tickers=merged_filters.tickers,
            written_date_from=merged_filters.written_date_from,
            written_date_to=merged_filters.written_date_to,
            limit=8,
        )
        if not rows:
            return {}  # signal to caller to fall through to deep_dive

        _ACTION_LABELS = {"u": "Upgrade", "d": "Downgrade", "id": "Initiation", "m": "Maintenance"}

        lines = ["Here is the latest structured data from the research database:\n"]
        for r in rows:
            broker = r.get("broker") or "Unknown"
            date_str = r.get("written_date") or "date unknown"
            action = _ACTION_LABELS.get(r.get("broker_action") or "", "")
            rating = r.get("rating") or ""
            tp = r.get("target_price")
            tp_str = f"  Target price: **{tp}**\n" if tp is not None else ""
            action_str = f"  Action: **{action}**\n" if action else ""
            rating_str = f"  Rating: **{rating}**\n" if rating else ""
            summary = (r.get("dense_summary") or "").strip()
            summary_str = f"  Summary: {summary[:300]}\n" if summary else ""
            lines.append(
                f"**{broker}** ({date_str}) — {r.get('filename', '')}\n"
                f"{action_str}{rating_str}{tp_str}{summary_str}"
            )

        print(f"[point_fact] answered from {len(rows)} metadata row(s), no embedding needed")
        return {
            "answer": "\n".join(lines),
            "chunks_used": [],
            "inferred_filters": analysis.get("hard_filters") or {},
            "query_type": "point_fact",
            "is_enumeration": False,
        }

    def _handle_comparison(
        self,
        question: str,
        standalone_query: str,
        merged_filters: "RetrievalFilters",
        analysis: dict,
        top_k: int,
        trimmed_history: list,
    ) -> dict:
        """Parallel retrieval across multiple subjects for comparison queries.

        When the user asks "what do Goldman vs JPMorgan say about TSMC?", retrieves
        chunks from each broker separately and builds a sectioned context so the
        synthesizer can produce a structured side-by-side answer.

        Falls back to the standard deep_dive path if subjects cannot be identified.
        """
        subjects = merged_filters.sender_companies or []
        if len(subjects) < 2:
            return {}  # fall through to deep_dive

        # Retrieve separately for each broker (up to first 3)
        subject_chunks: Dict[str, List] = {}
        for subject in subjects[:3]:
            sub_filter = RetrievalFilters(
                sender_companies=[subject],
                tickers=merged_filters.tickers,
                written_date_from=merged_filters.written_date_from,
                written_date_to=merged_filters.written_date_to,
                coverage_period_from=merged_filters.coverage_period_from,
                coverage_period_to=merged_filters.coverage_period_to,
                sector=merged_filters.sector,
                report_type=merged_filters.report_type,
                asset_class=merged_filters.asset_class,
            )
            chunks = self.retrieve_relevant_chunks(
                standalone_query, top_k=max(top_k, 5), filters=sub_filter,
                similarity_threshold=self.SIMILARITY_THRESHOLD_FEW_FILTERS,
                bm25_keywords=analysis.get("keywords") or [],
            )
            subject_chunks[subject] = chunks
            print(f"[comparison] {subject}: {len(chunks)} chunks")

        all_chunks = [c for cs in subject_chunks.values() for c in cs]
        if not all_chunks:
            return {}

        # Build sectioned context with a label per broker
        context_parts: List[str] = []
        context_refs: set = set()
        seen: set = set()
        for subject, chunks in subject_chunks.items():
            if not chunks:
                continue
            context_parts.append(f"\n{'='*60}\n[{subject.upper()}]\n{'='*60}")
            for n, chunk in enumerate(chunks, 1):
                parent, prev_sib, next_sib = self._get_chunk_family(chunk)
                block = self._format_chunk_block(
                    chunk, parent, prev_sib, next_sib, seen, context_refs,
                    label=f"{subject} chunk {n}"
                )
                context_parts.append(block)
        context = "\n\n".join(context_parts)

        comparison_instruction = (
            "The user is asking for a comparison. Structure your answer with one section per "
            "subject (broker/institution) using a bold header (e.g. **Goldman Sachs**). "
            "Within each section: summarize that subject's view, key metrics, and recommendations "
            "drawn from the provided context. End with a brief 'Comparison Summary' paragraph "
            "highlighting key agreements and divergences. Apply the same citation rules as usual: "
            "cite sources at the end of each paragraph, not inline."
        )

        system = (
            "You are a financial analysis assistant. Your answers must be grounded exclusively "
            "in the provided document context — never in your training knowledge.\n\n"
            + comparison_instruction + "\n\n"
            "Each chunk has a 'source:' line — use these exact filenames and pages for citations."
        )
        system_and_context = f"{system}\n\nContext:\n{context}"

        if trimmed_history:
            contents = []
            for i, msg in enumerate(trimmed_history):
                role = "user" if msg["role"] == "user" else "model"
                text = (f"{system_and_context}\n\n{question}" if i == 0 else msg["content"])
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
            if trimmed_history[-1]["role"] != "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        else:
            contents = [types.Content(role="user", parts=[types.Part(text=f"{system_and_context}\n\nQuestion: {question}")])]

        response = self.client.models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
            config={"temperature": 0.1},
        )
        raw_answer = (response.text or "").strip()
        verified = self._verify_citations(raw_answer, all_chunks, context_refs)
        consolidated = self._consolidate_citations(verified)

        print(f"[comparison] synthesized across {len(subjects)} subjects, {len(all_chunks)} total chunks")
        return {
            "answer": consolidated,
            "chunks_used": self._select_cited_chunks(consolidated, all_chunks),
            "inferred_filters": analysis.get("hard_filters") or {},
            "query_type": "comparison",
            "is_enumeration": False,
        }

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
        history: Optional[List[dict]] = None,
    ) -> dict:
        """Answer a question using RAG with optional conversation history.

        Pipeline:
          1. _analyze_query  → hard_filters + standalone_query (one Gemini Flash call)
          2. _merge_filters  → combine inferred filters with explicit caller filters
          3. retrieve_relevant_chunks(standalone_query, merged_filters)
          4. _build_context  → format retrieved chunks
          5. Multi-turn Gemini generation with history interlaced as user/model turns

        Args:
            question: The user's current question (original, not rewritten).
            top_k: Number of chunks to retrieve.
            filters: Explicit filters set by the user (always override inferred).
            history: Recent chat messages [{"role": "user"|"assistant", "content": str}, ...].
                     The last HISTORY_WINDOW entries are used.
        """
        if filters is None:
            filters = RetrievalFilters()

        # Load user preference memory and build a hint for query analysis.
        memory = get_user_memory()
        user_hint = memory.get_hint()

        # Step 1 & 2: Analyze query, extract hard filters + standalone search query.
        analysis = self._analyze_query(question, history=history, user_hint=user_hint)
        standalone_query = analysis.get("standalone_query") or question
        is_followup = analysis.get("is_followup", False)
        trimmed_history = (history or [])[-HISTORY_WINDOW:]

        query_type = analysis.get("query_type", "rag")
        response_track = analysis.get("response_track", "deep_dive")
        # Whether the user asked for an enumeration of content (e.g. "10 trends"). Used by
        # the frontend to deterministically number list items as 1), 2), 3), ...
        is_enumeration = _is_content_enumeration(question)
        print(f"[DEBUG] standalone_query: {standalone_query!r}")
        print(f"[DEBUG] is_followup: {is_followup}")
        print(f"[DEBUG] query_type: {query_type!r}")
        print(f"[DEBUG] response_track: {response_track!r}")
        print(f"[DEBUG] is_enumeration: {is_enumeration}")
        print(f"[DEBUG] is_underspecified(llm): {analysis.get('is_underspecified')}")
        print(f"[DEBUG] inferred hard_filters: {analysis.get('hard_filters')}")

        citation_instruction = (
            "Organize the answer into short paragraphs, each covering one idea or theme. "
            "Do NOT scatter citations inline after every sentence — it makes the text hard "
            "to read. Instead, write the prose cleanly, then place the supporting citations "
            "together at the END of each paragraph on their own line, e.g. "
            "'Sources: (report_a.pdf, p.3), (report_b.pdf, p.7)'.\n"
            "CITE PRECISELY — this is critical:\n"
            "- Cite a (filename, page) ONLY if a specific statement in that paragraph is "
            "directly drawn from that exact page's content. \n"
            "- Do NOT list every page that happens to appear in the context. The context "
            "includes neighbouring 'Parent'/'sibling' pages purely for background; do not "
            "cite those unless you actually used a fact from them.\n"
            "- The citation list should be the MINIMAL set of pages that supports what you "
            "wrote. If a short paragraph cites many pages, that is a red flag you are "
            "over-citing — trim it to the pages that genuinely back a stated fact.\n"
            "- Never cite a page to signal it is 'related'; only cite it as the source of a "
            "specific claim you made.\n"
            "LISTS — when the user asks for a number of items (e.g. '10 trends', '5 risks') "
            "or the answer is naturally an enumeration, you MUST format it as a numbered "
            "list and follow ALL of these rules:\n"
            "  • Start EVERY item with its number and a closing parenthesis: '1)', '2)', "
            "'3)', ... (not '1.', not a bullet).\n"
            "  • Immediately BELOW each item, on its own line, put that item's citation "
            "prefixed with 'Source:'. Include a page if you are sure of it "
            "('Source: (filename.pdf, p.N)'); if you know the document but not the exact "
            "page, cite the document alone ('Source: (filename.pdf)') — that is fine.\n"
            "  • EVERY item MUST have a Source line naming a document that is actually in the "
            "provided context. If an item cannot be tied to any document in the context, DO "
            "NOT include it — choose a different, attributable one. Never output an item with "
            "no source, and never name a document that is not in the context.\n"
            "  • Each item's Source line cites only the document(s) backing THAT item; never "
            "share one citation block across items or move citations to the end.\n"
            "Example:\n"
            "  1) Revenue grew 12% year over year.\n"
            "     Source: (report_a.pdf, p.3)\n"
            "  2) Operating margin expanded to 18%.\n"
            "     Source: (report_b.pdf)"
        )

        # Follow-ups ("expand on that", "what about Baidu's Ernie model?", "summarize the
        # themes") flow through the normal RAG path below — which already interleaves the
        # conversation history into generation — so the model can RETRIEVE fresh document
        # content when the follow-up needs it, instead of being limited to what was already
        # said. is_followup only relaxes the retrieval threshold (a reference-y rewrite can
        # score low), so the follow-up still pulls relevant chunks rather than coming back empty.

        # ── List-documents path ───────────────────────────────────────────────
        # For queries that enumerate files rather than ask for content analysis,
        # skip chunk RAG entirely and return a document inventory.
        merged_filters = self._merge_filters(filters, analysis)

        if query_type == "list_documents":
            return self._answer_list_query(question, standalone_query, merged_filters, analysis)

        # ── Specialized response tracks ────────────────────────────────────────

        if response_track == "point_fact" and not is_followup:
            result = self._handle_point_fact(question, merged_filters, analysis)
            if result:
                _record_memory(memory, analysis)
                return result
            # Empty result → fall through to deep_dive
            print("[point_fact] no metadata rows found, falling back to deep_dive")
            response_track = "deep_dive"

        if response_track == "comparison" and not is_followup:
            result = self._handle_comparison(
                question, standalone_query, merged_filters, analysis,
                top_k, trimmed_history,
            )
            if result:
                _record_memory(memory, analysis)
                return result
            print("[comparison] could not identify ≥2 subjects, falling back to deep_dive")
            response_track = "deep_dive"

        # ── Normal RAG path (deep_dive / sector_sweep / fallback) ─────────────

        # Clarification gate: a query scoped to a company/period but naming no concrete topic
        # (e.g. "what has SinoPac said in Q1 2025?") yields an invented search query and
        # diffuse retrieval. Ask the user to narrow rather than guessing. Skip on follow-ups
        # (handled above) so a reply like "topic X" still gets answered.
        #
        # Prefer the model's judgment (folded into _analyze_query — no extra call) and only
        # clarify when a filter actually scopes the search. Fall back to the deterministic
        # heuristic if the analysis call errored and didn't return the field.
        if "is_underspecified" in analysis:
            underspecified = analysis["is_underspecified"] and self._has_active_filter(merged_filters)
        else:
            underspecified = self._is_underspecified(question, merged_filters)
        if underspecified:
            print("[DEBUG] underspecified query → asking user to clarify")
            return self._clarify_message(merged_filters, analysis)

        # Tiered similarity threshold: relax when metadata filters already constrain the pool.
        _metadata_filter_fields = [
            merged_filters.sender_companies,
            merged_filters.tickers,
            merged_filters.written_date_from,
            merged_filters.written_date_to,
            merged_filters.coverage_period_from,
            merged_filters.coverage_period_to,
            merged_filters.report_type,
            merged_filters.sector,
            merged_filters.asset_class,
        ]
        _active_filter_count = sum(1 for f in _metadata_filter_fields if f)
        if _active_filter_count == 0:
            effective_threshold = self.SIMILARITY_THRESHOLD
        elif _active_filter_count <= 2:
            effective_threshold = self.SIMILARITY_THRESHOLD_FEW_FILTERS
        else:
            effective_threshold = self.SIMILARITY_THRESHOLD_MANY_FILTERS
        print(f"[DEBUG] active_metadata_filters={_active_filter_count}, similarity_threshold={effective_threshold}")

        # Enumeration ("10 trends across X"): override for document breadth. Relax the
        # threshold (the filter already guarantees relevance) and cap chunks-per-document so
        # retrieval spreads across many docs — giving the model a real page to cite for each
        # item instead of fabricating one (which the verifier then strips, leaving it blank).
        per_document_cap = None
        if is_enumeration:
            effective_threshold = self.ENUMERATION_SIMILARITY_THRESHOLD
            per_document_cap = self.ENUMERATION_PER_DOC_CAP
            print(f"[DEBUG] enumeration retrieval: threshold={effective_threshold}, "
                  f"per_document_cap={per_document_cap}")
        elif is_followup:
            # A follow-up's rewritten query is often reference-y and scores low, so relax the
            # threshold to make sure it still pulls relevant chunks (the conversation history,
            # interleaved into generation below, keeps the model on-topic).
            effective_threshold = min(effective_threshold, self.SIMILARITY_THRESHOLD_FEW_FILTERS)
            print(f"[DEBUG] follow-up retrieval: relaxed threshold={effective_threshold}")

        # Step 3: Retrieve chunks using the focused standalone query.
        chunks = self.retrieve_relevant_chunks(
            standalone_query, top_k=top_k, filters=merged_filters,
            similarity_threshold=effective_threshold, per_document_cap=per_document_cap,
            bm25_keywords=analysis.get("keywords") or [],
        )

        if not chunks:
            return {
                "answer": (
                    "No sufficiently relevant content was found in the uploaded documents "
                    "to answer this question. Try rephrasing, or check that the relevant "
                    "document has been uploaded and its embeddings backfilled."
                ),
                "chunks_used": [],
                "inferred_filters": analysis.get("hard_filters") or {},
                "query_type": "rag",
                "is_enumeration": is_enumeration,
            }

        # Step 4: Build context string from retrieved chunks.
        # context_refs = every (filename, page) shown to the model, for citation verification.
        context, context_refs = self._build_context(chunks)

        system = (
            "You are a financial analysis assistant. Your answers must be grounded exclusively "
            "in the provided document context below — never in your training knowledge.\n\n"
            "Rules:\n"
            "1. Every specific claim (numbers, dates, company actions, prices, forecasts) "
            "must be directly supported by the context. Keep the prose clean: group related "
            "claims into a paragraph and place the supporting (filename, page) citations "
            "TOGETHER at the end of that paragraph — never inline after each sentence. Cite "
            "PRECISELY: only the exact pages a stated fact came from, never every page in the "
            "context, and never neighbouring background (Parent/sibling) pages you did not "
            "actually draw a fact from. A short paragraph with many cited pages means you are "
            "over-citing — trim to the minimal supporting set.\n"
            "2. You MAY reason, synthesize, and identify trends across the provided chunks — "
            "but only from what the documents actually say. Drawing a conclusion like "
            "'revenue has trended upward across these reports' is allowed if the chunks support it.\n"
            "3. You may NOT use general industry knowledge, assumptions, or facts from your "
            "training data to fill gaps. If the context does not contain enough information "
            "to support a claim, say so explicitly: 'The provided documents do not state...'\n"
            "4. If the question cannot be answered at all from the context, say: "
            "'The uploaded documents do not contain sufficient information to answer this question.'\n"
            "5. Do not invent or guess document names, page numbers, or figures.\n\n"
            "Each chunk has a 'source:' line with the filename and page — use these for citations."
        )
        system_and_context = f"{system}\n\nContext:\n{context}"

        # Step 5: Build contents for Gemini.
        # With history: interleave prior turns so the model has conversation context.
        # Without history: single-turn (identical to original behaviour).
        if trimmed_history:
            contents = []
            # Inject system prompt + retrieved context into the very first user turn
            # so all subsequent turns are grounded in the same document context.
            first_question = trimmed_history[0]["content"]
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"{system_and_context}\n\nQuestion: {first_question}")],
            ))
            for msg in trimmed_history[1:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
            # Final turn: the current question (original, not standalone_query)
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=f"Question: {question}\n\n{citation_instruction}")],
            ))
        else:
            # Stateless single-turn path
            contents = (
                f"{system_and_context}\n\n"
                f"Question: {question}\n\n"
                f"{citation_instruction}"
            )

        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=contents,
                    config={"temperature": 0},
                )
                break
            except google.api_core.exceptions.ResourceExhausted:
                if attempt == 3:
                    raise
                wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                print(f"[WARNING] Gemini rate limited, retrying in {wait}s (attempt {attempt + 1}/4)")
                time.sleep(wait)
        answer = response.text if hasattr(response, "text") else str(response)

        # Verify citations against what the model was actually shown (deterministic, no
        # extra LLM call): strip any fabricated (filename, page) pairs and log a breakdown.
        answer = self._verify_citations(answer, chunks, context_refs)
        # Merge repeated same-document citations: (doc, p.1), (doc, p.2) → (doc, pp. 1, 2).
        answer = self._consolidate_citations(answer)
        # For lists, drop any item whose citation didn't survive verification so the user
        # never sees an unsourced item (guarantees every item carries a citation).
        if is_enumeration:
            answer = self._enforce_sourced_items(answer)

        # Surface only the chunks the answer actually cited as sources — not the full
        # retrieved pool (which includes low-similarity chunks the model never used).
        cited_chunks = self._select_cited_chunks(answer, chunks)

        _record_memory(memory, analysis)

        return {
            "answer": answer,
            "chunks_used": [
                {
                    "chunk_id": str(c.id),
                    "document_id": c.document_id,
                    "page_number": c.page_number,
                    "metadata": c.metadata_ or {},
                }
                for c in cited_chunks
            ],
            "inferred_filters": analysis.get("hard_filters") or {},
            "query_type": response_track if response_track != "deep_dive" else "rag",
            "is_enumeration": is_enumeration,
        }

    def _diversify_by_document(
        self, chunks: List[PDFChunk], top_k: int, per_doc_cap: int
    ) -> List[PDFChunk]:
        """Spread retrieval across documents for breadth. Round-robins one chunk at a time
        from each document (best-first within a doc, since `chunks` arrive in similarity
        order), taking at most `per_doc_cap` per document. So pass 1 grabs every document's
        top chunk before any document gets a second — no single doc can monopolise the
        budget, and the model gets a citable page from many distinct documents."""
        by_doc: Dict[int, List[PDFChunk]] = {}
        for c in chunks:  # already ordered best → worst by similarity
            by_doc.setdefault(c.document_id, []).append(c)

        # Within each document, prefer PAGE-LEVEL chunks (they carry a citable page number)
        # over document/section summaries — otherwise the model gets only summaries (page=None)
        # and can't cite anything. sort() is stable, so similarity order is preserved within
        # each group. (Summaries still reach the model via _build_context's parent expansion.)
        for doc_id in by_doc:
            by_doc[doc_id].sort(key=lambda c: c.page_number is None)

        selected: List[PDFChunk] = []
        counts: Dict[int, int] = {doc_id: 0 for doc_id in by_doc}
        while len(selected) < top_k:
            progressed = False
            for doc_id, queue in by_doc.items():
                if counts[doc_id] >= per_doc_cap or not queue:
                    continue
                selected.append(queue.pop(0))
                counts[doc_id] += 1
                progressed = True
                if len(selected) >= top_k:
                    break
            if not progressed:
                break  # every doc is exhausted or at its cap
        return selected

    def _diversify_chunks(self, chunks: List[PDFChunk], top_k: int) -> List[PDFChunk]:
        """
        Promote diversity across sections and hierarchy levels.

        Strategy:
            - Prefer document-level and section-level chunks.
            - Spread page-level chunks across different sections.
        """
        # Keep original ranking index as tie-breaker
        ranked = list(enumerate(chunks))

        doc_level: List[Tuple[int, PDFChunk]] = []
        section_level: Dict[str, List[Tuple[int, PDFChunk]]] = {}
        page_level: Dict[Tuple[int, Optional[str]], List[Tuple[int, PDFChunk]]] = {}

        for idx, c in ranked:
            meta = c.metadata_ or {}
            level = meta.get("level")
            section_id = meta.get("section_id")

            if level == "document":
                doc_level.append((idx, c))
            elif level == "section":
                section_level.setdefault(section_id or f"sec-{idx}", []).append((idx, c))
            else:
                key = (c.document_id, section_id)
                page_level.setdefault(key, []).append((idx, c))

        selected: List[PDFChunk] = []

        # 1) At most one document-level chunk per document.
        for _, c in sorted(doc_level, key=lambda t: t[0]):
            if len(selected) >= top_k:
                break
            if c not in selected:
                selected.append(c)

        if len(selected) >= top_k:
            return selected[:top_k]

        # 2) One section-level chunk per section in order.
        for sec_id, items in sorted(section_level.items(), key=lambda kv: kv[0] or ""):
            if len(selected) >= top_k:
                break
            items_sorted = sorted(items, key=lambda t: t[0])
            _, c = items_sorted[0]
            if c not in selected:
                selected.append(c)

        if len(selected) >= top_k:
            return selected[:top_k]

        # 3) Round-robin across sections for page-level chunks.
        # Convert dict values to queues.
        queues: List[List[Tuple[int, PDFChunk]]] = [
            sorted(v, key=lambda t: t[0]) for _, v in sorted(page_level.items(), key=lambda kv: kv[0])
        ]

        exhausted = False
        while len(selected) < top_k and not exhausted:
            exhausted = True
            for q in queues:
                if not q:
                    continue
                exhausted = False
                _, c = q.pop(0)
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= top_k:
                        break

        return selected[:top_k]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemini RAG over verbalized PDF pages")
    parser.add_argument(
        "--db-url",
        default=os.getenv("PDF_SUMMARIZER_DB_URL", "postgresql+psycopg://user:password@localhost/pdf_summarizer"),
        help="Database URL",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="Backfill embeddings for pages")
    backfill.add_argument("--batch-size", type=int, default=64)
    backfill.add_argument("--max-batches", type=int, default=None)
    backfill.add_argument(
        "--reset", action="store_true",
        help="Clear ALL existing embeddings first, then re-embed every chunk "
             "(use when switching embedding models)",
    )
    backfill.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to pause between batches to stay under API rate/token limits",
    )

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", help="Question")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--filename", action="append", help="Filter by filename")
    ask.add_argument("--doc-id", type=int, action="append", help="Filter by doc id")
    ask.add_argument("--page-min", type=int, default=None)
    ask.add_argument("--page-max", type=int, default=None)

    args = parser.parse_args()
    pipeline = GeminiRAGPipeline(database_url=args.db_url)

    if args.command == "backfill":
        n = pipeline.backfill_embeddings(
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            reset=args.reset,
            sleep=args.sleep,
        )
        print(f"Embedded {n} chunk(s).")
    elif args.command == "ask":
        filters = RetrievalFilters(
            document_ids=args.doc_id,
            filenames=args.filename,
            page_min=args.page_min,
            page_max=args.page_max,
        )
        result = pipeline.answer_question(args.question, top_k=args.top_k, filters=filters)
        print("\n=== Answer ===\n")
        print(result["answer"])
        print("\n=== Chunks used ===")
        for m in result["chunks_used"]:
            print(m)


if __name__ == "__main__":
    main()
