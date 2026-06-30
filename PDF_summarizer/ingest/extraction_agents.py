"""
Ingestion agents (Agents 1–3) — PDF content extraction team.

  Agent 1 — Spatial & Graph Deconstructor  (gemini-2.5-flash, multimodal)
  Agent 2 — Core Financial Matrix Extractor (gemini-2.5-flash + context cache)
  Agent 3 — QA Audit Guard                  (gemini-2.5-pro, failsafe only)

Also exposes extract_metadata(), a 4-stage repair chain that wraps Agents 2 & 3.
"""

import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types

_PDF_SUMMARIZER = Path(__file__).resolve().parent.parent
if str(_PDF_SUMMARIZER) not in sys.path:
    sys.path.insert(0, str(_PDF_SUMMARIZER))

from broker_cache import BrokerContextCache

_FLASH   = "models/gemini-3.5-flash"           # primary extraction (upgraded from 2.5-flash)
_PRO     = "models/gemini-3.1-pro-preview"   # QA fallback (Agent 3) — thinking mode required


def _client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class EpsPeData(BaseModel):
    eps_fy1: Optional[float] = Field(None, description="Current fiscal year EPS estimate")
    eps_fy2: Optional[float] = Field(None, description="Next fiscal year EPS estimate")
    pe_fy1: Optional[float] = Field(None, description="Current fiscal year P/E multiple")
    pe_fy2: Optional[float] = Field(None, description="Next fiscal year P/E multiple")


class FinancialReportMetadata(BaseModel):
    """Structured output schema for Agents 2 & 3. Maps 1:1 to pdf_documents columns."""
    broker: str
    report_date: Optional[date] = None
    broker_action: Literal["u", "d", "id", "m"]
    rating: Optional[Literal["Overweight", "Equal-weight", "Underweight"]] = None
    target_price: Optional[float] = None
    tickers: Optional[List[str]] = Field(None, description="List of ticker symbols mentioned (e.g. ['AAPL', '2330.TW', '00700.HK']). Null for macro/strategy reports with no specific stock coverage.")
    eps_pe: Optional[EpsPeData] = None
    dense_summary: str
    email_type: Literal["sales", "analyst"] = Field(
        "analyst",
        description=(
            "sales = digest/round-up covering many companies or multiple reports in one email "
            "(e.g. morning notes, weekly wrap, sector digests, top-ideas lists). "
            "analyst = single focused report on one company/ticker/theme."
        ),
    )
    report_subtype: Optional[Literal["formal_report", "model_update", "earnings", "brief"]] = Field(
        None,
        description=(
            "Only set when email_type=analyst. "
            "formal_report = full initiation or comprehensive coverage note. "
            "model_update = primarily a financial model revision (EPS/TP change with brief commentary). "
            "earnings = any earnings-related note (preview, preview/review, or post-results analysis). "
            "brief = short commentary, color piece, or quick take (usually <1 page)."
        ),
    )

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
    """Agent 1: Convert a cropped chart/graph image to a Markdown data table."""
    c = _client(api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    prompt = (
        f"Spatial context:\n{context_text}\n\nConvert the chart/table above to Markdown:"
        if context_text else "Convert the chart/table to Markdown:"
    )
    response = c.models.generate_content(
        model=_FLASH,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            system_instruction=_AGENT1_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        request_options=types.RequestOptions(timeout=_API_TIMEOUT),
    )
    return (response.text or "[NO_DATA]").strip()


# ── Agent 2: Core Financial Matrix Extractor ──────────────────────────────────

_API_TIMEOUT = 90  # seconds — prevents hung Gemini connections from stalling ingest


def _run_agent2_raw(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> str:
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
    prompt = (
        "Extract metadata from this broker research email. "
        "The content may be in English, Mandarin, Cantonese, Traditional Chinese, or Simplified Chinese — "
        "extract all fields regardless of language.\n\n"
        "FIELD INSTRUCTIONS:\n"
        "• broker_action: u=upgrade, d=downgrade, id=initiation, m=maintain/reiterate.\n"
        "• rating: Overweight/Equal-weight/Underweight only. "
        "Chinese: 买入/增持=Overweight, 中性/持有=Equal-weight, 减持/卖出=Underweight.\n"
        "• tickers: list every stock/bond ticker symbol explicitly mentioned; null only for pure macro with zero security coverage.\n"
        "• dense_summary: 2–4 coherent English prose sentences — thesis, key findings, outlook. No bullet lists.\n"
        "• email_type: 'sales' if this is a digest/round-up covering many companies or topics in one email "
        "(morning notes, weekly wrap, top-ideas, sector digests). "
        "'analyst' if it is a focused note on one company/ticker.\n"
        "• report_subtype (analyst emails only): "
        "'formal_report'=full initiation or comprehensive coverage note, "
        "'model_update'=mainly EPS/TP revision with brief commentary, "
        "'earnings'=any earnings-related note (pre-results, post-results, or both), "
        "'brief'=short color or quick take (<1 page equivalent). "
        "Set null for sales emails.\n\n"
        f"{aoim_text}"
    )
    response = c.models.generate_content(
        model=_PRO,
        contents=prompt,
        config=config,
        request_options=types.RequestOptions(timeout=_API_TIMEOUT),
    )
    return response.text or ""


def run_agent2(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """Agent 2: Extract structured broker report metadata from AOIM text."""
    raw = _run_agent2_raw(aoim_text, broker, cache, api_key)
    return FinancialReportMetadata.model_validate_json(raw)


def _run_agent2_repair(
    failed_json: str,
    error_message: str,
    aoim_text: str,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """Flash self-correction: feed the validation error back to fix it (~40% of failures)."""
    c = _client(api_key)
    prompt = (
        "The JSON output below failed schema validation. "
        "Fix ONLY the invalid fields — do not change correct ones. "
        "If the source is in Chinese: 买入/增持=Overweight, 中性/持有=Equal-weight, 减持/卖出=Underweight; "
        "dense_summary must be in English.\n\n"
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
        request_options=types.RequestOptions(timeout=_API_TIMEOUT),
    )
    return FinancialReportMetadata.model_validate_json(response.text)


# ── Agent 3: QA Audit Guard ───────────────────────────────────────────────────

_AGENT3_SYSTEM = """\
You are an elite financial data auditor and deep-reasoning engine.

A lower-tier parsing model has broken validation schema constraints or returned ambiguous
data while processing a broker research report. Your task:

1. Analyze the source text carefully. The report may be in English, Mandarin, Cantonese,
   Traditional Chinese, or Simplified Chinese — extract fields regardless of language.
2. Pinpoint the exact field(s) that caused the schema violation.
3. Generate a flawless JSON payload that corrects every constraint failure.

Chinese rating mapping:
  买入 / 增持 / 强烈推荐  →  "Overweight"
  中性 / 持有 / 观望      →  "Equal-weight"
  减持 / 卖出 / 回避      →  "Underweight"

Chinese action mapping:
  上调评级 / 升级  →  "u"  (upgrade)
  下调评级 / 降级  →  "d"  (downgrade)
  首次覆盖 / 开始覆盖  →  "id"  (initiation)
  维持 / 重申  →  "m"  (maintain)

dense_summary must be written in English even if the source is Chinese.
Be meticulous. Every field must conform to the schema contract.
"""


def run_agent3(
    aoim_text: str,
    failed_json: str,
    error_message: str,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """Agent 3: QA fallback — re-extract with gemini-2.5-pro after Agent 2 fails."""
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
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
        request_options=types.RequestOptions(timeout=_API_TIMEOUT),
    )
    return FinancialReportMetadata.model_validate_json(response.text)


# ── Orchestrated extraction (4-stage repair chain) ────────────────────────────

def extract_metadata(
    aoim_text: str,
    broker: str,
    cache: Optional[BrokerContextCache] = None,
    api_key: Optional[str] = None,
) -> FinancialReportMetadata:
    """
    4-stage repair chain wrapping Agents 2 & 3.

    Stage 1 — Agent 2 (Flash, cached): normal extraction.
    Stage 2 — Rule engine: fix obvious formatting errors (zero LLM cost).
    Stage 3 — Agent 2 Flash self-correction: feed error back to Flash.
    Stage 4 — Agent 3 (Pro): last resort, <2% of emails.
    """
    import json as _json

    raw_json = ""
    error_msg = ""

    try:
        raw_json = _run_agent2_raw(aoim_text, broker, cache=cache, api_key=api_key)
        return FinancialReportMetadata.model_validate_json(raw_json)
    except Exception as e:
        error_msg = str(e)
        print(f"[Agent2] Stage 1 failed ({broker}): {error_msg[:80]}")

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

    try:
        result = _run_agent2_repair(raw_json, error_msg, aoim_text, api_key=api_key)
        print("[Agent2] Stage 3 Flash self-correction succeeded")
        return result
    except Exception as e2:
        error_msg = str(e2)
        print(f"[Agent2] Stage 3 failed: {error_msg[:80]} — escalating to Agent 3")

    return run_agent3(aoim_text, raw_json, error_msg, api_key=api_key)
