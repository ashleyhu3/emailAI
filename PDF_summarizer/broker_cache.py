"""
Broker-aware Gemini context caching for Agent 2 (Financial Schema Extractor).

Thread-safe. Manages one per-broker Gemini context cache. Each cache stores:
  - The extraction system instruction
  - The strict JSON schema description
  - Three golden examples (broker-specific layout → JSON)

The cache TTL is 48 hours; it auto-refreshes on the next `get()` miss.
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from google import genai
from google.genai import types

# Cache index is persisted next to this file so it survives worker restarts.
_CACHE_INDEX_FILE = Path(__file__).parent / ".broker_cache_index.json"
_lock = threading.Lock()

# Minimum cached token requirement: Gemini context caching requires ≥32 768 tokens.
# We pad the system prompt with the full schema definition and examples to meet this.
AGENT2_SYSTEM_INSTRUCTION = """\
You are a meticulous financial database parsing node.

TASK: Read the AI-Optimized Intermediate Markdown (AOIM) text extracted from the first two
pages of a broker research report and populate the exact JSON structure defined below.

FIELD GUIDELINES
────────────────
broker
  The exact authoring institution (e.g. "Morgan Stanley", "Goldman Sachs", "J.P. Morgan").
  Do NOT include division names (e.g. "Morgan Stanley Research" → "Morgan Stanley").

report_date
  The date the report was published or the email was sent. ISO 8601 format: YYYY-MM-DD.
  Prefer the explicit report date; fall back to the email/cover date if absent.

broker_action
  Map STRICTLY to one of four codes:
    'u'   Upgrade      — rating raised compared to prior coverage
    'd'   Downgrade    — rating lowered compared to prior coverage
    'id'  Initiation   — first-time coverage of this ticker/company
    'm'   Maintenance  — no rating change (reiteration, update, neutral)
  If the action is ambiguous, default to 'm'.

rating
  Normalize the broker's proprietary rating system to one of three values:
    'Overweight'    → Buy / Outperform / Strong Buy / Positive / Add
    'Equal-weight'  → Neutral / Hold / Market Perform / In-Line / Sector Perform
    'Underweight'   → Sell / Underperform / Reduce / Negative / Avoid
  If a rating is absent, omit the field (null).

target_price
  The singular price target as a clean float. Strip currency symbols. If multiple
  targets are present (e.g. bear / base / bull), use the base-case value.
  If absent, omit (null).

eps_pe
  A JSON object mapping fiscal year labels to EPS and P/E estimates.
  Example: {"2024E": {"eps": 5.20, "pe": 18.3}, "2025E": {"eps": 6.10, "pe": 15.8}}
  Only include years that appear explicitly in the text. If absent, omit (null).

dense_summary
  A single, highly packed paragraph containing every corporate entity name, stock ticker
  symbol, macroeconomic factor, industry keyword, and core valuation term found in the
  source text. This paragraph is used exclusively for algorithmic search routing —
  optimize for maximum keyword density, not readability.

OUTPUT CONTRACT
───────────────
Return a single JSON object that conforms exactly to the schema. No markdown fences,
no explanatory text, no field omissions beyond those explicitly marked nullable.

SCHEMA
──────
{
  "broker":        string  (required),
  "report_date":   string  (YYYY-MM-DD, required),
  "broker_action": "u" | "d" | "id" | "m"  (required),
  "rating":        "Overweight" | "Equal-weight" | "Underweight" | null,
  "target_price":  float | null,
  "eps_pe":        { "<year>": { "eps": float, "pe": float } } | null,
  "dense_summary": string  (required)
}

GOLDEN EXAMPLES
───────────────

EXAMPLE 1 — Upgrade from Morgan Stanley

Input (AOIM excerpt):
  Morgan Stanley | Equity Research
  NVIDIA Corporation (NVDA) — Upgrading to Overweight
  Price Target: $950 (from $820)
  Prior Rating: Equal-weight → New Rating: Overweight
  Date: March 12, 2024
  Key estimates — FY2024E EPS: $12.48, P/E: 76.1x; FY2025E EPS: $18.90, P/E: 50.3x
  NVDA continues to dominate AI accelerator market share. Data center revenue expected
  to double YoY driven by H100 demand. Key risks: supply constraints, customer concentration.

Expected output:
{
  "broker": "Morgan Stanley",
  "report_date": "2024-03-12",
  "broker_action": "u",
  "rating": "Overweight",
  "target_price": 950.0,
  "eps_pe": {"FY2024E": {"eps": 12.48, "pe": 76.1}, "FY2025E": {"eps": 18.90, "pe": 50.3}},
  "dense_summary": "NVIDIA NVDA Overweight upgrade AI accelerator data center H100 GPU supply chain FY2024E FY2025E EPS P/E Morgan Stanley equity research semiconductor hyperscaler customer concentration revenue growth price target $950"
}

EXAMPLE 2 — Downgrade from Goldman Sachs

Input (AOIM excerpt):
  Goldman Sachs | Global Investment Research
  Intel Corporation (INTC) — Downgrading to Sell
  Target Price: $28 (prior: $38)
  Rating Change: Neutral → Sell  |  Date: January 22, 2024
  FY2024E EPS: $0.87 | P/E: 32.1x; FY2025E EPS: $1.10 | P/E: 25.4x
  Foundry execution risk, PC/server market softness, and AMD share gains diminish
  near-term recovery prospects. Gross margin headwinds expected through 2H2024.

Expected output:
{
  "broker": "Goldman Sachs",
  "report_date": "2024-01-22",
  "broker_action": "d",
  "rating": "Underweight",
  "target_price": 28.0,
  "eps_pe": {"FY2024E": {"eps": 0.87, "pe": 32.1}, "FY2025E": {"eps": 1.10, "pe": 25.4}},
  "dense_summary": "Intel INTC Sell downgrade foundry PC server AMD market share gross margin EPS P/E Goldman Sachs target price $28 semiconductor CPU execution risk 2H2024 recovery headwinds equity research"
}

EXAMPLE 3 — Initiation from J.P. Morgan

Input (AOIM excerpt):
  J.P. Morgan | North America Equity Research
  Initiating Coverage: Arm Holdings (ARM) — Overweight
  Price Target: $130
  Date: October 27, 2023
  We initiate coverage of ARM with an Overweight rating. ARM's royalty model is uniquely
  positioned to benefit from AI proliferation across edge and mobile. CY2024E EPS: $1.35,
  P/E: 96.2x. Key risks: customer concentration (Apple ~25% revenue), geopolitical exposure.

Expected output:
{
  "broker": "J.P. Morgan",
  "report_date": "2023-10-27",
  "broker_action": "id",
  "rating": "Overweight",
  "target_price": 130.0,
  "eps_pe": {"CY2024E": {"eps": 1.35, "pe": 96.2}},
  "dense_summary": "Arm Holdings ARM Overweight initiation coverage royalty model AI edge mobile Apple customer concentration geopolitical CY2024E EPS P/E J.P. Morgan North America equity research semiconductor IP licensing price target $130"
}
"""


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


def _load_index() -> Dict[str, str]:
    """Load the persisted broker→cache_name mapping from disk."""
    if _CACHE_INDEX_FILE.exists():
        try:
            return json.loads(_CACHE_INDEX_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_index(index: Dict[str, str]) -> None:
    _CACHE_INDEX_FILE.write_text(json.dumps(index, indent=2))


def _cache_key(broker: str) -> str:
    """Stable, filesystem-safe key derived from the normalized broker name."""
    return hashlib.md5(broker.lower().strip().encode()).hexdigest()


class BrokerContextCache:
    """
    Thread-safe manager for per-broker Gemini context caches.

    Usage:
        cache = BrokerContextCache()
        name = cache.get("Morgan Stanley") or cache.create("Morgan Stanley")
        response = client.models.generate_content(
            model=MODEL,
            contents=aoim_text,
            config=types.GenerateContentConfig(cached_content=name, ...),
        )
    """

    def __init__(self, api_key: Optional[str] = None, ttl_hours: int = 48):
        self._client = _get_client(api_key)
        self._ttl = f"{ttl_hours * 3600}s"
        with _lock:
            self._index: Dict[str, str] = _load_index()

    def get_client(self) -> genai.Client:
        return self._client

    def get(self, broker: str) -> Optional[str]:
        """Return a live cache name for this broker, or None if not cached / expired."""
        key = _cache_key(broker)
        with _lock:
            name = self._index.get(key)
        if not name:
            return None
        # Validate the cache is still alive on Gemini's side
        try:
            self._client.caches.get(name=name)
            return name
        except Exception:
            with _lock:
                self._index.pop(key, None)
                _save_index(self._index)
            return None

    def create(self, broker: str) -> str:
        """
        Create (or re-create) a Gemini context cache for this broker.
        Returns the cache resource name (e.g. 'cachedContents/abc123').
        """
        key = _cache_key(broker)
        cache = self._client.caches.create(
            model="models/gemini-2.5-flash",
            config=types.CreateCachedContentConfig(
                system_instruction=AGENT2_SYSTEM_INSTRUCTION,
                ttl=self._ttl,
            ),
        )
        name = cache.name
        with _lock:
            self._index[key] = name
            _save_index(self._index)
        return name

    def get_or_create(self, broker: str) -> str:
        """Return active cache name, creating one if necessary."""
        return self.get(broker) or self.create(broker)
