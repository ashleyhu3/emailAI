"""
User preference memory for the RAG pipeline.

Tracks which tickers, brokers, and sectors the user queries most frequently
across sessions, then injects a personalised hint into _analyze_query() so
ambiguous queries ("the stock I follow", "my usual broker") can be resolved
without an extra LLM call.

Persistence: a single JSON file at PDF_summarizer/.cache/user_memory.json.
Thread-safe: all mutations go through _save() which writes atomically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_MEMORY_FILENAME = "user_memory.json"
_TOP_N_TICKERS = 8
_TOP_N_BROKERS = 5
_TOP_N_SECTORS = 3
_RECENT_QUERIES_MAX = 30


class UserMemory:
    """Lightweight, persistent preference tracker.

    Counts how often each ticker / broker / sector appears across answered
    queries, and surfaces the top-N in a prompt hint block.  A ``record()``
    call after every successful answer keeps the counts current.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._path = (cache_dir or _DEFAULT_CACHE_DIR) / _MEMORY_FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"tickers": {}, "brokers": {}, "sectors": {}, "recent_queries": []}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # ── mutation ─────────────────────────────────────────────────────────────

    def record(
        self,
        tickers: Optional[List[str]] = None,
        brokers: Optional[List[str]] = None,
        sector: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> None:
        """Update frequency counters after a successful answered query."""
        for tk in (tickers or []):
            if tk:
                key = tk.upper().strip()
                self._data["tickers"][key] = self._data["tickers"].get(key, 0) + 1

        for br in (brokers or []):
            if br:
                key = br.strip()
                self._data["brokers"][key] = self._data["brokers"].get(key, 0) + 1

        if sector:
            key = sector.strip()
            self._data["sectors"][key] = self._data["sectors"].get(key, 0) + 1

        if query_text:
            recent = self._data.setdefault("recent_queries", [])
            recent.append(query_text[:200])
            self._data["recent_queries"] = recent[-_RECENT_QUERIES_MAX:]

        self._save()

    # ── read ─────────────────────────────────────────────────────────────────

    def top_tickers(self, n: int = _TOP_N_TICKERS) -> List[str]:
        return _top_keys(self._data.get("tickers", {}), n)

    def top_brokers(self, n: int = _TOP_N_BROKERS) -> List[str]:
        return _top_keys(self._data.get("brokers", {}), n)

    def top_sectors(self, n: int = _TOP_N_SECTORS) -> List[str]:
        return _top_keys(self._data.get("sectors", {}), n)

    def get_hint(self) -> str:
        """Return a one-paragraph hint block for injection into _analyze_query().

        Empty string if the user has no recorded preferences yet.
        """
        tickers = self.top_tickers(5)
        brokers = self.top_brokers(3)
        sectors = self.top_sectors(2)

        if not (tickers or brokers or sectors):
            return ""

        parts: List[str] = []
        if tickers:
            parts.append(f"Most-queried tickers: {', '.join(tickers)}")
        if brokers:
            parts.append(f"Most-queried brokers: {', '.join(brokers)}")
        if sectors:
            parts.append(f"Most-queried sectors: {', '.join(sectors)}")

        return (
            "=== User preference context (from prior sessions) ===\n"
            + "\n".join(parts)
            + "\n"
            "Use this to resolve vague references (e.g. 'my usual stock', 'the broker I follow'). "
            "Do NOT inject these as hard_filters unless the question explicitly refers to them.\n"
            "=== End user context ==="
        )

    def stats(self) -> dict:
        return {
            "tickers": len(self._data.get("tickers", {})),
            "brokers": len(self._data.get("brokers", {})),
            "sectors": len(self._data.get("sectors", {})),
            "recent_queries": len(self._data.get("recent_queries", [])),
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _top_keys(counter: Dict[str, int], n: int) -> List[str]:
    return [k for k, _ in sorted(counter.items(), key=lambda x: -x[1])[:n]]


# ── module-level singleton ────────────────────────────────────────────────────

_memory: Optional[UserMemory] = None


def get_user_memory(cache_dir: Optional[Path] = None) -> UserMemory:
    global _memory
    if _memory is None:
        _memory = UserMemory(cache_dir=cache_dir)
    return _memory
