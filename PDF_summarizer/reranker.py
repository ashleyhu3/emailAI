"""
Cascade cross-encoder reranker for the RAG pipeline.

Wraps sentence-transformers CrossEncoder with lazy model loading, a process-level
singleton, and a no-op fallback when the library is not installed.

Usage:
    from reranker import get_reranker

    reranker = get_reranker()
    if reranker.is_available():
        chunks = reranker.rerank(query, chunks, top_k=20)
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from database import PDFChunk

# Model choices (from fastest to best quality):
#   cross-encoder/ms-marco-MiniLM-L-6-v2   — fast, 6-layer,  ~22 MB
#   cross-encoder/ms-marco-MiniLM-L-12-v2  — balanced, 12-layer, ~33 MB
#   cross-encoder/ms-marco-electra-base     — highest quality, slower
DEFAULT_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# Max tokens passed to the cross-encoder per (query, passage) pair.
# Longer passages are truncated; 512 is a safe ceiling for MiniLM.
_MAX_LENGTH = 512
# Cap on how many characters of chunk text we send to the CE (beyond this the model
# truncates anyway, so trimming early avoids building huge Python strings).
_TEXT_CHAR_LIMIT = 1800


def _chunk_text(chunk: "PDFChunk") -> str:
    """Concatenate the summary and raw content for reranking."""
    summary = (chunk.verbalized_summary or "").strip()
    content = (chunk.raw_content or "").strip()
    if summary and content:
        combined = f"{summary}\n\n{content}"
    else:
        combined = summary or content
    return combined[:_TEXT_CHAR_LIMIT]


class CrossEncoderReranker:
    """Lazy-loading cross-encoder reranker.

    The model is downloaded from HuggingFace on first use (~22 MB for the
    default MiniLM-L-6-v2) and cached in ~/.cache/huggingface/hub.
    All subsequent calls in the same process reuse the loaded model.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """True if sentence-transformers is installed and the model can load."""
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder
        print(f"[reranker] loading {self.model_id} …")
        t0 = time.perf_counter()
        self._model = CrossEncoder(self.model_id, max_length=_MAX_LENGTH)
        print(f"[reranker] model loaded in {time.perf_counter() - t0:.1f}s")

    def rerank(
        self,
        query: str,
        chunks: List["PDFChunk"],
        top_k: int,
    ) -> List["PDFChunk"]:
        """Re-score chunks against query using the cross-encoder; return top-k.

        Scores are not stored on the chunk objects — the caller receives a new
        list ordered by descending CE score, truncated to top_k.
        If sentence-transformers is unavailable, returns the original list unchanged.
        """
        if not chunks:
            return chunks
        if not self.is_available():
            return chunks[:top_k]

        if self._model is None:
            self._load()

        t0 = time.perf_counter()
        pairs = [(query, _chunk_text(c)) for c in chunks]
        scores = self._model.predict(pairs)  # ndarray of floats
        elapsed = time.perf_counter() - t0

        ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        result = [c for _, c in ranked[:top_k]]

        top_score = float(ranked[0][0]) if ranked else 0.0
        print(
            f"[reranker] scored {len(chunks)} chunks in {elapsed:.2f}s "
            f"→ top-{top_k} kept (top score={top_score:.3f})"
        )
        return result


# ── module-level singleton ────────────────────────────────────────────────────

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_id: str = DEFAULT_MODEL) -> CrossEncoderReranker:
    """Return the process-level reranker singleton (lazy-initialised)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(model_id=model_id)
    return _reranker
