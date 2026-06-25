"""
Embedding utilities: single-text and batch embedding via Gemini, plus a persistent
disk cache that avoids re-embedding identical boilerplate across ingestion runs.

Public API (imported by rag_gemini.py and any future embedding consumer):
    embed_text(text)          → List[float]
    embed_texts_batch(texts)  → List[List[float]]
    EmbeddingCache            — SHA-256-keyed pickle cache with LRU eviction
    _get_embedding_cache()    — process-level singleton cache
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIMS = 768  # must match Vector(768) in database.py


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


def embed_text(text: str) -> List[float]:
    """Embed a single text string. Returns [] if text is empty."""
    if not text.strip():
        return []
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMS),
    )
    return list(result.embeddings[0].values)


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts in one API call. Returns a parallel list of vectors;
    empty-string inputs get an empty list back.

    Each input is wrapped in its own ``types.Content`` so the model returns ONE embedding
    per input. This matters for gemini-embedding-2: a bare list of strings is treated as a
    single aggregated input, whereas a list of Content objects yields per-input embeddings.
    """
    if not texts:
        return []
    client = _get_client()
    indexed = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed:
        return [[] for _ in texts]
    indices, non_empty = zip(*indexed)

    contents = [
        types.Content(parts=[types.Part.from_text(text=t)]) for t in non_empty
    ]
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMS),
    )
    embs = result.embeddings or []
    if len(embs) != len(non_empty):
        raise RuntimeError(
            f"{EMBEDDING_MODEL} returned {len(embs)} embeddings for {len(non_empty)} "
            f"inputs — expected one per input."
        )

    out: List[List[float]] = [[] for _ in texts]
    for rank, original_idx in enumerate(indices):
        out[original_idx] = list(embs[rank].values)
    return out


class EmbeddingCache:
    """
    Persistent disk cache for embedding vectors, keyed by content SHA-256.

    Prevents re-embedding identical boilerplate (legal disclaimers, analyst certifications,
    standard headers) that appears in every report from the same broker. Expected savings:
    30-50% of embedding API calls for a corpus with shared boilerplate.

    Thread/process-safe: uses filelock for atomic writes; each worker keeps an in-memory
    read-through layer so lookups cost zero I/O after the initial load.
    """

    _MAX_ENTRIES = 50_000
    _FLUSH_INTERVAL = 20

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        self._model_name = model_name
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".cache" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_slug = hashlib.md5(model_name.encode()).hexdigest()[:8]
        self._path = cache_dir / f"emb_{model_slug}.pkl"
        self._lock_path = str(cache_dir / f"emb_{model_slug}.lock")

        self._memory: dict = {}
        self._dirty = 0
        self._load()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()

    def _load(self) -> None:
        try:
            from filelock import FileLock
            with FileLock(self._lock_path, timeout=5):
                if self._path.exists():
                    with open(self._path, "rb") as f:
                        self._memory = pickle.load(f)
            if self._memory:
                print(f"[EmbeddingCache] {len(self._memory):,} entries loaded")
        except Exception as exc:
            print(f"[EmbeddingCache] Load failed ({exc}) — starting fresh")
            self._memory = {}

    def _write(self) -> None:
        """Atomic write: merge in-memory entries with on-disk state then swap."""
        try:
            from filelock import FileLock
            with FileLock(self._lock_path, timeout=10):
                on_disk: dict = {}
                if self._path.exists():
                    try:
                        with open(self._path, "rb") as f:
                            on_disk = pickle.load(f)
                    except Exception:
                        pass
                on_disk.update(self._memory)
                if len(on_disk) > self._MAX_ENTRIES:
                    overflow = len(on_disk) - self._MAX_ENTRIES
                    for k in list(on_disk.keys())[:overflow]:
                        del on_disk[k]
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(on_disk, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(self._path)
            self._dirty = 0
        except Exception as exc:
            print(f"[EmbeddingCache] Flush failed ({exc})")

    def get(self, text: str) -> Optional[List[float]]:
        return self._memory.get(self._key(text))

    def put(self, text: str, embedding: List[float]) -> None:
        key = self._key(text)
        if key not in self._memory:
            self._memory[key] = embedding
            self._dirty += 1
            if self._dirty >= self._FLUSH_INTERVAL:
                self._write()

    def flush(self) -> int:
        """Force-write all dirty entries. Returns number of entries flushed."""
        count = self._dirty
        if count:
            self._write()
        return count

    def stats(self) -> dict:
        return {
            "entries": len(self._memory),
            "path": str(self._path),
            "dirty": self._dirty,
        }


_embedding_cache: Optional[EmbeddingCache] = None


def _get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(model_name=EMBEDDING_MODEL)
    return _embedding_cache
