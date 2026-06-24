# Backend (FastAPI)

REST API that wraps the `PDF_summarizer/` RAG pipeline and the agent/canvas features for the
React frontend.

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API docs: **http://localhost:8000/docs**

Requires the same environment as the pipeline (`GEMINI_API_KEY`, `PDF_SUMMARIZER_DB_URL`) — see
the repo-root README.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/documents` | List all ingested documents + metadata |
| `POST` | `/documents/upload` | Upload a PDF; runs the full ingestion + embedding pipeline |
| `DELETE` | `/documents/{id}` | Remove a document and its chunks |
| `POST` | `/queries/ask` | Ask a question → classification → handling → answer with citations |
| `POST` | `/queries/backfill` | Backfill embeddings for chunks missing a vector |
| `POST` | `/agent/run` | Run a research agent: decompose a goal → sub-queries → synthesized report |
| `GET/POST/GET/PUT/DELETE` | `/canvas` `/canvas/{id}` | CRUD for saved research canvases (ReactFlow state) |

### `POST /queries/ask` response shape

```jsonc
{
  "answer": "...",                 // generated answer with inline (document) citations
  "chunks_used": [ ... ],          // documents/pages actually cited (drives sidebar highlight)
  "inferred_filters": { ... },     // hard filters auto-extracted from the question
  "query_type": "rag",             // "rag" | "list_documents" | "clarify"
  "is_enumeration": false           // true for "list N things" → numbered, per-item sourcing
}
```

## Structure

| File | Role |
|------|------|
| `main.py` | FastAPI app, CORS, router wiring (`/documents`, `/queries`, `/agent`, `/canvas`) |
| `models.py` | Pydantic request/response models (`AskResponse`, `ChunkRef`, agent/canvas DTOs) |
| `dependencies.py` | Shared singletons (DB manager, RAG pipeline) injected into routes |
| `agent.py` | Research agent: goal decomposition + multi-step synthesis (Gemini 3.5 Flash) |
| `canvas_db.py` | Persistence for canvases and ReactFlow node/edge state |
| `routes/` | One router per feature area |

The heavy lifting (classification, retrieval, generation, citation verification) lives in
`PDF_summarizer/rag_gemini.py`; the backend is a thin async wrapper that runs it in a threadpool.
