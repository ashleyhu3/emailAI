# Frontend (React + Vite)

Visual research client for the Financial PDF Research Platform: a node-based canvas (ReactFlow)
plus a chat view, talking to the FastAPI backend.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` → `http://localhost:8000` (see `vite.config.ts`), so the FastAPI
backend must be running. Other scripts: `npm run build`, `npm run preview`, `npm run lint`.

## What's here

- **Canvas** — right-click to add Query, Note, and Research Agent nodes; drag edges to link findings; canvases auto-save.
- **Chat** — ask questions against the document set. The UI reflects the backend's query handling:
  - **Clarification** — a broad, underspecified question ("what did SinoPac say in Q1 2025?") returns a prompt to narrow instead of a guessed answer.
  - **Document inventory** vs **RAG** answers, and **numbered, per-item-sourced lists** for "list N things" queries.
  - **Source pills** are rendered beneath each block of the answer (parsed from the answer's citations), and the **cited documents are highlighted in the sidebar** (in citation order).
- **Sidebar** — upload PDFs, filter the document set (company, author, date, ticker, report type, asset class, sector), and see which documents an answer cited.

## Structure

| Path | Role |
|------|------|
| `src/api/client.ts` | Axios client (`baseURL: '/api'`) — ask, upload, documents, agent, canvas |
| `src/components/ChatView.tsx` | Chat UI: send questions, render answers + source pills, drive sidebar highlight |
| `src/components/Sidebar.tsx` | Documents (with cited-highlight + count), Filters, Canvases, Chats |
| `src/components/Toolbar.tsx` | Canvas toolbar (new/save/load canvases) |
| `src/store/` | Zustand stores: `chatStore`, `documentStore`, `filterStore`, `canvasStore` |
| `src/lib/citations.ts` | Parse in-text citations (`(file.pdf, p.3)` / document-level `(file.pdf)`) into source pills |
| `src/types/index.ts` | Shared API types (`AskResponse`, `ChunkRef`, etc.) |
| `src/hooks/useAutoSave.ts` | Debounced canvas auto-save |

## Notes for editing

- Citation parsing in `lib/citations.ts` mirrors the backend regex in
  `PDF_summarizer/rag_gemini.py` (`_CITATION_RE`/`_expand_pages`) — keep them in sync so the pills
  match what the backend verified.
- `query_type` (`rag` | `list_documents` | `clarify`) and `is_enumeration` come from the backend
  and control how `ChatView` renders the answer (auto-filter chips, numbering, etc.).

---

<details>
<summary>Vite + ESLint template notes</summary>

This app was scaffolded with the React + TypeScript + Vite template. For type-aware lint rules,
see the [Vite React ESLint docs](https://react.dev/learn/react-compiler/installation).
</details>
