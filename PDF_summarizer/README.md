# PDF Summarizer Pipeline

Docling + Gemini pipeline for image-heavy financial PDFs, plus the RAG query engine. Charts and tables are verbalized to text so they can be embedded and searched alongside the prose.

## Workflow

1. **Parse** – Docling extracts page layout and page images
2. **Verbalize** – Gemini 2.5 Flash describes every chart/graph/table on each page
3. **Store** – Three chunk levels per document (document summary → section summaries → per-page content) in Postgres
4. **Embed** – `gemini-embedding-2` (768 dims) over the verbalized text; pgvector cosine search
5. **Query** – classify → handle → answer with citations (see *Query handling* below)

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"
```

**Note:** Postgres + pgvector is required (SQLite is not supported for vector search).

```sql
CREATE DATABASE finance_rag;
\c finance_rag
CREATE EXTENSION IF NOT EXISTS vector;
```

## Usage

### 1. Process PDFs (Parse + Verbalize + Store)

```bash
# Single file or a directory (directory ingestion is parallelized)
python pipeline.py research_pdfs/ --db-url "$PDF_SUMMARIZER_DB_URL" --max-workers 8
```

### 2. Backfill embeddings

```bash
# Embeds only chunks that don't have a vector yet
python rag_gemini.py backfill --db-url "$PDF_SUMMARIZER_DB_URL"

# Re-embed EVERYTHING (use when switching EMBEDDING_MODEL). --reset clears existing
# vectors first; --sleep throttles between batches to avoid rate limits.
python rag_gemini.py backfill --reset --sleep 2 --db-url "$PDF_SUMMARIZER_DB_URL"
```

### 3. Ask questions (RAG)

```bash
python rag_gemini.py ask "What were the main revenue drivers?" --db-url "$PDF_SUMMARIZER_DB_URL"

# With filters
python rag_gemini.py ask "Summarize risk factors" --filename report.pdf --page-min 5 --page-max 20
```

### 4. Backfill / re-extract metadata (no re-ingestion)

```bash
python pipeline.py --backfill-metadata --db-url "$PDF_SUMMARIZER_DB_URL"          # fill missing
python pipeline.py --backfill-metadata --force --db-url "$PDF_SUMMARIZER_DB_URL"  # re-extract all
```

## Query handling

Every question flows through three stages — **classification → handling → output** — in a loop.

**1. Classification** (`_analyze_query`, one Gemini 3.5 Flash call) returns:
- **hard filters** (company, ticker, date, sector, coverage period). Explicit periods such as
  "Q1 2025" are parsed deterministically into a coverage-period range.
- a self-contained **`standalone_query`** for embedding (resolves references, never invents topics).
- **`query_type`**: `rag`, `list_documents`, or `clarify`.
- **`is_followup`** and **`is_underspecified`** flags.

**2. Handling**
- **Underspecified** (filter but no topic, e.g. "what did SinoPac say in Q1 2025?") → ask the user to narrow.
- **`list_documents`** → deterministic document inventory, no generation call.
- **`rag`** → embed → pgvector search with a **tiered similarity threshold** (relaxed as more metadata
  filters constrain the pool) → expand each hit with parent + sibling pages → generate with Gemini 3.5 Flash.
  **Enumerations** ("10 trends…") retrieve for *breadth* across documents (per-document cap); **follow-ups**
  also retrieve, with conversation history interleaved.

**3. Output** — citations are verified at the **document level**: a cited document that wasn't retrieved is
stripped (fabrication); a correct document with an unverifiable page degrades to a document-level citation;
any unsourced list item is dropped.

## Key modules

| File | Role |
|------|------|
| `pipeline.py` | Ingestion CLI: parse → verbalize → store → extract metadata |
| `pdf_processor.py` | Docling extraction + Gemini verbalization/summarization |
| `database.py` | SQLAlchemy ORM + pgvector search (`DatabaseManager`) |
| `rag_gemini.py` | Embedding backfill + the full query pipeline (`GeminiRAGPipeline`) |
| `utils.py` | SHA256 hashing for duplicate detection |

## Schema

- **pdf_documents** – file-level metadata (filename, hash, page count, sender/company, dates,
  tickers, report_type, sector, asset_class, coverage_period_from/to)
- **pdf_chunks** – one row per chunk:
  - `id` – UUID primary key
  - `embedding` – `Vector(768)` from `verbalized_summary` (search against this)
  - `raw_content` – original Docling markdown (used for answering)
  - `verbalized_summary` – Gemini chart/table description (embedded & searched)
  - `metadata_` – JSONB: `level`, `page_number`, `section_id`, parent/sibling links

## Environment

- `GEMINI_API_KEY` – required for verbalization, embedding, and generation
- `PDF_SUMMARIZER_DB_URL` – Postgres connection string (default for the CLI `--db-url`)

## Models

- **Generation / classification / agent:** `models/gemini-3.5-flash` (`GENERATION_MODEL`)
- **Verbalization / summarization (ingestion):** `models/gemini-2.5-flash` (`VERBALIZE_MODEL`, `TEXT_SUMMARY_MODEL`)
- **Embeddings:** `models/gemini-embedding-2` at 768 dims (`EMBEDDING_MODEL`, must match `Vector(768)`)
