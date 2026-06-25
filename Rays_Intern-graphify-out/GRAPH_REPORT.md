# Graph Report - /Users/ashleyhu/Desktop/rays-research/Rays_Intern  (2026-06-24)

## Corpus Check
- 80 files · ~53,107 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 426 nodes · 588 edges · 19 communities (13 shown, 6 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Python Dependencies|Python Dependencies]]
- [[_COMMUNITY_Email Fetching & Parsing|Email Fetching & Parsing]]
- [[_COMMUNITY_Document Preprocessing|Document Preprocessing]]
- [[_COMMUNITY_Metadata Field Extraction|Metadata Field Extraction]]
- [[_COMMUNITY_Embedding & Cache|Embedding & Cache]]
- [[_COMMUNITY_Extraction Agents 1-3|Extraction Agents 1-3]]
- [[_COMMUNITY_Query Routing Agents 4-6|Query Routing Agents 4-6]]
- [[_COMMUNITY_User Memory & Personalization|User Memory & Personalization]]
- [[_COMMUNITY_PDF Link Extraction|PDF Link Extraction]]
- [[_COMMUNITY_Cross-Encoder Reranking|Cross-Encoder Reranking]]
- [[_COMMUNITY_Hot Folder Ingest|Hot Folder Ingest]]
- [[_COMMUNITY_Example Scripts|Example Scripts]]
- [[_COMMUNITY_Document Removal Script|Document Removal Script]]
- [[_COMMUNITY_Ingest Package Init|Ingest Package Init]]
- [[_COMMUNITY_Worker Startup|Worker Startup]]
- [[_COMMUNITY_Docling Sample|Docling Sample]]
- [[_COMMUNITY_Base Classes|Base Classes]]
- [[_COMMUNITY_Pydantic Base Models|Pydantic Base Models]]

## God Nodes (most connected - your core abstractions)
1. `_process_email()` - 21 edges
2. `EmailPayload` - 15 edges
3. `FinancialReportMetadata` - 12 edges
4. `UserMemory` - 12 edges
5. `EmbeddingCache` - 11 edges
6. `parse_eml_bytes()` - 11 edges
7. `fetch_broker_emails()` - 10 edges
8. `extract_fields_deterministically()` - 10 edges
9. `extract_metadata()` - 9 edges
10. `_message_to_payload()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `IngestTriggerRequest` --uses--> `EmailPayload`  [INFERRED]
  Rays_Intern/backend/routes/ingest.py → Rays_Intern/PDF_summarizer/ingest/email_fetcher.py
- `Path` --uses--> `EmailPayload`  [INFERRED]
  Rays_Intern/backend/hot_folder.py → Rays_Intern/PDF_summarizer/ingest/email_fetcher.py
- `Any` --uses--> `EmailPayload`  [INFERRED]
  Rays_Intern/backend/routes/ingest.py → Rays_Intern/PDF_summarizer/ingest/email_fetcher.py
- `BrokerContextCache` --uses--> `EmailPayload`  [INFERRED]
  Rays_Intern/PDF_summarizer/ingest/worker.py → Rays_Intern/PDF_summarizer/ingest/email_fetcher.py
- `DatabaseManager` --uses--> `EmailPayload`  [INFERRED]
  Rays_Intern/PDF_summarizer/ingest/worker.py → Rays_Intern/PDF_summarizer/ingest/email_fetcher.py

## Import Cycles
- 1-file cycle: `Rays_Intern/PDF_summarizer/ingest/email_fetcher.py -> Rays_Intern/PDF_summarizer/ingest/email_fetcher.py`

## Hyperedges (group relationships)
- **RAG Ingestion Pipeline Flow (Parse → Verbalize → Store → Embed)** — pdf_summarizer_readme_docling_parser, pdf_summarizer_readme_gemini_verbalization, pdf_summarizer_readme_pgvector_storage, pdf_summarizer_readme_gemini_embedding, pdf_summarizer_readme_three_chunk_hierarchy [EXTRACTED 1.00]
- **Backend Core Modules (main, models, dependencies, agent, canvas_db, routes)** — backend_readme_main_module, backend_readme_models_module, backend_readme_dependencies_module, backend_readme_agent_module, backend_readme_canvas_db_module, backend_readme_routes_dir [EXTRACTED 1.00]
- **Frontend State + UI Layer (ChatView, Sidebar, Toolbar, Zustand stores, API client)** — frontend_readme_chatview, frontend_readme_sidebar, frontend_readme_toolbar, frontend_readme_zustand_stores, frontend_readme_api_client [EXTRACTED 1.00]

## Communities (19 total, 6 thin omitted)

### Community 0 - "Python Dependencies"
Cohesion: 0.01
Nodes (153): accelerate, annotated-doc, annotated-types, antlr4-python3-runtime, anyio, attrs, beautifulsoup4, celery (+145 more)

### Community 1 - "Email Fetching & Parsing"
Cohesion: 0.06
Nodes (58): datetime, _decode_str(), EmailPayload, _extract_parts(), fetch_broker_emails(), load_config(), _message_hash(), _parse_date() (+50 more)

### Community 2 - "Document Preprocessing"
Cohesion: 0.08
Nodes (39): clean_html_to_aoim(), _docling_converter(), email_html_to_rag_chunks(), full_pdf_to_rag_chunks(), Phase 2: Dual-Stream Local Preprocessing — 0 token cost.  Stream A (HTML emails), Lazy-import Docling (heavy dependency) only when needed., Parse only `page_range` pages from `pdf_bytes` with Docling.      Returns:, Convert a broker digest email's HTML into RAG chunks.      Used when an email ha (+31 more)

### Community 3 - "Metadata Field Extraction"
Cohesion: 0.09
Nodes (26): date, _detect_action(), _detect_rating(), _detect_target_price(), _extract_company_from_subject(), extract_fields_deterministically(), extract_relevant_aoim_sections(), _extract_tickers() (+18 more)

### Community 4 - "Embedding & Cache"
Cohesion: 0.13
Nodes (13): embed_text(), embed_texts_batch(), EmbeddingCache, _get_client(), _get_embedding_cache(), Embedding utilities: single-text and batch embedding via Gemini, plus a persiste, Atomic write: merge in-memory entries with on-disk state then swap., Force-write all dirty entries. Returns number of entries flushed. (+5 more)

### Community 5 - "Extraction Agents 1-3"
Cohesion: 0.16
Nodes (18): Re-export shim — kept for backward compatibility.  Agents 1–3 (extraction) live, _client(), EpsPeData, extract_metadata(), FinancialReportMetadata, Ingestion agents (Agents 1–3) — PDF content extraction team.    Agent 1 — Spatia, Agent 2: Extract structured broker report metadata from AOIM text., Flash self-correction: feed the validation error back to fix it (~40% of failure (+10 more)

### Community 6 - "Query Routing Agents 4-6"
Cohesion: 0.18
Nodes (19): BaseModel, _client(), GatewayIntentParams, _get_cached_routing(), QueryRoutingParams, RAG query agents (Agents 4–6) — interactive query team.    Agent 4 — Gateway Int, Agent 4: Classify user intent and extract active document context from history., Agent 5a: Generate SQL filter parameters from a user query. (+11 more)

### Community 7 - "User Memory & Personalization"
Cohesion: 0.18
Nodes (8): get_user_memory(), User preference memory for the RAG pipeline.  Tracks which tickers, brokers, and, Return a one-paragraph hint block for injection into _analyze_query()., Lightweight, persistent preference tracker.      Counts how often each ticker /, Update frequency counters after a successful answered query., _top_keys(), UserMemory, Path

### Community 8 - "PDF Link Extraction"
Cohesion: 0.18
Nodes (15): _browser_cookiejar_for_url(), _cookies_for_url(), extract_links_from_html(), extract_pdfs_from_email(), fetch_pdf_from_url(), _is_pdf_response(), _load_portal_cookies(), PDF link extractor for notification emails.  Many broker notification emails con (+7 more)

### Community 9 - "Cross-Encoder Reranking"
Cohesion: 0.19
Nodes (9): _chunk_text(), CrossEncoderReranker, get_reranker(), Cascade cross-encoder reranker for the RAG pipeline.  Wraps sentence-transformer, Return the process-level reranker singleton (lazy-initialised)., Concatenate the summary and raw content for reranking., Lazy-loading cross-encoder reranker.      The model is downloaded from HuggingFa, True if sentence-transformers is installed and the model can load. (+1 more)

### Community 10 - "Hot Folder Ingest"
Cohesion: 0.33
Nodes (8): _already_seen(), _infer_broker(), _ingest_pdf_file(), _make_handler(), Hot-folder watcher: auto-ingest PDFs that land in ~/Downloads (or INGEST_HOT_FOL, Start the background watchdog thread. Safe to call multiple times — only     sta, start_hot_folder_watcher(), Path

### Community 11 - "Example Scripts"
Cohesion: 0.25
Nodes (7): example_directory(), example_query_database(), example_single_pdf(), Example usage of the Docling + Gemini verbalization pipeline., Process a single PDF from research_pdfs., Process all PDFs in research_pdfs., Query the database for processed documents and chunks.

## Knowledge Gaps
- **158 isolated node(s):** `Client`, `Path`, `date`, `Response`, `start_worker.sh script` (+153 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_process_email()` connect `Document Preprocessing` to `PDF Link Extraction`, `Hot Folder Ingest`, `Metadata Field Extraction`, `Extraction Agents 1-3`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `datetime` connect `Email Fetching & Parsing` to `Document Preprocessing`, `Metadata Field Extraction`, `Extraction Agents 1-3`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `EmailPayload` connect `Email Fetching & Parsing` to `Hot Folder Ingest`, `Document Preprocessing`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `EmailPayload` (e.g. with `IngestTriggerRequest` and `Path`) actually correct?**
  _`EmailPayload` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `FinancialReportMetadata` (e.g. with `BrokerContextCache` and `DatabaseManager`) actually correct?**
  _`FinancialReportMetadata` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Client`, `Path`, `Embedding utilities: single-text and batch embedding via Gemini, plus a persiste` to the rest of the system?**
  _254 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Python Dependencies` be split into smaller, more focused modules?**
  _Cohesion score 0.012987012987012988 - nodes in this community are weakly interconnected._