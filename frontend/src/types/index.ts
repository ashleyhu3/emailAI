export interface Document {
  id: number;
  filename: string;
  total_pages: number;
  file_size_bytes: number;
  sender_name: string | null;
  sender_company: string | null;
  sent_date: string | null;
  written_date: string | null;
  tickers: string[] | null;
  report_type: string | null;
  sector: string | null;
  asset_class: string | null;
  coverage_period_from: string | null;
  coverage_period_to: string | null;
  uploaded_at: string;
  processed_at: string | null;
}

export interface ChunkRef {
  chunk_id: string;
  document_id: number;
  page_number: number | null;
  metadata: Record<string, unknown>;
}

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskRequest {
  question: string;
  history?: HistoryMessage[];
  top_k?: number;
  document_ids?: number[];
  filenames?: string[];
  page_min?: number;
  page_max?: number;
  sender_names?: string[];
  sender_companies?: string[];
  written_date_from?: string;
  written_date_to?: string;
  tickers?: string[];
  report_type?: string;
  sector?: string;
  asset_class?: string;
  coverage_period_from?: string;
  coverage_period_to?: string;
}

export interface AskResponse {
  answer: string;
  chunks_used: ChunkRef[];
  inferred_filters?: Record<string, unknown>;
  query_type?: 'rag' | 'list_documents' | 'clarify' | 'chart';
  is_enumeration?: boolean;
  chart_html?: string;
}

export interface SubQueryResult {
  question: string;
  answer: string;
  chunks_used: ChunkRef[];
}

export interface AgentResult {
  goal: string;
  sub_queries: SubQueryResult[];
  synthesis: string;
}

export interface CanvasMeta {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface CanvasDetail extends CanvasMeta {
  nodes: unknown[];
  edges: unknown[];
}

// ── Node data shapes ────────────────────────────────────────────────────────

export interface QueryNodeData {
  question: string;
  documentIds: number[];
  pageMin?: number;
  pageMax?: number;
  loading?: boolean;
}

export interface AnswerNodeData {
  question: string;
  answer: string;
  chunksUsed: ChunkRef[];
}

export interface NoteNodeData {
  text: string;
}

export interface AgentNodeData {
  goal: string;
  topK: number;
  loading?: boolean;
}

export interface SynthesisNodeData {
  goal: string;
  synthesis: string;
  subQueries: SubQueryResult[];
  expanded?: boolean;
}
