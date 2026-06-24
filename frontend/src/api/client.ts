import axios, { AxiosError } from 'axios';
import type {
  Document,
  AskRequest,
  AskResponse,
  AgentResult,
  CanvasMeta,
  CanvasDetail,
} from '../types';

const http = axios.create({ baseURL: '/api' });

// Log 404s clearly to help diagnose routing issues
http.interceptors.response.use(
  r => r,
  (err: AxiosError) => {
    if (err.response?.status === 404) {
      console.error(`[API 404] ${err.config?.method?.toUpperCase()} ${err.config?.url}`, err.response.data);
    }
    return Promise.reject(err);
  }
);

// ── Documents ──────────────────────────────────────────────────────────────

export const listDocuments = (): Promise<Document[]> =>
  http.get('/documents').then(r => r.data);

export const uploadDocument = (file: File): Promise<{ status: string; document_id?: number }> => {
  const form = new FormData();
  form.append('file', file);
  return http.post('/documents/upload', form).then(r => r.data);
};

export const deleteDocument = (id: number): Promise<void> =>
  http.delete(`/documents/${id}`).then(r => r.data);

// ── Queries ────────────────────────────────────────────────────────────────

export const askQuestion = (req: AskRequest): Promise<AskResponse> =>
  http.post('/queries/ask', req).then(r => r.data);

export const backfillEmbeddings = (): Promise<{ embedded_count: number }> =>
  http.post('/queries/backfill').then(r => r.data);

// ── Agent ──────────────────────────────────────────────────────────────────

export const runAgent = (
  goal: string,
  top_k = 3,
  document_ids?: number[],
): Promise<AgentResult> =>
  http.post('/agent/run', { goal, top_k, document_ids }).then(r => r.data);

// ── Canvas ─────────────────────────────────────────────────────────────────

export const listCanvases = (): Promise<CanvasMeta[]> =>
  http.get('/canvas').then(r => r.data);

export const createCanvas = (name: string): Promise<CanvasMeta> =>
  http.post('/canvas', { name }).then(r => r.data);

export const loadCanvas = (id: string): Promise<CanvasDetail> => {
  if (!id) return Promise.reject(new Error('loadCanvas called with empty id'));
  return http.get(`/canvas/${id}`).then(r => r.data);
};

export const saveCanvas = (
  id: string,
  name: string,
  nodes: unknown[],
  edges: unknown[],
): Promise<CanvasMeta> => {
  if (!id) return Promise.reject(new Error('saveCanvas called with empty id'));
  return http.put(`/canvas/${id}`, { name, nodes, edges }).then(r => r.data);
};

export const deleteCanvas = (id: string): Promise<void> => {
  if (!id) return Promise.reject(new Error('deleteCanvas called with empty id'));
  return http.delete(`/canvas/${id}`).then(r => r.data);
};
