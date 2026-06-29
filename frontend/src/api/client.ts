import axios, { AxiosError } from 'axios';
import type { AskRequest, AskResponse } from '../types';

const http = axios.create({ baseURL: '/api' });

http.interceptors.response.use(
  r => r,
  (err: AxiosError) => {
    if (err.response?.status === 404) {
      console.error(`[API 404] ${err.config?.method?.toUpperCase()} ${err.config?.url}`, err.response.data);
    }
    return Promise.reject(err);
  }
);

export const askQuestion = (req: AskRequest): Promise<AskResponse> =>
  http.post('/queries/ask', req).then(r => r.data);

export const fetchDocumentContent = (documentId: number): Promise<DocumentContent> =>
  http.get(`/documents/${documentId}/content`).then(r => r.data);

export interface DocumentContent {
  id: number;
  filename: string;
  broker: string | null;
  sender_company: string | null;
  written_date: string | null;
  rating: string | null;
  target_price: number | null;
  tickers: string[] | null;
  sector: string | null;
  report_type: string | null;
  dense_summary: string | null;
  pages: { page_number: number | null; content: string }[];
}
