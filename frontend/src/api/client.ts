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
