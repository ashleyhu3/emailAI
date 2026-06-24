import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChunkRef } from '../types';

export interface FilterContext {
  senderNames?: string[];
  senderCompanies?: string[];
  writtenDateFrom?: string;
  writtenDateTo?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  chunks?: ChunkRef[];
  filterContext?: FilterContext;
  inferredFilters?: Record<string, unknown>;
  queryType?: 'rag' | 'list_documents' | 'clarify';
  isEnumeration?: boolean;
}

export interface ChatSession {
  id: string;
  name: string;
  messages: ChatMessage[];
  createdAt: string;
}

interface ChatStore {
  sessions: ChatSession[];
  activeSessionId: string | null;
  activeSession: () => ChatSession | null;
  newSession: (name?: string) => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, name: string) => void;
  setActiveSession: (id: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      activeSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find(s => s.id === activeSessionId) ?? null;
      },

      newSession: (name = 'New Chat') => {
        const id = `chat-${Date.now()}`;
        const session: ChatSession = {
          id,
          name,
          messages: [],
          createdAt: new Date().toISOString(),
        };
        set(s => ({ sessions: [session, ...s.sessions], activeSessionId: id }));
        return id;
      },

      deleteSession: (id) => {
        set(s => {
          const sessions = s.sessions.filter(sess => sess.id !== id);
          const activeSessionId =
            s.activeSessionId === id ? (sessions[0]?.id ?? null) : s.activeSessionId;
          return { sessions, activeSessionId };
        });
      },

      renameSession: (id, name) => {
        set(s => ({
          sessions: s.sessions.map(sess => sess.id === id ? { ...sess, name } : sess),
        }));
      },

      setActiveSession: (id) => set({ activeSessionId: id }),

      addMessage: (sessionId, message) => {
        set(s => ({
          sessions: s.sessions.map(sess =>
            sess.id === sessionId
              ? { ...sess, messages: [...sess.messages, message] }
              : sess
          ),
        }));
      },
    }),
    { name: 'chat-sessions' }
  )
);
