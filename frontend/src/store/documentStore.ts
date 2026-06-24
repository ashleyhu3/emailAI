import { create } from 'zustand';
import { listDocuments, uploadDocument, deleteDocument } from '../api/client';
import type { Document } from '../types';

interface DocumentStore {
  documents: Document[];
  loading: boolean;
  // Documents the RAG retrieved chunks from (the candidate pool sent to the model).
  fetchedIds: number[];
  // Documents actually cited in the answer (a subset of fetched).
  chosenIds: number[];
  fetchDocuments: () => Promise<void>;
  upload: (file: File) => Promise<void>;
  remove: (id: number) => Promise<void>;
  setHighlightedDocs: (fetched: number[], chosen: number[]) => void;
  clearHighlights: () => void;
}

export const useDocumentStore = create<DocumentStore>((set, get) => ({
  documents: [],
  loading: false,
  fetchedIds: [],
  chosenIds: [],

  fetchDocuments: async () => {
    set({ loading: true });
    try {
      const docs = await listDocuments();
      set({ documents: docs });
    } finally {
      set({ loading: false });
    }
  },

  upload: async (file: File) => {
    await uploadDocument(file);
    await get().fetchDocuments();
  },

  remove: async (id: number) => {
    await deleteDocument(id);
    set(s => ({ documents: s.documents.filter(d => d.id !== id) }));
  },

  setHighlightedDocs: (fetched: number[], chosen: number[]) =>
    set({ fetchedIds: fetched, chosenIds: chosen }),

  clearHighlights: () => set({ fetchedIds: [], chosenIds: [] }),
}));

// Expose to browser console in dev for testing without an API call:
// __setHighlightedDocs([1, 2, 3], [1])  // fetched, chosen
if (import.meta.env.DEV) {
  (window as unknown as Record<string, unknown>).__setHighlightedDocs =
    (fetched: number[], chosen: number[]) =>
      useDocumentStore.getState().setHighlightedDocs(fetched, chosen);
}
