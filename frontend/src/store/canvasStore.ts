import { create } from 'zustand';
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
  type Connection,
} from '@xyflow/react';
import {
  listCanvases,
  createCanvas,
  loadCanvas,
  saveCanvas,
  deleteCanvas,
} from '../api/client';
import type { CanvasMeta } from '../types';

interface CanvasStore {
  nodes: Node[];
  edges: Edge[];
  canvasId: string | null;
  canvasName: string;
  isDirty: boolean;
  savedCanvases: CanvasMeta[];

  // ReactFlow handlers
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  // Node manipulation
  addNode: (node: Node) => void;
  updateNodeData: (id: string, data: Partial<Node['data']>) => void;

  // Canvas lifecycle
  newCanvas: (name?: string) => Promise<void>;
  loadCanvas: (id: string) => Promise<void>;
  saveCanvas: () => Promise<void>;
  fetchSavedCanvases: () => Promise<void>;
  removeCanvas: (id: string) => Promise<void>;
}

export const useCanvasStore = create<CanvasStore>((set, get) => ({
  nodes: [],
  edges: [],
  canvasId: null,
  canvasName: 'Untitled Canvas',
  isDirty: false,
  savedCanvases: [],

  onNodesChange: (changes) =>
    set(s => ({
      nodes: applyNodeChanges(changes, s.nodes),
      isDirty: true,
    })),

  onEdgesChange: (changes) =>
    set(s => ({
      edges: applyEdgeChanges(changes, s.edges),
      isDirty: true,
    })),

  onConnect: (connection) =>
    set(s => ({
      edges: addEdge({ ...connection, animated: true }, s.edges),
      isDirty: true,
    })),

  addNode: (node) =>
    set(s => ({ nodes: [...s.nodes, node], isDirty: true })),

  updateNodeData: (id, data) =>
    set(s => ({
      nodes: s.nodes.map(n =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n
      ),
      isDirty: true,
    })),

  newCanvas: async (name = 'Untitled Canvas') => {
    const canvas = await createCanvas(name);
    set({
      canvasId: canvas.id,
      canvasName: canvas.name,
      nodes: [],
      edges: [],
      isDirty: false,
    });
    await get().fetchSavedCanvases();
  },

  loadCanvas: async (id) => {
    const detail = await loadCanvas(id);
    set({
      canvasId: detail.id,
      canvasName: detail.name,
      nodes: detail.nodes as Node[],
      edges: detail.edges as Edge[],
      isDirty: false,
    });
  },

  saveCanvas: async () => {
    const { canvasId, canvasName, nodes, edges } = get();
    if (!canvasId) return;
    await saveCanvas(canvasId, canvasName, nodes, edges);
    set({ isDirty: false });
  },

  fetchSavedCanvases: async () => {
    const canvases = await listCanvases();
    set({ savedCanvases: canvases });
  },

  removeCanvas: async (id) => {
    await deleteCanvas(id);
    set(s => ({
      savedCanvases: s.savedCanvases.filter(c => c.id !== id),
      // If active canvas was deleted, reset
      ...(s.canvasId === id
        ? { canvasId: null, nodes: [], edges: [], canvasName: 'Untitled Canvas' }
        : {}),
    }));
  },
}));
