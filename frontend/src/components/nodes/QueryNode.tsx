import { useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { askQuestion } from '../../api/client';
import { useCanvasStore } from '../../store/canvasStore';
import { useDocumentStore } from '../../store/documentStore';
import type { QueryNodeData } from '../../types';

export function QueryNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as QueryNodeData;
  const [question, setQuestion] = useState(nodeData.question || '');
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>(nodeData.documentIds || []);
  const [loading, setLoading] = useState(false);
  const { addNode, onConnect } = useCanvasStore();
  const { nodes } = useCanvasStore();
  const { documents } = useDocumentStore();

  const currentNode = nodes.find(n => n.id === id);
  const pos = currentNode?.position ?? { x: 0, y: 0 };

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const result = await askQuestion({
        question,
        top_k: 3,
        document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
      });

      const answerId = `answer-${Date.now()}`;
      addNode({
        id: answerId,
        type: 'answerNode',
        position: { x: pos.x + 340, y: pos.y },
        data: {
          question,
          answer: result.answer,
          chunksUsed: result.chunks_used,
        },
      });

      onConnect({ source: id, target: answerId, sourceHandle: null, targetHandle: null });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white border-2 border-blue-400 rounded-xl shadow-lg w-72 p-4 text-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      <div className="flex items-center gap-2 mb-3">
        <span className="text-blue-500 text-base">🔍</span>
        <span className="font-semibold text-gray-700">Query</span>
      </div>

      <textarea
        className="w-full border border-gray-200 rounded-lg p-2 text-xs resize-none focus:outline-none focus:ring-2 focus:ring-blue-300 mb-2"
        rows={3}
        placeholder="Ask a question about your documents…"
        value={question}
        onChange={e => setQuestion(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleAsk(); }}
      />

      {documents.length > 0 && (
        <select
          multiple
          className="w-full border border-gray-200 rounded-lg p-1 text-xs mb-2 max-h-20"
          value={selectedDocIds.map(String)}
          onChange={e => {
            const vals = Array.from(e.target.selectedOptions).map(o => Number(o.value));
            setSelectedDocIds(vals);
          }}
        >
          {documents.map(d => (
            <option key={d.id} value={d.id}>{d.filename}</option>
          ))}
        </select>
      )}

      <button
        onClick={handleAsk}
        disabled={loading || !question.trim()}
        className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 text-white text-xs font-semibold py-1.5 rounded-lg transition-colors"
      >
        {loading ? 'Asking…' : 'Ask (⌘↵)'}
      </button>

      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}
