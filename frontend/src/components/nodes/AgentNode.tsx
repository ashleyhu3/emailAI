import { useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { runAgent } from '../../api/client';
import { useCanvasStore } from '../../store/canvasStore';
import { useDocumentStore } from '../../store/documentStore';
import type { AgentNodeData } from '../../types';

export function AgentNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as AgentNodeData;
  const [goal, setGoal] = useState(nodeData.goal || '');
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const { addNode, onConnect, nodes } = useCanvasStore();
  const { documents } = useDocumentStore();

  const currentNode = nodes.find(n => n.id === id);
  const pos = currentNode?.position ?? { x: 0, y: 0 };

  async function handleRun() {
    if (!goal.trim()) return;
    setLoading(true);
    try {
      const result = await runAgent(
        goal,
        3,
        selectedDocIds.length > 0 ? selectedDocIds : undefined,
      );

      const stamp = Date.now();
      const colOffset = 340;
      const rowSpacing = 200;

      // Spawn a QueryNode + AnswerNode pair for each sub-query
      result.sub_queries.forEach((sq, i) => {
        const qId = `agent-q-${stamp}-${i}`;
        const aId = `agent-a-${stamp}-${i}`;
        const y = pos.y + i * rowSpacing;

        addNode({
          id: qId,
          type: 'queryNode',
          position: { x: pos.x + colOffset, y },
          data: { question: sq.question, documentIds: [], loading: false },
        });

        addNode({
          id: aId,
          type: 'answerNode',
          position: { x: pos.x + colOffset * 2, y },
          data: {
            question: sq.question,
            answer: sq.answer,
            chunksUsed: sq.chunks_used,
          },
        });

        onConnect({ source: id,  target: qId, sourceHandle: null, targetHandle: null });
        onConnect({ source: qId, target: aId, sourceHandle: null, targetHandle: null });
      });

      // Spawn a SynthesisNode below the sub-query rows
      const synthId = `agent-synth-${stamp}`;
      addNode({
        id: synthId,
        type: 'synthesisNode',
        position: {
          x: pos.x + colOffset,
          y: pos.y + result.sub_queries.length * rowSpacing + 20,
        },
        data: {
          goal: result.goal,
          synthesis: result.synthesis,
          subQueries: result.sub_queries,
          expanded: false,
        },
      });
      onConnect({ source: id, target: synthId, sourceHandle: null, targetHandle: null });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white border-2 border-purple-400 rounded-xl shadow-lg w-72 p-4 text-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      <div className="flex items-center gap-2 mb-3">
        <span className="text-purple-500 text-base">🤖</span>
        <span className="font-semibold text-gray-700">Research Agent</span>
      </div>

      <textarea
        className="w-full border border-gray-200 rounded-lg p-2 text-xs resize-none focus:outline-none focus:ring-2 focus:ring-purple-300 mb-2"
        rows={3}
        placeholder="Enter a high-level research goal…"
        value={goal}
        onChange={e => setGoal(e.target.value)}
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
        onClick={handleRun}
        disabled={loading || !goal.trim()}
        className="w-full bg-purple-500 hover:bg-purple-600 disabled:bg-gray-200 text-white text-xs font-semibold py-1.5 rounded-lg transition-colors"
      >
        {loading ? 'Researching…' : 'Run Agent'}
      </button>

      {loading && (
        <p className="text-xs text-gray-400 mt-2 text-center animate-pulse">
          Decomposing goal into sub-queries…
        </p>
      )}

      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}
