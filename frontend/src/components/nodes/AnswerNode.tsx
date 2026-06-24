import { useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { AnswerNodeData } from '../../types';
import { useDocumentStore } from '../../store/documentStore';

export function AnswerNode({ data }: NodeProps) {
  const nodeData = data as unknown as AnswerNodeData;
  const [expanded, setExpanded] = useState(false);
  const { documents } = useDocumentStore();
  const docMap = Object.fromEntries(documents.map(d => [d.id, d.filename]));

  return (
    <div className="bg-white border-2 border-green-400 rounded-xl shadow-lg w-80 p-4 text-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-green-500 text-base">💡</span>
          <span className="font-semibold text-gray-700">Answer</span>
        </div>
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {nodeData.question && (
        <p className="text-xs text-gray-400 italic mb-2 truncate">
          Q: {nodeData.question}
        </p>
      )}

      <div
        className={`text-xs text-gray-700 leading-relaxed whitespace-pre-wrap ${
          expanded ? '' : 'max-h-32 overflow-hidden'
        }`}
      >
        {nodeData.answer}
      </div>

      {!expanded && nodeData.answer?.length > 200 && (
        <button
          onClick={() => setExpanded(true)}
          className="text-xs text-blue-400 hover:text-blue-600 mt-1"
        >
          Show more
        </button>
      )}

      {nodeData.chunksUsed && nodeData.chunksUsed.length > 0 && (
        <div className="mt-3 pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-400 font-medium mb-1">Sources</p>
          <div className="flex flex-wrap gap-1">
            {nodeData.chunksUsed.map((c, i) => (
              <span
                key={i}
                className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded"
                title={`Chunk ${c.chunk_id}`}
              >
                {docMap[c.document_id] ?? `doc ${c.document_id}`}
                {c.page_number != null ? ` p.${c.page_number}` : ''}
              </span>
            ))}
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}
