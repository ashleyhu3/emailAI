import { useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { SynthesisNodeData } from '../../types';

export function SynthesisNode({ data }: NodeProps) {
  const nodeData = data as unknown as SynthesisNodeData;
  const [expanded, setExpanded] = useState(nodeData.expanded ?? false);

  return (
    <div className="bg-white border-2 border-indigo-400 rounded-xl shadow-lg w-96 p-4 text-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-indigo-500 text-base">📊</span>
          <span className="font-semibold text-gray-700">Synthesis Report</span>
        </div>
        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      <p className="text-xs text-gray-400 italic mb-3 truncate">
        Goal: {nodeData.goal}
      </p>

      <div
        className={`text-xs text-gray-700 leading-relaxed whitespace-pre-wrap ${
          expanded ? '' : 'max-h-40 overflow-hidden'
        }`}
      >
        {nodeData.synthesis}
      </div>

      {!expanded && (nodeData.synthesis?.length ?? 0) > 300 && (
        <button
          onClick={() => setExpanded(true)}
          className="text-xs text-indigo-400 hover:text-indigo-600 mt-1"
        >
          Show full report
        </button>
      )}

      {expanded && nodeData.subQueries && nodeData.subQueries.length > 0 && (
        <div className="mt-3 pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-400 font-medium mb-2">
            Sub-queries ({nodeData.subQueries.length})
          </p>
          <div className="space-y-2">
            {nodeData.subQueries.map((sq, i) => (
              <div key={i} className="bg-gray-50 rounded-lg p-2">
                <p className="text-xs font-medium text-gray-600 mb-1">{sq.question}</p>
                <p className="text-xs text-gray-500 line-clamp-2">{sq.answer}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}
