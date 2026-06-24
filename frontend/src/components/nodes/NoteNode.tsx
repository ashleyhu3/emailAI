import { useState } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { useCanvasStore } from '../../store/canvasStore';
import type { NoteNodeData } from '../../types';

export function NoteNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as NoteNodeData;
  const [text, setText] = useState(nodeData.text || '');
  const { updateNodeData } = useCanvasStore();

  function handleBlur() {
    updateNodeData(id, { text });
  }

  return (
    <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl shadow-md w-56 p-3 text-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3" />

      <div className="flex items-center gap-1 mb-2">
        <span className="text-yellow-500 text-base">📝</span>
        <span className="font-semibold text-gray-600 text-xs">Note</span>
      </div>

      <textarea
        className="w-full bg-transparent text-xs text-gray-700 resize-none focus:outline-none leading-relaxed"
        rows={5}
        placeholder="Add a note…"
        value={text}
        onChange={e => setText(e.target.value)}
        onBlur={handleBlur}
      />

      <Handle type="source" position={Position.Right} className="w-3 h-3" />
    </div>
  );
}
