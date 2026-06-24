import { useCallback, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
} from '@xyflow/react';
import { useCanvasStore } from './store/canvasStore';
import { Sidebar } from './components/Sidebar';
import { Toolbar } from './components/Toolbar';
import { ChatView } from './components/ChatView';
import { QueryNode } from './components/nodes/QueryNode';
import { AnswerNode } from './components/nodes/AnswerNode';
import { NoteNode } from './components/nodes/NoteNode';
import { AgentNode } from './components/nodes/AgentNode';
import { SynthesisNode } from './components/nodes/SynthesisNode';
import { useAutoSave } from './hooks/useAutoSave';

const nodeTypes: NodeTypes = {
  queryNode: QueryNode,
  answerNode: AnswerNode,
  noteNode: NoteNode,
  agentNode: AgentNode,
  synthesisNode: SynthesisNode,
};

const CONTEXT_MENU_ITEMS = [
  { label: '🔍 Add Query Node',      type: 'queryNode',   data: { question: '', documentIds: [] } as Record<string, unknown> },
  { label: '📝 Add Note',            type: 'noteNode',    data: { text: '' } as Record<string, unknown> },
  { label: '🤖 Add Research Agent',  type: 'agentNode',   data: { goal: '', topK: 3 } as Record<string, unknown> },
];

export default function App() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode } = useCanvasStore();
  const [view, setView] = useState<'canvas' | 'chat'>('canvas');

  useAutoSave();

  const menuPos = useRef<{ flowX: number; flowY: number }>({ flowX: 200, flowY: 200 });
  const menuEl = useRef<HTMLDivElement>(null);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    menuPos.current = { flowX: e.clientX - rect.left, flowY: e.clientY - rect.top };
    if (menuEl.current) {
      menuEl.current.style.left = `${e.clientX}px`;
      menuEl.current.style.top = `${e.clientY}px`;
      menuEl.current.style.display = 'block';
    }
  }, []);

  function hideMenu() {
    if (menuEl.current) menuEl.current.style.display = 'none';
  }

  function addFromMenu(type: string, data: Record<string, unknown>) {
    addNode({
      id: `${type}-${Date.now()}`,
      type,
      position: { x: menuPos.current.flowX, y: menuPos.current.flowY },
      data,
    });
    hideMenu();
  }

  return (
    <div className="flex h-full" onClick={hideMenu}>
      <Sidebar setView={setView} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Toolbar view={view} setView={setView} />

        {view === 'chat' && (
          <div className="flex-1 min-h-0 overflow-hidden">
            <ChatView />
          </div>
        )}

        <div className={`flex-1 relative ${view === 'chat' ? 'hidden' : ''}`} onContextMenu={handleContextMenu}>
          {/* Empty state hint */}
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className="text-center">
                <p className="text-4xl mb-3">🖱️</p>
                <p className="text-gray-500 font-medium mb-1">Right-click anywhere to add a node</p>
                <p className="text-gray-400 text-sm">Query · Note · Research Agent</p>
              </div>
            </div>
          )}

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            className="bg-gray-50"
          >
            <Background color="#e5e7eb" gap={24} />
            <Controls />
            <MiniMap
              nodeColor={(n) => {
                if (n.type === 'queryNode')     return '#60a5fa';
                if (n.type === 'answerNode')    return '#34d399';
                if (n.type === 'noteNode')      return '#fbbf24';
                if (n.type === 'agentNode')     return '#a78bfa';
                if (n.type === 'synthesisNode') return '#818cf8';
                return '#9ca3af';
              }}
            />
          </ReactFlow>

          {/* Right-click context menu */}
          <div
            ref={menuEl}
            className="hidden fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 min-w-48"
            style={{ display: 'none' }}
            onClick={e => e.stopPropagation()}
          >
            {CONTEXT_MENU_ITEMS.map(item => (
              <button
                key={item.type}
                className="w-full text-left text-sm text-gray-700 hover:bg-gray-50 px-4 py-2 transition-colors"
                onClick={() => addFromMenu(item.type, item.data)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
