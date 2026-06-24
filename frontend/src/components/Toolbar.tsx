import { useCanvasStore } from '../store/canvasStore';

type View = 'canvas' | 'chat';

interface ToolbarProps {
  view: View;
  setView: (v: View) => void;
}

export function Toolbar({ view, setView }: ToolbarProps) {
  const { canvasName, canvasId, isDirty, saveCanvas, newCanvas } = useCanvasStore();

  return (
    <div className="h-12 bg-white border-b border-gray-200 flex items-center justify-between px-4 shrink-0">
      {/* View toggle */}
      <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
        <button
          onClick={() => setView('canvas')}
          className={`text-xs px-3 py-1 rounded-md transition-colors font-medium ${
            view === 'canvas'
              ? 'bg-white text-gray-800 shadow-sm'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Canvas
        </button>
        <button
          onClick={() => setView('chat')}
          className={`text-xs px-3 py-1 rounded-md transition-colors font-medium ${
            view === 'chat'
              ? 'bg-white text-gray-800 shadow-sm'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Chat
        </button>
      </div>

      {/* Canvas-only controls */}
      {view === 'canvas' && (
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-gray-800">{canvasName}</span>
          {isDirty && <span className="text-xs text-gray-400 italic">Unsaved</span>}
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                const name = prompt('New canvas name:', 'Untitled Canvas');
                if (name) newCanvas(name);
              }}
              className="text-xs border border-gray-200 hover:border-gray-400 text-gray-600 px-3 py-1.5 rounded-lg transition-colors"
            >
              New Canvas
            </button>
            <button
              onClick={saveCanvas}
              disabled={!canvasId || !isDirty}
              className="text-xs bg-green-500 hover:bg-green-600 disabled:bg-gray-200 disabled:text-gray-400 text-white px-3 py-1.5 rounded-lg transition-colors font-medium"
            >
              Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
