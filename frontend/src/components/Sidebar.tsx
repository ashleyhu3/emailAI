import { useChatStore } from '../store/chatStore';

export function Sidebar() {
  const { sessions, activeSessionId, newSession, deleteSession, setActiveSession, renameSession } =
    useChatStore();

  function handleNew() {
    newSession(`New chat`);
  }

  return (
    <aside className="w-64 bg-gray-900 text-gray-100 flex flex-col h-full shrink-0">
      {/* Header */}
      <div className="px-4 pt-5 pb-3 border-b border-gray-700">
        <span className="text-sm font-semibold tracking-wide text-gray-100">Research Assistant</span>
      </div>

      {/* New chat button */}
      <div className="px-3 py-3">
        <button
          onClick={handleNew}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-gray-800 border border-gray-700 hover:border-gray-600 transition-colors"
        >
          <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New chat
        </button>
      </div>

      {/* Chat history */}
      <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-0.5">
        {sessions.length === 0 ? (
          <p className="text-xs text-gray-500 text-center mt-4">No chats yet</p>
        ) : (
          sessions.map(s => (
            <div
              key={s.id}
              className={`group flex items-center gap-1 rounded-lg px-3 py-2 text-sm cursor-pointer transition-colors ${
                s.id === activeSessionId
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
              onClick={() => setActiveSession(s.id)}
            >
              <svg className="w-3.5 h-3.5 shrink-0 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <span
                className="flex-1 truncate"
                onDoubleClick={e => {
                  e.stopPropagation();
                  const name = prompt('Rename chat:', s.name);
                  if (name?.trim()) renameSession(s.id, name.trim());
                }}
                title={s.name}
              >
                {s.name}
              </span>
              <button
                onClick={e => { e.stopPropagation(); deleteSession(s.id); }}
                className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 transition-opacity shrink-0 ml-1"
                title="Delete"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
