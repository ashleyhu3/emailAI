import { useEffect, useRef, useState } from 'react';
import { askQuestion } from '../api/client';
import { useChatStore } from '../store/chatStore';
import type { HistoryMessage } from '../types';
import { extractCitations, stripCitations } from '../lib/citations';

const HISTORY_WINDOW = 14;

function splitBlocks(content: string): string[] {
  const lines = content.split('\n');
  const blocks: string[] = [];
  let current: string[] = [];
  const flush = () => {
    const text = current.join('\n').trim();
    if (text) blocks.push(text);
    current = [];
  };
  const numbered = /^\s*\d+[).]\s/;
  for (const line of lines) {
    if (line.trim() === '') { flush(); continue; }
    if (numbered.test(line)) flush();
    current.push(line);
  }
  flush();
  return blocks;
}

function stripBold(text: string): string {
  return text.replace(/\*\*(.+?)\*\*/g, '$1');
}

function stripLeadingNumber(text: string): { text: string; hadNumber: boolean } {
  const m = text.match(/^\s*\d+\s*[).:\-]\s+/);
  return m ? { text: text.slice(m[0].length), hadNumber: true } : { text, hadNumber: false };
}

function renderAnswer(content: string, isEnumeration = false) {
  const blocks = splitBlocks(content);
  let itemNo = 0;
  return (
    <div className="flex flex-col gap-2">
      {blocks.map((block, i) => {
        const cites = extractCitations(block);
        let prose = stripBold(stripCitations(block));
        let number: number | null = null;
        if (isEnumeration) {
          const stripped = stripLeadingNumber(prose);
          if (stripped.hadNumber || cites.length > 0) {
            prose = stripped.text;
            itemNo += 1;
            number = itemNo;
          }
        }
        return (
          <div key={i} className="flex flex-col gap-1">
            {prose && (
              <div className="whitespace-pre-wrap leading-relaxed">
                {number != null && <span className="font-semibold">{number}) </span>}
                {prose}
              </div>
            )}
            {cites.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-0.5">
                {cites.map((c, j) => (
                  <span
                    key={j}
                    className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-200"
                  >
                    {c.filename}{c.pages.length > 0 ? ` p.${c.pages.join(', ')}` : ''}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function ChatView() {
  const { sessions, activeSessionId, activeSession, newSession, addMessage } = useChatStore();

  const session = activeSession();
  const messages = session?.messages ?? [];

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-create a session on first load
  useEffect(() => {
    if (!activeSessionId && sessions.length === 0) {
      newSession('New chat');
    } else if (!activeSessionId && sessions.length > 0) {
      useChatStore.getState().setActiveSession(sessions[0].id);
    }
  }, [activeSessionId, sessions, newSession]);

  function getOrCreateSessionId(): string {
    if (activeSessionId) return activeSessionId;
    return newSession('New chat');
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    const sessionId = getOrCreateSessionId();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    const currentMessages = activeSession()?.messages ?? [];
    addMessage(sessionId, { id: `${Date.now()}`, role: 'user', content: question });
    setLoading(true);

    const history: HistoryMessage[] = currentMessages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-HISTORY_WINDOW)
      .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));

    try {
      const result = await askQuestion({
        question,
        history: history.length > 0 ? history : undefined,
      });
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'assistant',
        content: result.answer,
        chunks: result.chunks_used,
        queryType: result.query_type,
        isEnumeration: result.is_enumeration,
        chartHtml: result.chart_html,
      });

      // Auto-name the session from the first user question (if still default name)
      const sess = useChatStore.getState().sessions.find(s => s.id === sessionId);
      if (sess && (sess.name === 'New chat' || sess.name.startsWith('Chat'))) {
        useChatStore.getState().renameSession(sessionId, question.slice(0, 40) + (question.length > 40 ? '…' : ''));
      }
    } catch {
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'system',
        content: 'Something went wrong. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleTextareaChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-white">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {isEmpty ? (
          /* Empty state — centered prompt */
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
            <div className="w-12 h-12 rounded-2xl bg-gray-900 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <p className="text-xl font-semibold text-gray-800">What would you like to know?</p>
            <p className="text-sm text-gray-400 max-w-sm">
              Ask anything about your research reports — ratings, price targets, analyst views, or sector trends.
            </p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
            {messages.map(msg => (
              <div
                key={msg.id}
                className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {msg.role === 'system' ? (
                  <p className="text-xs text-gray-400 text-center w-full py-1">{msg.content}</p>
                ) : msg.role === 'user' ? (
                  <div className="max-w-xl bg-gray-100 text-gray-800 rounded-2xl rounded-br-sm px-4 py-3 text-sm whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </div>
                ) : (
                  <div className={`flex gap-3 ${msg.chartHtml ? 'w-full max-w-none' : 'max-w-2xl'}`}>
                    {/* Avatar */}
                    <div className="w-7 h-7 rounded-full bg-gray-900 flex items-center justify-center shrink-0 mt-0.5">
                      <svg className="w-3.5 h-3.5 text-white" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm0 14.5a6.5 6.5 0 110-13 6.5 6.5 0 010 13z" />
                        <path d="M10 6a1 1 0 00-1 1v3a1 1 0 00.293.707l2 2a1 1 0 001.414-1.414L11 9.586V7a1 1 0 00-1-1z" />
                      </svg>
                    </div>
                    <div className={`text-sm text-gray-800 pt-0.5 ${msg.chartHtml ? 'flex-1 min-w-0' : ''}`}>
                      {renderAnswer(msg.content, msg.isEnumeration)}
                      {msg.chartHtml && (
                        <div className="mt-3 rounded-xl border border-gray-200 overflow-hidden shadow-sm bg-white">
                          <iframe
                            srcDoc={msg.chartHtml}
                            className="w-full"
                            style={{ height: '520px', border: 'none' }}
                            title="Morgan Stanley Research Chart"
                            sandbox="allow-scripts"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-gray-900 flex items-center justify-center shrink-0">
                  <svg className="w-3.5 h-3.5 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm0 14.5a6.5 6.5 0 110-13 6.5 6.5 0 010 13z" />
                    <path d="M10 6a1 1 0 00-1 1v3a1 1 0 00.293.707l2 2a1 1 0 001.414-1.414L11 9.586V7a1 1 0 00-1-1z" />
                  </svg>
                </div>
                <div className="flex items-center gap-1 pt-2">
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-gray-100 bg-white px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-3 bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm focus-within:border-gray-400 transition-colors">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Message Research Assistant…"
              rows={1}
              className="flex-1 bg-transparent resize-none text-sm text-gray-800 placeholder-gray-400 outline-none leading-relaxed"
              style={{ maxHeight: '160px' }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="bg-gray-900 hover:bg-gray-700 disabled:bg-gray-200 text-white disabled:text-gray-400 rounded-xl p-2 shrink-0 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
