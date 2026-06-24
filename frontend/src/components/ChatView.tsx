import { useEffect, useRef, useState } from 'react';
import { askQuestion, uploadDocument } from '../api/client';
import { useChatStore } from '../store/chatStore';
import type { FilterContext } from '../store/chatStore';
import { useDocumentStore } from '../store/documentStore';
import { useFilterStore } from '../store/filterStore';
import type { HistoryMessage } from '../types';
import { extractCitations, stripCitations } from '../lib/citations';

// Last 14 messages (7 Q&A pairs) sent as conversation context with every request.
const HISTORY_WINDOW = 14;

// Split an answer into blocks: each numbered list item ("1)" / "1.", including its
// indented "Source:" continuation line) or each blank-line-separated paragraph.
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
    if (line.trim() === '') {
      flush();
      continue;
    }
    if (numbered.test(line)) flush();
    current.push(line);
  }
  flush();
  return blocks;
}

// No markdown renderer is wired up; just drop **bold** markers so filenames read cleanly.
function stripBold(text: string): string {
  return text.replace(/\*\*(.+?)\*\*/g, '$1');
}

// Strip a leading list marker ("1)", "1.", "1 -", "1:") so we can renumber consistently.
function stripLeadingNumber(text: string): { text: string; hadNumber: boolean } {
  const m = text.match(/^\s*\d+\s*[).:\-]\s+/);
  return m ? { text: text.slice(m[0].length), hadNumber: true } : { text, hadNumber: false };
}

// Render an assistant answer with each block's source citations as blue pills directly
// beneath that block. When the response is an enumeration (e.g. "list 10 trends"), number
// the items 1), 2), 3), ... deterministically so the format is consistent regardless of
// whether the model numbered them itself.
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
          // A block is a list item if it was numbered or carries its own source.
          if (stripped.hadNumber || cites.length > 0) {
            prose = stripped.text;
            itemNo += 1;
            number = itemNo;
          }
        }
        return (
          <div key={i} className="flex flex-col gap-1">
            {prose && (
              <div className="whitespace-pre-wrap">
                {number != null && <span className="font-semibold">{number}) </span>}
                {prose}
              </div>
            )}
            {cites.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {cites.map((c, j) => (
                  <span
                    key={j}
                    className="text-xs px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200"
                  >
                    {c.filename}
                    {c.pages.length > 0 ? ` p.${c.pages.join(', ')}` : ''}
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
  const { documents, fetchDocuments, setHighlightedDocs, clearHighlights } = useDocumentStore();
  const {
    company, author, writtenDateFrom, writtenDateTo,
    ticker, reportType, sector, assetClass,
    setCompany, setAuthor, setWrittenDateFrom, setWrittenDateTo,
    setTicker, setReportType, setSector, setAssetClass,
    reset: resetFilters, activeCount,
  } = useFilterStore();

  const session = activeSession();
  const messages = session?.messages ?? [];

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Ensure there's always an active session
  useEffect(() => {
    if (!activeSessionId && sessions.length === 0) {
      newSession('Chat 1');
    } else if (!activeSessionId && sessions.length > 0) {
      useChatStore.getState().setActiveSession(sessions[0].id);
    }
  }, [activeSessionId, sessions, newSession]);

  function getOrCreateSessionId(): string {
    if (activeSessionId) return activeSessionId;
    return newSession();
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    const sessionId = getOrCreateSessionId();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    // Snapshot current messages BEFORE adding the new user message so we don't
    // include the message being sent in the history we pass to the backend.
    const currentMessages = activeSession()?.messages ?? [];

    addMessage(sessionId, { id: `${Date.now()}`, role: 'user', content: question });
    setLoading(true);
    const filterContext = buildFilterContext();

    // Build conversation history: last HISTORY_WINDOW user+assistant messages.
    const history: HistoryMessage[] = currentMessages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-HISTORY_WINDOW)
      .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }));

    try {
      const result = await askQuestion({
        question,
        history: history.length > 0 ? history : undefined,
        sender_companies: company ? [company] : undefined,
        sender_names: author ? [author] : undefined,
        written_date_from: writtenDateFrom || undefined,
        written_date_to: writtenDateTo || undefined,
        tickers: ticker ? [ticker] : undefined,
        report_type: reportType || undefined,
        sector: sector || undefined,
        asset_class: assetClass || undefined,
      });
      // Highlight, on the left, exactly the documents this response references — for every
      // response type. RAG answers carry their cited chunks (document_id verified by the
      // backend); list-document and follow-up answers don't, so fall back to matching any
      // document whose filename appears in the answer text (the listed/cited filenames).
      let citedIds = [...new Set(result.chunks_used.map(c => c.document_id))];
      if (citedIds.length === 0) {
        citedIds = documents.filter(d => result.answer.includes(d.filename)).map(d => d.id);
      }
      // Order the cited documents by where each first appears in the answer text, so the
      // sidebar reflects the order they were actually cited in the response.
      const citeOrder = (id: number) => {
        const fn = documents.find(d => d.id === id)?.filename;
        const idx = fn ? result.answer.indexOf(fn) : -1;
        return idx === -1 ? Number.MAX_SAFE_INTEGER : idx;
      };
      citedIds = [...citedIds].sort((a, b) => citeOrder(a) - citeOrder(b));
      setHighlightedDocs(citedIds, citedIds);
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'assistant',
        content: result.answer,
        chunks: result.chunks_used,
        filterContext,
        inferredFilters: result.inferred_filters,
        queryType: result.query_type,
        isEnumeration: result.is_enumeration,
      });
    } catch {
      clearHighlights();
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'system',
        content: 'Error getting answer. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;

    const sessionId = getOrCreateSessionId();
    setUploading(true);

    addMessage(sessionId, {
      id: `${Date.now()}`,
      role: 'system',
      content: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}... This may take several minutes.`,
    });

    for (const file of files) {
      try {
        const result = await uploadDocument(file);
        addMessage(sessionId, {
          id: `${Date.now()}-${file.name}`,
          role: 'system',
          content: result.status === 'skipped'
            ? `⚠️ "${file.name}" is already in the database — skipped.`
            : `✓ "${file.name}" ingested and ready to query.`,
        });
      } catch {
        addMessage(sessionId, {
          id: `${Date.now()}-${file.name}-err`,
          role: 'system',
          content: `✗ "${file.name}" failed to upload. Please try again.`,
        });
      }
    }

    setUploading(false);
    if (fileRef.current) fileRef.current.value = '';
    await fetchDocuments();
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
    e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
  }

  const filterCount = activeCount();

  // Count only user + assistant messages (not system notices) to determine
  // whether the history window has been exceeded.
  const conversationMessageCount = messages.filter(
    m => m.role === 'user' || m.role === 'assistant'
  ).length;
  const historyOverflow = conversationMessageCount > HISTORY_WINDOW;

  function buildFilterContext(): FilterContext | undefined {
    const ctx: FilterContext = {};
    if (company) ctx.senderCompanies = [company];
    if (author) ctx.senderNames = [author];
    if (writtenDateFrom) ctx.writtenDateFrom = writtenDateFrom;
    if (writtenDateTo) ctx.writtenDateTo = writtenDateTo;
    return Object.keys(ctx).length > 0 ? ctx : undefined;
  }

  function formatFilterContext(ctx: FilterContext): string {
    const parts: string[] = [];
    if (ctx.senderCompanies?.length) parts.push(ctx.senderCompanies.join(', '));
    if (ctx.senderNames?.length) parts.push(`by ${ctx.senderNames.join(', ')}`);
    if (ctx.writtenDateFrom || ctx.writtenDateTo)
      parts.push(`written ${ctx.writtenDateFrom ?? '…'}–${ctx.writtenDateTo ?? '…'}`);
    return parts.join(' · ');
  }

  /** Format auto-inferred hard filters from the backend as readable chip labels. */
  function inferredFilterChips(filters: Record<string, unknown>): string[] {
    const chips: string[] = [];
    const companies = filters.sender_companies as string[] | null;
    if (companies?.length) chips.push(`🏢 ${companies.join(', ')}`);
    const from = filters.written_date_from as string | null;
    const to = filters.written_date_to as string | null;
    if (from || to) chips.push(`📅 published ${from ?? '…'} → ${to ?? '…'}`);
    const covFrom = filters.coverage_period_from as string | null;
    const covTo = filters.coverage_period_to as string | null;
    if (covFrom || covTo) chips.push(`🗓️ covers ${covFrom ?? '…'} → ${covTo ?? '…'}`);
    const pageMin = filters.page_min as number | null;
    const pageMax = filters.page_max as number | null;
    if (pageMin != null || pageMax != null) {
      chips.push(`📄 pp. ${pageMin ?? '…'}–${pageMax ?? '…'}`);
    }
    return chips;
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Context window overflow warning */}
      {historyOverflow && (
        <div className="flex items-start gap-2 px-4 py-2.5 bg-amber-50 border-b border-amber-200 text-xs text-amber-800">
          <svg className="w-3.5 h-3.5 shrink-0 mt-0.5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          <span>
            <strong>Long conversation:</strong> only the last 7 exchanges are included in each response.
            Earlier messages are no longer in context.
          </span>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <p className="text-lg font-medium mb-1">Ask a question</p>
              <p className="text-sm">Upload a PDF using the paperclip, then start asking questions</p>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'system' ? (
              <div className="text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-lg px-3 py-2 max-w-xl mx-auto text-center">
                {msg.content}
              </div>
            ) : (
              <div className={`flex flex-col gap-1 max-w-2xl ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white rounded-br-sm'
                    : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                }`}>
                  {msg.role === 'assistant' ? renderAnswer(msg.content, msg.isEnumeration) : msg.content}
                </div>
                {msg.role === 'assistant' && msg.filterContext && (
                  <p className="text-xs text-gray-400 px-1 flex items-center gap-1">
                    <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1 0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z" />
                    </svg>
                    {formatFilterContext(msg.filterContext)}
                  </p>
                )}
                {msg.role === 'assistant' && msg.queryType !== 'list_documents' &&
                  msg.inferredFilters &&
                  inferredFilterChips(msg.inferredFilters).length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1 items-center">
                    <span className="text-xs text-indigo-400 font-medium">Auto-filtered:</span>
                    {inferredFilterChips(msg.inferredFilters).map((chip, i) => (
                      <span key={i} className="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded-full">
                        {chip}
                      </span>
                    ))}
                  </div>
                )}
                {msg.role === 'assistant' && msg.queryType === 'list_documents' && (
                  <p className="text-xs text-gray-400 px-1">Document inventory</p>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1 items-center">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Active filter chips — sourced from global filterStore */}
      {filterCount > 0 && (
        <div className="border-t border-gray-100 px-4 py-2 flex flex-wrap gap-1.5 items-center bg-blue-50">
          <span className="text-xs text-blue-400 font-medium shrink-0">Filtering:</span>
          {company && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full">
              {company}
              <button onClick={() => setCompany('')} className="text-blue-400 hover:text-blue-600 leading-none">×</button>
            </span>
          )}
          {author && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-purple-700 border border-purple-200 px-2 py-0.5 rounded-full">
              by {author}
              <button onClick={() => setAuthor('')} className="text-purple-400 hover:text-purple-600 leading-none">×</button>
            </span>
          )}
          {(writtenDateFrom || writtenDateTo) && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-gray-700 border border-gray-200 px-2 py-0.5 rounded-full">
              written {writtenDateFrom || '…'}–{writtenDateTo || '…'}
              <button onClick={() => { setWrittenDateFrom(''); setWrittenDateTo(''); }} className="text-gray-400 hover:text-gray-600 leading-none">×</button>
            </span>
          )}
          {ticker && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full">
              {ticker}
              <button onClick={() => setTicker('')} className="text-emerald-400 hover:text-emerald-600 leading-none">×</button>
            </span>
          )}
          {reportType && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-orange-700 border border-orange-200 px-2 py-0.5 rounded-full">
              {reportType.replace(/_/g, ' ')}
              <button onClick={() => setReportType('')} className="text-orange-400 hover:text-orange-600 leading-none">×</button>
            </span>
          )}
          {assetClass && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-teal-700 border border-teal-200 px-2 py-0.5 rounded-full">
              {assetClass.replace(/_/g, ' ')}
              <button onClick={() => setAssetClass('')} className="text-teal-400 hover:text-teal-600 leading-none">×</button>
            </span>
          )}
          {sector && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-indigo-700 border border-indigo-200 px-2 py-0.5 rounded-full">
              {sector}
              <button onClick={() => setSector('')} className="text-indigo-400 hover:text-indigo-600 leading-none">×</button>
            </span>
          )}
          <button onClick={resetFilters} className="text-xs text-blue-400 hover:text-blue-600 ml-1">
            Clear all
          </button>
        </div>
      )}

      {/* Input bar */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-2">
          {/* Upload button */}
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            title="Upload PDF(s)"
            className="text-gray-400 hover:text-blue-500 disabled:opacity-40 transition-colors p-1 shrink-0 mb-0.5"
          >
            {uploading ? (
              <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            )}
          </button>
          <input ref={fileRef} type="file" accept=".pdf" multiple className="hidden" onChange={handleUpload} />

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 bg-transparent resize-none text-sm text-gray-800 placeholder-gray-400 outline-none"
            style={{ maxHeight: '128px' }}
          />

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 text-white disabled:text-gray-400 rounded-xl p-1.5 shrink-0 transition-colors mb-0.5"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1.5 px-1">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
