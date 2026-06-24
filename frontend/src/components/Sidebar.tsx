import { useEffect, useRef, useState } from 'react';

const CIRCUMFERENCE = 2 * Math.PI * 9; // r=9 in a 24×24 viewBox
import { useDocumentStore } from '../store/documentStore';
import { useCanvasStore } from '../store/canvasStore';
import { useChatStore } from '../store/chatStore';
import { useFilterStore, REPORT_TYPE_OPTIONS, ASSET_CLASS_OPTIONS } from '../store/filterStore';

export function Sidebar({ setView }: { setView: (v: 'canvas' | 'chat') => void }) {
  const { documents, loading, fetchDocuments, upload, remove, fetchedIds, chosenIds } = useDocumentStore();
  const { savedCanvases, fetchSavedCanvases, loadCanvas, removeCanvas, newCanvas } = useCanvasStore();
  const { sessions, activeSessionId, newSession, deleteSession, setActiveSession, renameSession, addMessage } = useChatStore();
  const {
    company, author, writtenDateFrom, writtenDateTo,
    ticker, reportType, sector, assetClass,
    setCompany, setAuthor, setWrittenDateFrom, setWrittenDateTo,
    setTicker, setReportType, setSector, setAssetClass,
    reset: resetFilters, activeCount,
  } = useFilterStore();
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [documentsOpen, setDocumentsOpen] = useState(true);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    if (!uploading) { setUploadProgress(0); return; }
    // Simulate progress: fill to ~85% over ~2 minutes, then hold until done.
    const id = setInterval(() => setUploadProgress(p => Math.min(p + 0.7, 85)), 1000);
    return () => clearInterval(id);
  }, [uploading]);

  useEffect(() => {
    fetchDocuments();
    fetchSavedCanvases();
  }, []);

  // Whenever a new response cites documents, open the Documents section so the blue
  // highlights are visible without the user having to expand it manually.
  useEffect(() => {
    if (chosenIds.length > 0) setDocumentsOpen(true);
  }, [chosenIds]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      for (const file of files) {
        await upload(file);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(msg);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  // Rank a doc by highlight tier: chosen (cited) = 0, fetched (retrieved) = 1, none = 2.
  const docRank = (id: number) => (chosenIds.includes(id) ? 0 : fetchedIds.includes(id) ? 1 : 2);

  // Client-side filter + sort cited docs to the top, then retrieved, then the rest.
  // Within the cited tier, preserve the order they were cited in the answer (chosenIds
  // is already ordered by first appearance in the response text).
  const filteredDocs = documents
    .filter(doc => {
      if (company && !doc.sender_company?.toLowerCase().includes(company.toLowerCase())) return false;
      if (author && !doc.sender_name?.toLowerCase().includes(author.toLowerCase())) return false;
      if (writtenDateFrom && doc.written_date && doc.written_date < writtenDateFrom) return false;
      if (writtenDateTo && doc.written_date && doc.written_date > writtenDateTo) return false;
      if (ticker && !doc.tickers?.some(t => t.toLowerCase().includes(ticker.toLowerCase()))) return false;
      if (reportType && doc.report_type !== reportType) return false;
      if (sector && !doc.sector?.toLowerCase().includes(sector.toLowerCase())) return false;
      if (assetClass && doc.asset_class !== assetClass) return false;
      return true;
    })
    .sort((a, b) => {
      const rankDiff = docRank(a.id) - docRank(b.id);
      if (rankDiff !== 0) return rankDiff;
      if (docRank(a.id) === 0) return chosenIds.indexOf(a.id) - chosenIds.indexOf(b.id);
      return 0;
    });

  const count = activeCount();
  const hasHighlights = fetchedIds.length > 0;

  return (
    <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto">

        {/* Filters section */}
        <div className="border-b border-gray-200">
          <button
            onClick={() => setFiltersOpen(o => !o)}
            className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-gray-100 transition-colors"
          >
            <div className="flex items-center gap-1.5">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Filters</h2>
              {count > 0 && (
                <span className="text-xs bg-blue-500 text-white rounded-full px-1.5 py-0.5 leading-none">
                  {count}
                </span>
              )}
            </div>
            <svg
              className={`w-3.5 h-3.5 text-gray-400 transition-transform ${filtersOpen ? 'rotate-180' : ''}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {filtersOpen && (
            <div className="px-3 pb-3 space-y-2.5">
              {/* Company */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Company</label>
                <input
                  type="text"
                  value={company}
                  onChange={e => setCompany(e.target.value)}
                  placeholder="e.g. Apple"
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                />
              </div>

              {/* Author */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Author</label>
                <input
                  type="text"
                  value={author}
                  onChange={e => setAuthor(e.target.value)}
                  placeholder="e.g. John Smith"
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                />
              </div>

              {/* Date written */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Date written</label>
                <div className="flex items-center gap-1">
                  <input
                    type="date"
                    value={writtenDateFrom}
                    onChange={e => setWrittenDateFrom(e.target.value)}
                    className="flex-1 min-w-0 text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                  />
                  <span className="text-xs text-gray-400 shrink-0">–</span>
                  <input
                    type="date"
                    value={writtenDateTo}
                    onChange={e => setWrittenDateTo(e.target.value)}
                    className="flex-1 min-w-0 text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                  />
                </div>
              </div>

              {/* Ticker */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Ticker</label>
                <input
                  type="text"
                  value={ticker}
                  onChange={e => setTicker(e.target.value.toUpperCase())}
                  placeholder="e.g. BTC, AAPL"
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                />
              </div>

              {/* Report type */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Report type</label>
                <select
                  value={reportType}
                  onChange={e => setReportType(e.target.value)}
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                >
                  {REPORT_TYPE_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>

              {/* Asset class */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Asset class</label>
                <select
                  value={assetClass}
                  onChange={e => setAssetClass(e.target.value)}
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                >
                  {ASSET_CLASS_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>

              {/* Sector */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Sector</label>
                <input
                  type="text"
                  value={sector}
                  onChange={e => setSector(e.target.value)}
                  placeholder="e.g. Technology, Energy"
                  className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:ring-1 focus:ring-blue-300 bg-white"
                />
              </div>

              {count > 0 && (
                <button
                  onClick={resetFilters}
                  className="w-full text-xs text-red-400 hover:text-red-600 text-center py-1 transition-colors"
                >
                  Clear all filters
                </button>
              )}
            </div>
          )}
        </div>

        {/* Documents section */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <button
              onClick={() => setDocumentsOpen(o => !o)}
              className="flex items-center gap-1.5 hover:opacity-70 transition-opacity"
            >
              <svg
                className={`w-3.5 h-3.5 text-gray-400 transition-transform ${documentsOpen ? 'rotate-180' : ''}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Documents</h2>
              {documents.length > 0 && (
                <span className="text-xs text-gray-400">
                  {count > 0 ? `${filteredDocs.length}/${documents.length}` : documents.length}
                </span>
              )}
              {chosenIds.length > 0 && (
                <span className="text-xs bg-blue-500 text-white rounded-full px-1.5 py-0.5 leading-none">
                  {chosenIds.length} cited
                </span>
              )}
            </button>
            <button
              onClick={() => { setUploadError(null); fileRef.current?.click(); }}
              disabled={uploading}
              className="text-xs bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-2 py-1 rounded-md transition-colors"
            >
              + Upload
            </button>
            <input ref={fileRef} type="file" accept=".pdf" multiple className="hidden" onChange={handleUpload} />
          </div>

          {/* Highlight legend — shown after a query highlights documents */}
          {documentsOpen && hasHighlights && (
            <div className="flex items-center gap-3 mb-2 px-0.5 text-xs text-gray-400">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" />
                Cited in this answer
              </span>
            </div>
          )}

          {/* Upload progress banner */}
          {uploading && (
            <div className="mb-2 bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center gap-3">
              <div className="shrink-0">
                <svg width="36" height="36" viewBox="0 0 24 24">
                  {/* Track */}
                  <circle cx="12" cy="12" r="9" fill="none" stroke="#bfdbfe" strokeWidth="2.5" />
                  {/* Fill */}
                  <circle
                    cx="12" cy="12" r="9"
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeDasharray={CIRCUMFERENCE}
                    strokeDashoffset={CIRCUMFERENCE * (1 - uploadProgress / 100)}
                    style={{ transform: 'rotate(-90deg)', transformOrigin: 'center', transition: 'stroke-dashoffset 0.8s ease' }}
                  />
                  {/* Percentage */}
                  <text x="12" y="12" dominantBaseline="central" textAnchor="middle" fontSize="5.5" fill="#3b82f6" fontWeight="600">
                    {Math.round(uploadProgress)}%
                  </text>
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium text-blue-700 mb-0.5">Processing PDF…</p>
                <p className="text-xs text-blue-500 leading-tight">Parsing &amp; verbalizing charts.<br />Takes 1–3 min.</p>
              </div>
            </div>
          )}

          {uploadError && (
            <div className="mb-2 bg-red-50 border border-red-200 rounded-lg p-2 text-xs text-red-600">
              {uploadError}
            </div>
          )}

          {documentsOpen && (loading ? (
            <p className="text-xs text-gray-400 text-center py-2">Loading…</p>
          ) : documents.length === 0 && !uploading ? (
            <p className="text-xs text-gray-400 text-center py-2">No documents yet</p>
          ) : filteredDocs.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-2">No documents match filters</p>
          ) : (
            <ul className="space-y-1">
              {filteredDocs.map(doc => {
                const isChosen = chosenIds.includes(doc.id);
                const isFetched = fetchedIds.includes(doc.id);
                return (
                <li
                  key={doc.id}
                  className={`flex items-start justify-between gap-1 rounded-lg px-2 py-1.5 text-xs border transition-colors ${
                    isChosen
                      ? 'bg-blue-50 border-blue-400'
                      : isFetched
                        ? 'bg-white border-blue-300 border-dashed'
                        : 'bg-white border-gray-100'
                  }`}
                >
                  <div className="min-w-0">
                    <p className="font-medium text-gray-700 truncate flex items-center gap-1.5" title={doc.filename}>
                      {(isChosen || isFetched) && (
                        <span
                          className={`w-2 h-2 rounded-full shrink-0 ${
                            isChosen ? 'bg-blue-500' : 'border border-blue-400 bg-white'
                          }`}
                          title={isChosen ? 'Cited in answer' : 'Retrieved'}
                        />
                      )}
                      <span className="truncate">{doc.filename}</span>
                    </p>
                    <p className="text-gray-400">
                      {doc.total_pages} pages
                      {doc.sender_company && <span> · {doc.sender_company}</span>}
                    </p>
                    {doc.tickers && doc.tickers.length > 0 && (
                      <p className="text-gray-400 truncate">
                        {doc.tickers.slice(0, 4).join(' · ')}
                        {doc.tickers.length > 4 && ' …'}
                      </p>
                    )}
                    {(doc.report_type || doc.written_date) && (
                      <p className="text-gray-400">
                        {doc.report_type && (
                          <span className="capitalize">{doc.report_type.replace(/_/g, ' ')}</span>
                        )}
                        {doc.report_type && doc.written_date && ' · '}
                        {doc.written_date && `Written ${doc.written_date}`}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={() => remove(doc.id)}
                    className="text-gray-300 hover:text-red-400 shrink-0 mt-0.5"
                    title="Delete"
                  >
                    ×
                  </button>
                </li>
                );
              })}
            </ul>
          ))}
        </div>

        {/* Canvases section */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Canvases</h2>
            <button
              onClick={() => {
                const name = prompt('Canvas name:', 'Untitled Canvas');
                if (name) newCanvas(name);
              }}
              className="text-xs bg-purple-500 hover:bg-purple-600 text-white px-2 py-1 rounded-md transition-colors"
            >
              + New
            </button>
          </div>

          {savedCanvases.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-2">No saved canvases</p>
          ) : (
            <ul className="space-y-1">
              {savedCanvases.map(c => (
                <li
                  key={c.id}
                  className="flex items-center justify-between gap-1 bg-white rounded-lg px-2 py-1.5 text-xs border border-gray-100"
                >
                  <button
                    className="text-gray-700 hover:text-purple-600 font-medium truncate text-left flex-1"
                    onClick={() => loadCanvas(c.id)}
                    title={c.name}
                  >
                    {c.name}
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete canvas "${c.name}"?`)) removeCanvas(c.id);
                    }}
                    className="text-gray-300 hover:text-red-400 shrink-0"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Chats section */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Chats</h2>
            <button
              onClick={() => {
                const id = newSession(`Chat ${sessions.length + 1}`);
                addMessage(id, {
                  id: `sys-${Date.now()}`,
                  role: 'system',
                  content: 'New conversation started. The assistant has no memory of previous chats — each session is independent.',
                });
                setView('chat');
              }}
              className="text-xs bg-blue-500 hover:bg-blue-600 text-white px-2 py-1 rounded-md transition-colors"
            >
              + New
            </button>
          </div>

          {sessions.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-2">No chats yet</p>
          ) : (
            <ul className="space-y-1">
              {sessions.map(s => (
                <li
                  key={s.id}
                  className={`flex items-center justify-between gap-1 rounded-lg px-2 py-1.5 text-xs border ${
                    s.id === activeSessionId
                      ? 'bg-blue-50 border-blue-200'
                      : 'bg-white border-gray-100'
                  }`}
                >
                  <button
                    className="text-gray-700 hover:text-blue-600 font-medium truncate text-left flex-1"
                    onClick={() => { setActiveSession(s.id); setView('chat'); }}
                    title={s.name}
                    onDoubleClick={() => {
                      const name = prompt('Rename chat:', s.name);
                      if (name) renameSession(s.id, name);
                    }}
                  >
                    {s.name}
                  </button>
                  <button
                    onClick={() => deleteSession(s.id)}
                    className="text-gray-300 hover:text-red-400 shrink-0"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* How to use */}
        <div className="p-3">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">How to use</h2>
          <ul className="space-y-2 text-xs text-gray-500">
            <li className="flex gap-2">
              <span className="shrink-0">1.</span>
              <span>Upload a PDF above</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">2.</span>
              <span><strong className="text-gray-600">Right-click</strong> the canvas to add a node</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">3.</span>
              <span><strong className="text-gray-600">Query node</strong> — ask a question, get an answer</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">4.</span>
              <span><strong className="text-gray-600">Agent node</strong> — give a high-level goal, auto-generates multi-step research</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">5.</span>
              <span>Drag edges between nodes to link findings</span>
            </li>
          </ul>
        </div>

      </div>
    </div>
  );
}
