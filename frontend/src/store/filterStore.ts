import { create } from 'zustand';

export const REPORT_TYPE_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'equity_research', label: 'Equity Research' },
  { value: 'technical_analysis', label: 'Technical Analysis' },
  { value: 'macro', label: 'Macro' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'sector_note', label: 'Sector Note' },
  { value: 'strategy', label: 'Strategy' },
  { value: 'other', label: 'Other' },
];

export const ASSET_CLASS_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'equity', label: 'Equity' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'fixed_income', label: 'Fixed Income' },
  { value: 'commodity', label: 'Commodity' },
  { value: 'fx', label: 'FX' },
  { value: 'mixed', label: 'Mixed' },
];

interface FilterStore {
  company: string;
  author: string;
  writtenDateFrom: string;
  writtenDateTo: string;
  ticker: string;       // single ticker symbol, e.g. "BTC"
  reportType: string;   // controlled vocab
  sector: string;       // free text GICS sector
  assetClass: string;   // controlled vocab

  setCompany: (v: string) => void;
  setAuthor: (v: string) => void;
  setWrittenDateFrom: (v: string) => void;
  setWrittenDateTo: (v: string) => void;
  setTicker: (v: string) => void;
  setReportType: (v: string) => void;
  setSector: (v: string) => void;
  setAssetClass: (v: string) => void;
  reset: () => void;
  activeCount: () => number;
}

const defaults = {
  company: '',
  author: '',
  writtenDateFrom: '',
  writtenDateTo: '',
  ticker: '',
  reportType: '',
  sector: '',
  assetClass: '',
};

export const useFilterStore = create<FilterStore>()((set, get) => ({
  ...defaults,

  setCompany: (v) => set({ company: v }),
  setAuthor: (v) => set({ author: v }),
  setWrittenDateFrom: (v) => set({ writtenDateFrom: v }),
  setWrittenDateTo: (v) => set({ writtenDateTo: v }),
  setTicker: (v) => set({ ticker: v }),
  setReportType: (v) => set({ reportType: v }),
  setSector: (v) => set({ sector: v }),
  setAssetClass: (v) => set({ assetClass: v }),
  reset: () => set(defaults),

  activeCount: () => {
    const { company, author, writtenDateFrom, writtenDateTo, ticker, reportType, sector, assetClass } = get();
    return [company, author, writtenDateFrom, writtenDateTo, ticker, reportType, sector, assetClass]
      .filter(Boolean).length;
  },
}));
