// Parsing helpers for in-text source citations like "(report.pdf, p.3)" or
// "(report.pdf, pages 9, 11)". These mirror the backend regex in
// PDF_summarizer/rag_gemini.py (_CITATION_RE / _expand_pages) so the sources rendered
// in the chat stay consistent with the chunks the backend selected as cited.

export interface Citation {
  filename: string;
  pages: number[];
}

// Capturing: group 1 = filename (may contain one level of nested parens, e.g.
// "91APP (6741).pdf" or "email_abc123.eml"), group 2 = page spec (OPTIONAL).
// Global + case-insensitive. Matches both .pdf and .eml document citations.
export const CITATION_RE =
  /\(((?:[^()]|\([^()]*\))*?\.(?:pdf|eml))(?:[\s,]+((?:pp?\.?|pages?)?\s*\d[\d,\s\-–]*))?\)/gi;

/** Extract page numbers from a citation page spec, expanding ranges.
 *  "pages 9, 11, 12" -> [9, 11, 12];  "pp. 3-5" -> [3, 4, 5]. */
export function expandPages(spec: string | undefined): number[] {
  const pages = new Set<number>();
  if (!spec) return [];
  // Ranges first: "3-5" / "3–5".
  const rangeRe = /(\d+)\s*[-–]\s*(\d+)/g;
  let m: RegExpExecArray | null;
  while ((m = rangeRe.exec(spec)) !== null) {
    const lo = parseInt(m[1], 10);
    const hi = parseInt(m[2], 10);
    if (hi - lo > 0 && hi - lo < 100) {
      for (let p = lo; p <= hi; p++) pages.add(p);
    }
  }
  // Remaining standalone numbers (with ranges removed so endpoints aren't double-counted).
  const withoutRanges = spec.replace(/\d+\s*[-–]\s*\d+/g, ' ');
  const numRe = /\d+/g;
  while ((m = numRe.exec(withoutRanges)) !== null) {
    pages.add(parseInt(m[0], 10));
  }
  return [...pages].sort((a, b) => a - b);
}

/** Find all citation markers in `text`, deduped by filename with merged page lists,
 *  preserving first-seen order. */
export function extractCitations(text: string): Citation[] {
  const order: string[] = [];
  const pagesByKey = new Map<string, Set<number>>();
  const displayByKey = new Map<string, string>();

  CITATION_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = CITATION_RE.exec(text)) !== null) {
    const filename = m[1];
    const key = filename.toLowerCase();
    if (!pagesByKey.has(key)) {
      pagesByKey.set(key, new Set());
      displayByKey.set(key, filename);
      order.push(key);
    }
    for (const p of expandPages(m[2])) pagesByKey.get(key)!.add(p);
  }

  return order.map(key => ({
    filename: displayByKey.get(key)!,
    pages: [...pagesByKey.get(key)!].sort((a, b) => a - b),
  }));
}

/** Remove citation markers from prose, along with any leading "Source:"/"Sources:" label
 *  and the punctuation/whitespace left behind, so the text reads cleanly. */
export function stripCitations(text: string): string {
  let out = text.replace(CITATION_RE, '');
  // Drop a now-empty "Source:"/"Sources:" label (optionally with trailing separators).
  out = out.replace(/\bSources?:\s*(?=$|\n)/gim, '');
  // Tidy separators/space left where citations were removed.
  out = out.replace(/[ \t]*,(?=\s*,)/g, '');
  out = out.replace(/[ \t]{2,}/g, ' ');
  // Drop a space that now sits before sentence punctuation (e.g. "repricing ." -> "repricing.").
  out = out.replace(/[ \t]+([.,;:)])/g, '$1');
  out = out.replace(/[ \t]+(?=\n)/g, '');
  // Collapse a line that became blank apart from punctuation left over.
  out = out.replace(/\n[ \t]*[.,;]+[ \t]*(?=\n|$)/g, '');
  return out.replace(/[ \t]+$/gm, '').trimEnd();
}
