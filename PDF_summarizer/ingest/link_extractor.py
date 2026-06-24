"""
PDF link extractor for notification emails.

Many broker notification emails contain no PDF attachment — only a link to a
research portal (e.g. ResearchFN, Goldman portal, Barclays Live) where the PDF
can be downloaded.

This module:
  1. Extracts candidate URLs from email HTML.
  2. Follows each URL (handling redirects).
  3. Returns PDF bytes if the final response is a PDF.
  4. If the response is HTML, parses the page for nested PDF links and downloads those.
  5. Optionally uses browser cookies (stored in env/file) for authenticated portals.

Environment variables:
  PORTAL_COOKIES  — JSON string mapping domain → Cookie header value.
                    Example: {"researchfn.com": "session=abc123; token=xyz"}
                    Obtain from your browser DevTools → Network → any request → Cookie header.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Hard ceiling on total time spent fetching links per email
_GLOBAL_TIMEOUT_SECS = 30


# ── Heuristics for which links look like report links ────────────────────────

# URL patterns that are almost certainly NOT research reports
_SKIP_PATTERNS = re.compile(
    r"(unsubscribe|optout|opt-out|privacy|mailto:|tel:|javascript:|"
    r"linkedin\.com|twitter\.com|facebook\.com|instagram\.com|"
    r"researchfn\.com/?$|www\.researchfn\.com/?$|"  # root domain, no path
    r"\.(png|jpg|jpeg|gif|svg|ico|css|js|woff|woff2|ttf)(\?|$)|"
    # Government / regulatory domains — sanctions advisories, SEC filings, etc.
    r"(?:treasury\.gov|sec\.gov|fdic\.gov|federalreserve\.gov|cftc\.gov|"
    r"bis\.org|finra\.org|esma\.europa\.eu|fca\.org\.uk|mas\.gov\.sg|"
    r"hkma\.gov\.hk|sfc\.hk|mof\.gov\.cn|csrc\.gov\.cn|"
    r"home\.treasury\.gov|ofac\.treasury\.gov|"
    r"\.(gov|mil)/[^\"']*\.pdf)|"  # any .gov/.mil PDF
    # Compliance/regulatory/legal documents (not research)
    r"(?:compliance[-_]disclosure|regulatory[-_](?:disclosure|notice)|"
    r"privacy[-_](?:notice|policy)|terms[-_]of[-_]service|"
    r"ca[-_]ab[-_]\d{4}|form[-_]adv|proxy[-_]statement|"
    r"disclaimer[-_]only|legal[-_]notice)|"
    # Email tracking redirects and JWT token URLs (Goldman Marquee, etc.)
    # These redirect through auth portals and never yield a direct PDF
    r"(?:/t/r/[A-Za-z0-9+/=._-]{40,}|"      # token redirect paths
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})"  # JWT in URL path/query
    r")",
    re.IGNORECASE,
)

# Filename patterns for PDFs that are compliance/legal documents, not research
_NON_RESEARCH_PDF_RE = re.compile(
    r"(?:compliance[-_\s]disclosure|regulatory[-_\s](?:disclosure|notice)|"
    r"privacy[-_\s](?:notice|policy)|terms[-_\s](?:of[-_\s]service|and[-_\s]conditions)|"
    r"ca[-_]ab[-_]\d{4}|form[-_\s]adv|annual[-_\s]report[-_\s](?:to|for)|"
    r"proxy[-_\s]statement|disclaimer[-_\s]only|legal[-_\s]notice)",
    re.IGNORECASE,
)

# URL patterns that look especially promising
_REPORT_PATTERNS = re.compile(
    r"(\.pdf(\?|$)|/report|/research|/document|/download|/file|"
    r"researchfn\.com/c/|/view|portal|access|content|publication)",
    re.IGNORECASE,
)

# PDF content-type variants
_PDF_CONTENT_TYPES = (
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # sometimes used for binary downloads
)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,application/pdf,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _load_portal_cookies() -> dict:
    """
    Load per-domain cookie strings from the PORTAL_COOKIES environment variable.

    Format: JSON object mapping domain strings to Cookie header values.
    Example .env entry:
      PORTAL_COOKIES={"researchfn.com": "session=abc; _ga=GA1.2.xyz"}
    """
    raw = os.getenv("PORTAL_COOKIES", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _browser_cookiejar_for_url(url: str):
    """
    Return a requests-compatible cookiejar from the user's local browsers for
    the given URL's domain. Tries Chrome → Firefox → Safari in order.

    Returns None if browser-cookie3 is unavailable or no cookies found.
    Requires Terminal to have Full Disk Access on macOS for Safari.
    """
    try:
        import browser_cookie3
    except ImportError:
        return None

    domain = urlparse(url).netloc
    # Strip leading 'www.' and convert to dotted form for cookie matching
    base = domain.lstrip("www.")

    for loader_name, loader in [
        ("Chrome",  lambda: browser_cookie3.chrome(domain_name=f".{base}")),
        ("Firefox", lambda: browser_cookie3.firefox(domain_name=f".{base}")),
        ("Safari",  lambda: browser_cookie3.safari(domain_name=f".{base}")),
    ]:
        try:
            cj = loader()
            if any(True for _ in cj):
                print(f"[link_extractor] Using {loader_name} cookies for {base}")
                return cj
        except Exception:
            pass
    return None


def _cookies_for_url(url: str, portal_cookies: dict) -> dict:
    """Return a requests-compatible cookies dict for the given URL's domain."""
    domain = urlparse(url).netloc.lstrip("www.")
    for key, cookie_str in portal_cookies.items():
        if domain == key or domain.endswith("." + key):
            # Parse "name=value; name2=value2" into dict
            cookies = {}
            for pair in cookie_str.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    cookies[k.strip()] = v.strip()
            return cookies
    return {}


def _is_pdf_response(response: requests.Response) -> bool:
    ct = response.headers.get("content-type", "").lower()
    if any(ct.startswith(p) for p in _PDF_CONTENT_TYPES):
        if ct.startswith("application/octet-stream"):
            # Only count as PDF if Content-Disposition says .pdf or URL ends .pdf
            cd = response.headers.get("content-disposition", "")
            return ".pdf" in cd.lower() or ".pdf" in response.url.lower()
        return True
    # Fallback: check if the content starts with PDF magic bytes
    peek = response.content[:4] if response.content else b""
    return peek == b"%PDF"


def extract_links_from_html(html: str, base_url: str = "") -> List[str]:
    """
    Return a de-duplicated list of candidate report URLs extracted from email HTML.

    Filters out images, unsubscribe links, and other noise. Report-looking URLs
    are sorted to the front.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set = set()
    candidates: List[str] = []
    report_links: List[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href.startswith("http"):
            if base_url:
                href = urljoin(base_url, href)
            else:
                continue

        if href in seen:
            continue
        seen.add(href)

        if _SKIP_PATTERNS.search(href):
            continue

        if _REPORT_PATTERNS.search(href):
            report_links.append(href)
        else:
            candidates.append(href)

    # Report-pattern URLs first, then other candidates
    return report_links + candidates


def fetch_pdf_from_url(
    url: str,
    portal_cookies: Optional[dict] = None,
    timeout: tuple = (5, 10),
    max_depth: int = 1,
    _depth: int = 0,
    _extra_cookiejar=None,
) -> Optional[Tuple[str, bytes]]:
    """
    Follow ``url`` and return ``(filename, pdf_bytes)`` if a PDF is found,
    else ``None``.

    Handles:
    - Direct PDF responses.
    - HTML pages that contain a single dominant PDF download link.
    - One level of nested page parsing (depth-limited to avoid infinite loops).

    Args:
        url: URL to fetch.
        portal_cookies: Domain → cookie dict (from ``_load_portal_cookies()``).
        timeout: Request timeout in seconds.
        max_depth: Maximum number of HTML page hops before giving up.
        _depth: Internal recursion depth counter.
    """
    if _depth > max_depth:
        return None

    if portal_cookies is None:
        portal_cookies = _load_portal_cookies()

    cookies = _cookies_for_url(url, portal_cookies)

    try:
        session = requests.Session()
        if _extra_cookiejar is not None:
            session.cookies.update(_extra_cookiejar)
        try:
            resp = session.get(
                url,
                headers=_DEFAULT_HEADERS,
                cookies=cookies,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
        except requests.exceptions.SSLError:
            # Many Asian/European broker portals use private CAs not in macOS's bundle.
            # Retry once without verification — these are known-trusted broker domains.
            import warnings, urllib3
            warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            print(f"[link_extractor] SSL error — retrying with verify=False: {url[:80]}")
            resp = session.get(
                url,
                headers=_DEFAULT_HEADERS,
                cookies=cookies,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
                verify=False,
            )
        resp.raise_for_status()
    except requests.Timeout:
        print(f"[link_extractor] Timeout fetching {url[:80]}")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        # Auth-like errors: retry once with browser cookies before giving up
        if status in (401, 403, 500) and _depth == 0:
            browser_cj = _browser_cookiejar_for_url(url)
            if browser_cj is not None:
                return fetch_pdf_from_url(
                    url, portal_cookies=portal_cookies, timeout=timeout,
                    max_depth=max_depth, _depth=_depth + 1,
                    _extra_cookiejar=browser_cj,
                )
            print(f"[link_extractor] HTTP {status} — no browser cookies found for this domain: {url[:80]}")
        else:
            print(f"[link_extractor] HTTP {status} for {url[:80]}")
        return None
    except requests.RequestException as e:
        print(f"[link_extractor] Failed {url[:80]}: {type(e).__name__}")
        return None

    # ── Direct PDF ────────────────────────────────────────────────────────────
    if _is_pdf_response(resp):
        content = resp.content
        # Derive filename from URL or Content-Disposition
        cd = resp.headers.get("content-disposition", "")
        fname_match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', cd, re.IGNORECASE)
        filename = fname_match.group(1).strip() if fname_match else url.split("/")[-1].split("?")[0]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        # Reject compliance/legal documents by filename before Docling runs
        if _NON_RESEARCH_PDF_RE.search(filename):
            print(f"[link_extractor] Skipping non-research PDF (compliance/legal): {filename}")
            return None
        print(f"[link_extractor] Downloaded PDF: {filename} ({len(content):,} bytes) from {resp.url[:80]}")
        return filename, content

    # ── HTML page — look for PDF links inside ────────────────────────────────
    ct = resp.headers.get("content-type", "")
    if "text/html" not in ct:
        return None  # Not HTML, not PDF — give up

    # Read full body now (was streaming)
    html_content = resp.content.decode("utf-8", errors="replace")
    nested_links = extract_links_from_html(html_content, base_url=resp.url)

    # Try the most promising PDF links on this page
    for nested_url in nested_links[:5]:
        result = fetch_pdf_from_url(
            nested_url,
            portal_cookies=portal_cookies,
            timeout=timeout,
            max_depth=max_depth,
            _depth=_depth + 1,
        )
        if result:
            return result

    return None


def extract_pdfs_from_email(
    html_body: Optional[str],
    text_body: Optional[str] = None,
    portal_cookies: Optional[dict] = None,
    max_links: int = 5,
    global_timeout: int = _GLOBAL_TIMEOUT_SECS,
) -> List[Tuple[str, bytes]]:
    """
    Main entry point: extract all downloadable PDFs reachable from links in an email.

    Tries up to ``max_links`` candidate URLs in parallel, with a hard
    ``global_timeout`` wall-clock ceiling so the caller is never blocked longer
    than expected.

    Returns a list of ``(filename, pdf_bytes)`` tuples, one per downloaded PDF.
    Returns an empty list if no PDFs could be retrieved (auth required, dead links, etc.).

    Args:
        html_body: The email's HTML body.
        text_body: The email's plain-text body (fallback if HTML has no links).
        portal_cookies: Pre-loaded domain → cookie dict (loads from env if None).
        max_links: Cap on how many links to try concurrently.
        global_timeout: Maximum total seconds to spend on all link fetching.
    """
    if portal_cookies is None:
        portal_cookies = _load_portal_cookies()

    links: List[str] = []

    if html_body:
        links = extract_links_from_html(html_body)

    # Fallback: pull bare URLs from plain text
    if not links and text_body:
        links = re.findall(r"https?://[^\s\)>\"']+", text_body)
        links = [l for l in links if not _SKIP_PATTERNS.search(l)]

    if not links:
        return []

    candidate_urls = links[:max_links]
    print(f"[link_extractor] Trying {len(candidate_urls)} of {len(links)} candidate links")

    results: List[Tuple[str, bytes]] = []
    # Do NOT use `with pool:` — its __exit__ calls shutdown(wait=True) which
    # blocks until all threads finish, defeating the global_timeout.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidate_urls), 4))
    try:
        future_to_url = {
            pool.submit(fetch_pdf_from_url, url, portal_cookies): url
            for url in candidate_urls
        }
        done, _not_done = concurrent.futures.wait(
            future_to_url,
            timeout=global_timeout,
        )
        for f in done:
            try:
                pdf = f.result(timeout=0)
                if pdf:
                    results.append(pdf)
            except Exception:
                pass
        if _not_done:
            print(f"[link_extractor] {len(_not_done)} link(s) still pending after {global_timeout}s — abandoned")
    finally:
        # Abandon remaining threads without blocking
        pool.shutdown(wait=False, cancel_futures=True)

    return results
