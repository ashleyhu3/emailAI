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
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
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
    r"morganstanley\.com/matrix/?$|"                 # MS Matrix homepage — not a report
    r"ms\.email\.streetcontxt\.net|"                 # MS StreetContxt one-time tracking links — always fail automated; MS Matrix fallback handles these
    r"linkback\.morganstanley\.com|"                 # MS weblink tracking redirects — always 403 without browser session
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
    r"researchfn\.com/c/|/view|portal|access|content|publication|"
    r"matrix\.ms\.com|t\.cm\.morganstanley\.com)",  # MS Matrix portal and tracking links
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


# Domains whose cookies should be loaded into the headless browser to handle
# MS research links: StreetContxt delivery → Matrix portal → PDF download.
_BROKER_COOKIE_DOMAINS = [
    ".morganstanley.com",
    ".ms.com",
    ".streetcontxt.net",
    ".gs.com",
    ".goldmansachs.com",
    ".jpmorgan.com",
    ".barclays.com",
    ".db.com",
    ".ubs.com",
    ".jefferies.com",
]

# Arc browser profile paths (Chromium-based, uses "Arc Safe Storage" keychain entry)
_ARC_COOKIE_PATHS = [
    os.path.expanduser("~/Library/Application Support/Arc/User Data/Default/Cookies"),
    os.path.expanduser("~/Library/Application Support/Arc/User Data/Profile 1/Cookies"),
]


def _decrypt_arc_cookies() -> list:
    """
    Read and decrypt Arc browser cookies for broker domains.

    Arc uses the same AES-128-CBC encryption as Chrome but stores the key
    under "Arc Safe Storage" in the macOS Keychain.
    """
    try:
        from Crypto.Cipher import AES
    except ImportError:
        try:
            subprocess.run(
                ["pip", "install", "pycryptodome", "-q"],
                capture_output=True,
            )
            from Crypto.Cipher import AES
        except Exception:
            return []

    # Retrieve Arc's encryption key from macOS Keychain
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "Arc Safe Storage", "-w"],
        capture_output=True, text=True,
    )
    password = result.stdout.strip().encode()
    if not password:
        return []

    key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
    iv = b" " * 16

    def _decrypt(enc: bytes) -> str:
        data = enc[3:] if enc[:3] in (b"v10", b"v11") else enc
        # Arc (Chromium) prepends 32 bytes of random nonce after the v10/v11 prefix.
        # The CBC IV for the actual payload is bytes 16-32 (the second nonce block).
        if len(data) > 32:
            nonce_iv = data[16:32]
            payload = data[32:]
        else:
            nonce_iv = iv
            payload = data
        cipher = AES.new(key, AES.MODE_CBC, IV=nonce_iv)
        dec = cipher.decrypt(payload)
        pad = dec[-1]
        return dec[: -pad if 1 <= pad <= 16 else None].decode("utf-8", errors="replace")

    cookies: list = []
    for path in _ARC_COOKIE_PATHS:
        if not os.path.exists(path):
            continue
        tmp = tempfile.mktemp(suffix=".db")
        try:
            shutil.copy(path, tmp)
            conn = sqlite3.connect(tmp)
            conn.text_factory = bytes
            rows = conn.execute(
                "SELECT host_key, name, encrypted_value, value, path, is_secure, expires_utc "
                "FROM cookies"
            ).fetchall()
            conn.close()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        for host, name, enc_val, plain_val, cookie_path, secure, expires in rows:
            host_str = host.decode("utf-8", errors="replace")
            # Match only exact broker domain suffixes (avoid e.g. jobs.ubs.com matching .ubs.com)
            if not any(
                host_str == d.lstrip(".") or host_str.endswith(d if d.startswith(".") else f".{d}")
                for d in _BROKER_COOKIE_DOMAINS
            ):
                continue
            try:
                value = _decrypt(enc_val) if enc_val else plain_val.decode("utf-8", errors="replace")
            except Exception:
                continue
            cookie: dict = {
                "name": name.decode("utf-8", errors="replace"),
                "value": value,
                "domain": host_str if host_str.startswith(".") else f".{host_str}",
                "path": cookie_path.decode("utf-8", errors="replace") if cookie_path else "/",
                "secure": bool(secure),
            }
            # Chrome epoch: microseconds since 1601-01-01; convert to Unix seconds
            if expires and expires > 0:
                unix_ts = (expires / 1_000_000) - 11_644_473_600
                if unix_ts > 0:
                    cookie["expires"] = unix_ts
            cookies.append(cookie)

    return cookies


def _fetch_pdf_with_playwright(url: str, timeout_ms: int = 30_000) -> Optional[Tuple[str, bytes]]:
    """
    Launch a headless Chromium browser with Arc browser cookies injected,
    navigate to ``url``, and capture any PDF that gets downloaded or served.

    Returns ``(filename, pdf_bytes)`` or ``None``.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return None

    raw_cookies = _decrypt_arc_cookies()

    # Convert to Playwright format: use url instead of domain to avoid CDP conflicts.
    # Strip leading dot from domain to build a valid https URL.
    def _to_pw_cookie(c: dict) -> Optional[dict]:
        name = c["name"]
        # Skip names with characters invalid in cookie names (CDP rejects them)
        if any(ch in name for ch in ("/",)):
            return None
        # Use domain + path format to preserve subdomain scope (.matrix.ms.com etc.)
        # Playwright accepts domain with leading dot for subdomain matching.
        cookie: dict = {
            "name": name,
            "value": c["value"],
            "domain": c["domain"],
            "path": "/",
        }
        if "expires" in c:
            cookie["expires"] = c["expires"]
        return cookie

    all_cookies = [pw for c in raw_cookies if (pw := _to_pw_cookie(c)) is not None]
    print(f"[playwright] Launching headless browser for {url[:80]} "
          f"({len(all_cookies)} Arc broker cookies loaded)")

    pdf_data: list = []  # mutable capture for download handler

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(accept_downloads=True)
            if all_cookies:
                good, bad = 0, 0
                for c in all_cookies:
                    try:
                        ctx.add_cookies([c])
                        good += 1
                    except Exception:
                        bad += 1
                if bad:
                    print(f"[playwright] Cookies injected: {good} ok, {bad} skipped")

            page = ctx.new_page()

            def on_download(dl):
                try:
                    path = dl.path()
                    if path:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        if data[:4] == b"%PDF":
                            name = dl.suggested_filename or "report.pdf"
                            pdf_data.append((name, data))
                            print(f"[playwright] Captured download: {name} ({len(data):,} bytes)")
                except Exception as e2:
                    print(f"[playwright] Download capture error: {e2}")

            page.on("download", on_download)

            try:
                resp = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            except PWTimeout:
                print(f"[playwright] Page load timed out: {url[:80]}")
                ctx.close(); browser.close()
                return None

            # Matrix SPA takes ~12s to hydrate after DOMContentLoaded;
            # wait for JS to render and any auto-downloads to trigger.
            page.wait_for_timeout(15_000)

            # If a download was captured, return it
            if pdf_data:
                ctx.close(); browser.close()
                return pdf_data[0]

            # Check if the final page itself is a PDF (content-type or magic bytes)
            if resp and "pdf" in (resp.headers.get("content-type") or "").lower():
                body = page.evaluate("() => document.body?.innerText || ''")
                # body is text-decoded; we need raw bytes — fall through
                pass

            # Look for a direct PDF <a> or <embed> link on the loaded page
            final_url = page.url
            pdf_link = page.evaluate("""() => {
                const a = document.querySelector('a[href$=".pdf"], a[href*="/download"], a[href*="pdf"]');
                return a ? a.href : null;
            }""")
            if pdf_link:
                print(f"[playwright] Found PDF link on page: {pdf_link[:80]}")
                try:
                    with page.expect_download(timeout=timeout_ms) as dl_info:
                        page.goto(pdf_link, timeout=timeout_ms)
                    dl = dl_info.value
                    path = dl.path()
                    if path:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        if data[:4] == b"%PDF":
                            name = dl.suggested_filename or "report.pdf"
                            ctx.close(); browser.close()
                            return name, data
                except Exception:
                    pass

            ctx.close()
            browser.close()
    except Exception as e:
        print(f"[playwright] Error: {e}")

    return None


def fetch_ms_matrix_pdf(
    company_name: str,
    report_date: Optional[str] = None,
    timeout_ms: int = 45_000,
) -> Optional[Tuple[str, bytes]]:
    """
    Search MS Matrix research feed for today's report on *company_name* and
    download its PDF using the authenticated Arc browser session.

    This bypasses StreetContxt one-time tracking links by querying the Matrix
    feed API directly, which requires only the user's active Matrix session
    cookies (read from Arc).

    Args:
        company_name: Company ticker or name to match (case-insensitive substring).
        report_date: Optional ISO date string (YYYY-MM-DD) to restrict matches.
        timeout_ms: Playwright navigation timeout in milliseconds.

    Returns:
        ``(filename, pdf_bytes)`` or ``None``.
    """
    import base64

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return None

    cookies = _decrypt_arc_cookies()
    if not cookies:
        return None

    def _to_pw(c: dict) -> Optional[dict]:
        if "/" in c["name"]:
            return None
        cookie: dict = {"name": c["name"], "value": c["value"],
                        "domain": c["domain"], "path": "/"}
        if "expires" in c:
            cookie["expires"] = c["expires"]
        return cookie

    pw_cookies = [pw for c in cookies if (pw := _to_pw(c)) is not None]

    company_lower = company_name.lower()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(accept_downloads=True)
            for c in pw_cookies:
                try:
                    ctx.add_cookies([c])
                except Exception:
                    pass

            page = ctx.new_page()
            feed_payload: list = []

            def _capture_feed(resp):
                if resp.url.endswith("/feed") and "research-feed" in resp.url:
                    try:
                        import json as _json
                        feed_payload.append(_json.loads(resp.body()))
                    except Exception:
                        pass

            page.on("response", _capture_feed)

            # Load the research feed page to trigger the feed API call
            print(f"[ms_matrix] Loading research feed to find '{company_name}'")
            try:
                page.goto(
                    "https://ny.matrix.ms.com/eqr/research/portal/feed",
                    timeout=timeout_ms,
                    wait_until="domcontentloaded",
                )
            except PWTimeout:
                pass
            page.wait_for_timeout(8_000)

            if not feed_payload:
                print("[ms_matrix] No feed data captured")
                ctx.close(); browser.close()
                return None

            # Find matching articles by company name (and optional date)
            all_cards = feed_payload[0].get("all", {}).get("feedCards", [])
            matches = [
                c for c in all_cards
                if company_lower in c.get("hl", "").lower()
                and (not report_date or report_date in c.get("pd", ""))
            ]

            if not matches:
                print(f"[ms_matrix] No feed articles matched '{company_name}'")
                ctx.close(); browser.close()
                return None

            article = matches[0]
            article_path = article.get("reportUrl", "")
            article_url = f"https://ny.matrix.ms.com{article_path}"
            print(f"[ms_matrix] Found: {article.get('hl','')[:60]} → {article_url[:80]}")

            # Navigate to the article to find the PDF rendition URL
            try:
                page.goto(article_url, timeout=timeout_ms, wait_until="domcontentloaded")
            except PWTimeout:
                pass
            page.wait_for_timeout(8_000)

            pdf_href = page.evaluate("""() => {
                const a = document.querySelector('a[href*="/rendition/pdf/"]');
                return a ? a.href : null;
            }""")

            if not pdf_href:
                print("[ms_matrix] No PDF rendition link found on article page")
                ctx.close(); browser.close()
                return None

            print(f"[ms_matrix] PDF URL: {pdf_href[:80]}")

            # Download via JS fetch (stays in authenticated session)
            result = page.evaluate(f"""async () => {{
                const resp = await fetch({repr(pdf_href)}, {{credentials:'include'}});
                if (!resp.ok) return {{error: resp.status + ' ' + resp.statusText}};
                const buf = await resp.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let bin = '';
                bytes.forEach(b => bin += String.fromCharCode(b));
                return {{b64: btoa(bin), size: buf.byteLength}};
            }}""")

            ctx.close(); browser.close()

            if "error" in result:
                print(f"[ms_matrix] PDF fetch failed: {result['error']}")
                return None

            pdf_bytes = base64.b64decode(result["b64"])
            if pdf_bytes[:4] != b"%PDF":
                print("[ms_matrix] Downloaded content is not a PDF")
                return None

            filename = pdf_href.split("/")[-1].split("?")[0]
            print(f"[ms_matrix] Downloaded PDF: {filename} ({len(pdf_bytes):,} bytes)")
            return filename, pdf_bytes

    except Exception as e:
        print(f"[ms_matrix] Error: {e}")
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
        # Pre-load ALL configured portal cookies into the session so that
        # cross-domain redirects carry authentication (e.g. MS tracking links
        # t.cm.morganstanley.com → ny.matrix.ms.com need ms.com cookies).
        for _pck_str in portal_cookies.values():
            for _pair in _pck_str.split(";"):
                _pair = _pair.strip()
                if "=" in _pair:
                    _k, _v = _pair.split("=", 1)
                    session.cookies.set(_k.strip(), _v.strip())
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
        if _depth == 0:
            result = _fetch_pdf_with_playwright(url)
            if result:
                return result
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if _depth == 0:
            # Try headless browser first (handles auth portals + JS redirects)
            result = _fetch_pdf_with_playwright(url)
            if result:
                return result
            # Then fall back to plain browser-cookie injection
            browser_cj = _browser_cookiejar_for_url(url)
            if browser_cj is not None:
                return fetch_pdf_from_url(
                    url, portal_cookies=portal_cookies, timeout=timeout,
                    max_depth=max_depth, _depth=_depth + 1,
                    _extra_cookiejar=browser_cj,
                )
            print(f"[link_extractor] HTTP {status} — no browser session found: {url[:80]}")
        else:
            print(f"[link_extractor] HTTP {status} for {url[:80]}")
        return None
    except requests.RequestException as e:
        print(f"[link_extractor] Failed {url[:80]}: {type(e).__name__}")
        if _depth == 0:
            result = _fetch_pdf_with_playwright(url)
            if result:
                return result
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
