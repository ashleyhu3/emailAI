"""
Hot-folder watcher: auto-ingest PDFs that land in ~/Downloads (or INGEST_HOT_FOLDER).

Starts a background watchdog thread when imported. Any .pdf file created or moved
into the watched directory is ingested via the same pipeline as /ingest/upload-pdf.

Broker is inferred from the filename using DOMAIN_TO_BROKER keyword matching;
falls back to HOT_FOLDER_DEFAULT_BROKER env var or "Unknown".

To disable: set INGEST_HOT_FOLDER='' in .env.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_PDF_SUMMARIZER = Path(__file__).resolve().parent.parent / "PDF_summarizer"
if str(_PDF_SUMMARIZER) not in sys.path:
    sys.path.insert(0, str(_PDF_SUMMARIZER))


# ── broker inference from filename ──────────────────────────────────────────

_FILENAME_BROKER_HINTS: list[tuple[str, str]] = [
    ("daiwa",    "Daiwa Capital Markets"),
    ("daiwacm",  "Daiwa Capital Markets"),
    ("gs_",      "Goldman Sachs"),
    ("goldman",  "Goldman Sachs"),
    ("ms_",      "Morgan Stanley"),
    ("morganst", "Morgan Stanley"),
    ("jpm",      "J.P. Morgan"),
    ("jpmorgan", "J.P. Morgan"),
    ("ubs",      "UBS"),
    ("barclays", "Barclays"),
    ("db_",      "Deutsche Bank"),
    ("deutsche", "Deutsche Bank"),
    ("citi",     "Citi"),
    ("clsa",     "CLSA"),
    ("nomura",   "Nomura"),
    ("macquarie","Macquarie"),
    ("bofa",     "Bank of America"),
    ("tfzq",     "天风证券"),
    ("tianfeng", "天风证券"),
    ("citics",   "中信证券"),
]


def _infer_broker(filename: str) -> str:
    lower = filename.lower()
    for hint, broker in _FILENAME_BROKER_HINTS:
        if hint in lower:
            return broker
    return os.getenv("HOT_FOLDER_DEFAULT_BROKER", "Unknown")


# ── recently-seen set (de-dup within one session) ──────────────────────────

_seen_hashes: set[str] = set()
_seen_lock = threading.Lock()


def _already_seen(pdf_bytes: bytes) -> bool:
    h = hashlib.sha256(pdf_bytes).hexdigest()
    with _seen_lock:
        if h in _seen_hashes:
            return True
        _seen_hashes.add(h)
        return False


# ── ingest a single PDF file ────────────────────────────────────────────────

def _ingest_pdf_file(path: Path) -> None:
    try:
        pdf_bytes = path.read_bytes()
    except Exception as e:
        print(f"[hot_folder] Cannot read {path.name}: {e}")
        return

    if pdf_bytes[:4] != b"%PDF":
        return  # not a PDF

    if _already_seen(pdf_bytes):
        return

    broker = _infer_broker(path.name)
    print(f"[hot_folder] New PDF detected: {path.name!r} → broker={broker!r}")

    try:
        from datetime import datetime, timezone
        from ingest.email_fetcher import EmailPayload
        from ingest.worker import _process_email

        payload = EmailPayload(
            message_id=hashlib.sha256(pdf_bytes).hexdigest(),
            sender=f"hot-folder@local",
            sender_domain="local",
            broker=broker,
            report_date=datetime.now(timezone.utc),
            html_body=None,
            text_body=None,
            pdf_attachments=[(path.name, pdf_bytes)],
        )
        result = _process_email(payload)
        print(f"[hot_folder] Ingested {path.name!r}: {result.get('status')} doc_id={result.get('document_id')}")
    except Exception as e:
        print(f"[hot_folder] Ingest failed for {path.name!r}: {e}")


# ── watchdog event handler ──────────────────────────────────────────────────

def _make_handler():
    from watchdog.events import FileSystemEventHandler

    class _PDFHandler(FileSystemEventHandler):
        def _handle(self, path_str: str) -> None:
            path = Path(path_str)
            if path.suffix.lower() != ".pdf":
                return
            # Give the browser a moment to finish writing the file
            time.sleep(1.5)
            if path.exists():
                _ingest_pdf_file(path)

        def on_created(self, event):
            if not event.is_directory:
                self._handle(event.src_path)

        def on_moved(self, event):
            # Browsers often write to a .crdownload/.part then rename to .pdf
            if not event.is_directory:
                self._handle(event.dest_path)

    return _PDFHandler()


# ── public entry point ──────────────────────────────────────────────────────

_watcher_started = False
_watcher_lock = threading.Lock()


def start_hot_folder_watcher() -> Optional[str]:
    """
    Start the background watchdog thread. Safe to call multiple times — only
    starts once. Returns the watched directory path, or None if disabled.
    """
    global _watcher_started

    hot_folder = os.getenv("INGEST_HOT_FOLDER", str(Path.home() / "Downloads"))
    if not hot_folder:
        return None  # explicitly disabled

    watch_path = Path(hot_folder).expanduser().resolve()
    if not watch_path.is_dir():
        print(f"[hot_folder] Watch directory does not exist, skipping: {watch_path}")
        return None

    with _watcher_lock:
        if _watcher_started:
            return str(watch_path)
        _watcher_started = True

    from watchdog.observers import Observer

    observer = Observer()
    observer.schedule(_make_handler(), str(watch_path), recursive=False)
    observer.daemon = True
    observer.start()
    print(f"[hot_folder] Watching {watch_path} for new PDFs")
    return str(watch_path)
