"""
Phase 1–4 Worker: full ingestion pipeline as Celery tasks.

Celery Beat fires twice daily (07:00 and 18:00 UTC).
Each run: fetch new emails → prioritize → preprocess → extract metadata → store to PostgreSQL.

To start (requires Redis):
  # Terminal 1 — worker
  celery -A PDF_summarizer.ingest.worker worker --loglevel=info -c 4

  # Terminal 2 — beat scheduler
  celery -A PDF_summarizer.ingest.worker beat --loglevel=info

Manual trigger (no Celery needed):
  from PDF_summarizer.ingest.worker import run_ingest_now
  run_ingest_now()
"""

import hashlib
import io
import os
import pickle
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Resolve sibling imports
_PDF_SUMMARIZER = Path(__file__).resolve().parent.parent
if str(_PDF_SUMMARIZER) not in sys.path:
    sys.path.insert(0, str(_PDF_SUMMARIZER))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

from database import DatabaseManager
from broker_cache import BrokerContextCache
from ingest.email_fetcher import fetch_broker_emails, load_config, EmailPayload
from ingest.preprocessor import clean_html_to_aoim, slice_pdf_pages_to_aoim, full_pdf_to_rag_chunks, email_html_to_rag_chunks
from ingest.extraction_agents import run_agent1, extract_metadata, FinancialReportMetadata
from ingest.link_extractor import extract_pdfs_from_email, fetch_ms_matrix_pdf
from ingest.extractor import (
    extract_fields_deterministically,
    extract_relevant_aoim_sections,
    figure_needs_vision,
    is_likely_research_report,
    is_research_pdf_content,
)

# ── Celery app (optional — not needed for synchronous / API-driven ingestion) ─
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery
    from celery.schedules import crontab

    celery_app = Celery("ingest_worker", broker=REDIS_URL, backend=REDIS_URL)
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )
    celery_app.conf.beat_schedule = {
        "ingest-emails-morning": {
            "task": "PDF_summarizer.ingest.worker.ingest_email_batch",
            "schedule": crontab(hour=7, minute=0),
        },
        "ingest-emails-evening": {
            "task": "PDF_summarizer.ingest.worker.ingest_email_batch",
            "schedule": crontab(hour=18, minute=0),
        },
    }
    _CELERY_AVAILABLE = True
except ImportError:
    celery_app = None  # type: ignore[assignment]
    _CELERY_AVAILABLE = False


# ── Shared singletons (initialized lazily per-worker-process) ─────────────────
_db: Optional[DatabaseManager] = None
_broker_cache: Optional[BrokerContextCache] = None


def _get_db() -> DatabaseManager:
    global _db
    if _db is None:
        db_url = os.environ["PDF_SUMMARIZER_DB_URL"]
        _db = DatabaseManager(database_url=db_url)
    return _db


def _get_broker_cache() -> BrokerContextCache:
    global _broker_cache
    if _broker_cache is None:
        _broker_cache = BrokerContextCache()
    return _broker_cache


# ── Docling output cache (file-based, keyed by PDF content hash) ──────────────
# Avoids re-running the CPU-heavy Docling pipeline on the same PDF across emails.
# The same research report is often mass-distributed to thousands of recipients;
# without this cache every copy would re-slice the identical PDF bytes.

_DOCLING_CACHE = _PDF_SUMMARIZER / ".cache" / "docling"


def _cache_path(key: str, prefix: str) -> Path:
    _DOCLING_CACHE.mkdir(parents=True, exist_ok=True)
    return _DOCLING_CACHE / f"{prefix}_{key}.pkl"


def _cached_slice_pdf(pdf_bytes: bytes, page_range: tuple):
    """Return (aoim_text, figures) from cache or compute via Docling."""
    key = hashlib.md5(pdf_bytes + str(page_range).encode()).hexdigest()
    path = _cache_path(key, "aoim")
    if path.exists():
        try:
            with open(path, "rb") as f:
                print(f"[ingest] Docling AOIM cache hit ({len(pdf_bytes):,} bytes)")
                return pickle.load(f)
        except Exception:
            pass

    # Retry once on cold-start ML model initialization race
    try:
        result = slice_pdf_pages_to_aoim(pdf_bytes, page_range=page_range)
    except Exception as first_err:
        print(f"[ingest] Docling first attempt failed: {first_err} — retrying")
        result = slice_pdf_pages_to_aoim(pdf_bytes, page_range=page_range)

    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass
    return result


def _cached_full_pdf_chunks(pdf_bytes: bytes, filename: str) -> list:
    """Return RAG chunks from cache or compute via full_pdf_to_rag_chunks."""
    key = hashlib.md5(pdf_bytes + filename.encode()).hexdigest()
    path = _cache_path(key, "chunks")
    if path.exists():
        try:
            with open(path, "rb") as f:
                print(f"[ingest] RAG chunk cache hit for {filename}")
                return pickle.load(f)
        except Exception:
            pass

    result = full_pdf_to_rag_chunks(pdf_bytes, filename=filename)

    if result:
        try:
            with open(path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    return result or []


# ── Priority scoring ──────────────────────────────────────────────────────────
# Emails are sorted by priority before processing so high-value reports (upgrades
# from top-tier brokers) complete before lower-priority maintenance notes, even
# during back-to-back bursts.

_BROKER_TIER: dict = {
    # Tier 3 — bulge-bracket (most likely to move markets; process first)
    "goldman sachs": 3, "j.p. morgan": 3, "morgan stanley": 3,
    "ubs": 3, "barclays": 3, "bank of america": 3, "citi": 3,
    "deutsche bank": 3, "credit suisse": 3, "jefferies": 3,
    # Tier 2 — regional specialists
    "nomura": 2, "daiwa capital markets": 2, "clsa": 2, "macquarie": 2,
    "cgscimb": 2, "boci": 2, "hsbc": 2, "standard chartered": 2,
    "wolfe research": 2, "bernstein": 2, "btig": 2, "ccb international": 2,
}

_UPGRADE_RE = re.compile(r'\b(upgrad|downgrad|initiat)\b', re.I)


def _email_priority_score(payload: "EmailPayload") -> float:
    """
    Score an email so the batch can be sorted highest-priority first.

    Factors (additive):
    - Broker tier:    +3 (tier-1 bulge), +2 (tier-2 regional), +1 (others)
    - PDF attached:   +1.0 (research PDFs > digest text emails)
    - Recency:        +0.0 to +1.0 (linear decay over 7 days)
    - Action signal:  +0.5 (upgrade/downgrade/initiation in subject)
    """
    broker_lower = (payload.broker or "").lower()
    score = float(_BROKER_TIER.get(broker_lower, 1))

    if payload.pdf_attachments:
        score += 1.0

    if payload.report_date:
        try:
            age_days = (datetime.utcnow() - payload.report_date.replace(tzinfo=None)).days
            score += max(0.0, 1.0 - age_days / 7.0)
        except Exception:
            pass

    if _UPGRADE_RE.search(payload.subject or ""):
        score += 0.5

    return score


# ── Phase 1: fetch + deduplicate ──────────────────────────────────────────────

def _extract_sender_email(sender: str) -> str:
    """Extract bare email address from 'Display Name <email@domain>' or raw address."""
    if "<" in sender and ">" in sender:
        return sender.split("<")[-1].strip(">").strip().lower()
    return sender.strip().lower()


def _fetch_new_emails() -> List[EmailPayload]:
    db = _get_db()
    from sqlalchemy import text as sa_text
    session = db.get_session()
    try:
        rows = session.execute(
            sa_text("SELECT email_message_id FROM pdf_documents WHERE email_message_id IS NOT NULL")
        ).fetchall()
        existing_ids = {r[0] for r in rows}
    finally:
        session.close()

    # Also skip emails we already rejected — avoids re-triaging the same non-research
    # emails on every scheduled run (they stay unread in Gmail).
    existing_ids |= db.get_rejected_message_ids()

    # Prefer Gmail API (OAuth2) when credentials are configured — no app password needed.
    # Falls back to IMAP when Gmail API is not set up.
    try:
        from ingest.gmail_fetcher import is_available, fetch_broker_emails_gmail
        if is_available():
            print("[ingest] Using Gmail API (OAuth2)")
            return fetch_broker_emails_gmail(existing_ids=existing_ids)
    except ImportError:
        pass

    # IMAP fallback
    cfg = load_config()
    if not cfg["username"]:
        print("[ingest] No email source configured (set GMAIL_CREDENTIALS_FILE or IMAP_USER)")
        return []

    print("[ingest] Using IMAP")
    return fetch_broker_emails(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password=cfg["password"],
        broker_domains=cfg["broker_domains"],
        existing_ids=existing_ids,
        inbox=cfg["inbox"],
    )


# ── Phase 2 + 3 + 4: preprocess → extract → store ────────────────────────────

def _process_email(payload: EmailPayload) -> dict:
    """
    Full optimised ingestion pipeline for a single email.

    Optimisations vs the naive pipeline:
    - Report relevance triage before any expensive work (non-reports dropped early).
    - Deterministic field extraction from email text (zero LLM cost).
    - Docling AOIM + chunk output cached by PDF content hash (no re-slicing same PDF).
    - Agent-2 receives only high-signal AOIM sections (~60% input token reduction).
    - Agent-1 skipped for figures already captured as Markdown tables by Docling.
    - 4-stage repair chain: rule engine → Flash self-correction → Agent 3 (Pro).
    - Chunk content deduplication before embedding (skips boilerplate re-embedding).
    """
    db = _get_db()
    cache = _get_broker_cache()

    # ── 0. Early dedup: skip before any work ─────────────────────────────────
    existing = db.get_document_by_email_id(payload.message_id)
    if existing:
        return {"status": "skipped", "reason": "duplicate", "message_id": payload.message_id}

    sender_email = _extract_sender_email(payload.sender)

    # ── 0b. Sender reputation check ───────────────────────────────────────────
    # Senders with 3+ decisions and zero acceptances are auto-blocked without
    # running the full triage — they've never produced research and are wasting cycles.
    reputation = db.get_sender_reputation(sender_email)
    if reputation["auto_block"]:
        print(f"[ingest] Auto-blocked (sender reputation: {reputation['rejected']} rejections, 0 accepted): {sender_email}")
        db.log_triage_decision(
            message_id=payload.message_id,
            sender_email=sender_email,
            sender_domain=payload.sender_domain,
            broker=payload.broker,
            subject=payload.subject,
            decision="rejected",
            rejection_reason="sender_reputation",
        )
        return {"status": "skipped", "reason": "sender reputation: auto-blocked", "message_id": payload.message_id}

    # ── 1. Report relevance triage ────────────────────────────────────────────
    # Drop non-report emails (trade confirmations, newsletters, scheduling) before
    # invoking the heavy PDF + LLM pipeline.
    passed, triage_reason = is_likely_research_report(
        subject=payload.subject,
        text_body=payload.text_body,
        has_pdf=bool(payload.pdf_attachments),
        from_known_broker=True,  # already passed domain whitelist — trust the sender
    )
    if not passed:
        print(f"[ingest] Triage rejected ({triage_reason}): {sender_email} — {payload.subject[:60]}")
        db.log_triage_decision(
            message_id=payload.message_id,
            sender_email=sender_email,
            sender_domain=payload.sender_domain,
            broker=payload.broker,
            subject=payload.subject,
            decision="rejected",
            rejection_reason=triage_reason,
        )
        return {"status": "skipped", "reason": triage_reason, "message_id": payload.message_id}

    # ── 2. Deterministic field extraction (zero token cost) ───────────────────
    # Run regex-based extraction on the email body before any PDF slicing.
    # The extracted fields are prepended to the AOIM as hints for Agent 2, reducing
    # the fields it needs to generate and cutting output tokens.
    source_text = f"{payload.text_body or ''} {(payload.html_body or '')[:1000]}"
    pre_extracted = extract_fields_deterministically(source_text, subject=payload.subject)
    if pre_extracted.confident_fields:
        print(f"[ingest] Deterministic extraction: {pre_extracted.confident_fields}")

    # ── 3. Build AOIM from HTML / text body ──────────────────────────────────
    aoim_parts = []
    if payload.html_body:
        aoim_parts.append(clean_html_to_aoim(payload.html_body))
    elif payload.text_body:
        aoim_parts.append(payload.text_body[:4000])

    # ── 4. Follow HTML links to download linked PDFs (no attachment case) ────
    if not payload.pdf_attachments:
        try:
            linked_pdfs = extract_pdfs_from_email(
                html_body=payload.html_body,
                text_body=payload.text_body,
            )
            if linked_pdfs:
                print(f"[ingest] Downloaded {len(linked_pdfs)} PDF(s) from email links")
                payload.pdf_attachments.extend(linked_pdfs)
        except Exception as e:
            print(f"[ingest] Link extraction failed: {e}")

    # ── 4b. MS Matrix fallback: search feed for matching article + download PDF ─
    # StreetContxt links in MS emails are one-time-use and often time out before
    # our pipeline can follow them.  When no PDF was captured, query Matrix
    # directly using Arc's authenticated session cookies.
    if not payload.pdf_attachments and (payload.broker or "").lower() == "morgan stanley":
        import re as _re
        import datetime as _dt
        # Strip common email prefixes (FW:, RE:, IDEA:, FWD: etc.) then match
        # the first colon-delimited segment as company name.
        cleaned_subj = _re.sub(
            r"^(?:(?:fw|fwd|re|idea|note|update|flash|alert)\s*:\s*)+",
            "",
            (payload.subject or ""),
            flags=_re.IGNORECASE,
        ).strip()
        m = _re.match(r"([^:]+):", cleaned_subj)
        company = m.group(1).strip() if m else None
        if company:
            today = _dt.date.today().isoformat()
            # Use deterministic date if extracted, else today
            report_date = str(pre_extracted.report_date) if pre_extracted.report_date else today
            try:
                ms_pdf = fetch_ms_matrix_pdf(company, report_date=report_date)
                if ms_pdf:
                    print(f"[ingest] MS Matrix PDF fetched via feed search: {ms_pdf[0]}")
                    payload.pdf_attachments.append(ms_pdf)
            except Exception as e:
                print(f"[ingest] MS Matrix feed search failed: {e}")

    # ── 5. Process PDF attachments ────────────────────────────────────────────
    for pdf_filename, pdf_bytes in payload.pdf_attachments:
        try:
            # Slice pages 1–2 via Docling (cached by PDF content hash)
            pdf_aoim, figures = _cached_slice_pdf(pdf_bytes, page_range=(1, 2))

            # Content-based filter: reject compliance/legal PDFs that slipped past
            # the filename filter (e.g. numbered filenames like "528550.pdf")
            if not is_research_pdf_content(pdf_aoim, filename=pdf_filename):
                print(f"[ingest] PDF rejected as non-research content: {pdf_filename}")
                continue

            aoim_parts.append(f"\n\n[PDF: {pdf_filename}]\n{pdf_aoim}")

            # Agent 1: only for figures that aren't already captured as text tables
            for fig in figures:
                if not fig.get("image"):
                    continue
                caption = fig.get("caption", "")
                surrounding = fig.get("surrounding_text", caption)
                if not figure_needs_vision(caption, surrounding):
                    print(f"[ingest] Agent-1 skipped for figure on page {fig['page']} (text table detected)")
                    continue
                chart_md = run_agent1(image_bytes=fig["image"], context_text=caption)
                if chart_md != "[NO_DATA]":
                    aoim_parts.append(f"\n[Chart/Table from page {fig['page']}]\n{chart_md}")

        except Exception as e:
            print(f"[ingest] PDF preprocessing failed ({pdf_filename}): {e}")

    # ── 6. Target AOIM sections + prepend deterministic hints ─────────────────
    raw_aoim = "\n\n".join(p for p in aoim_parts if p).strip()
    if not raw_aoim:
        return {"status": "skipped", "reason": "no extractable content", "message_id": payload.message_id}

    # Trim AOIM to high-signal sections (~60% input token reduction for Agent 2)
    targeted_aoim = extract_relevant_aoim_sections(raw_aoim)

    # Prepend deterministic hints so Agent 2 can confirm/fill remaining fields
    hint_block = pre_extracted.as_hint_block()
    aoim_for_agent2 = f"{hint_block}\n\n{targeted_aoim}" if hint_block else targeted_aoim

    # ── 7. Agent 2 → repair chain → Agent 3 ──────────────────────────────────
    # Prepend broker as a locked hint — domain mapping is always more reliable
    # than content extraction for digest emails that mention many institutions.
    broker_hint = f"broker: {payload.broker} (LOCKED — from sender domain, do not change)\n\n"
    aoim_for_agent2 = broker_hint + aoim_for_agent2

    try:
        metadata: FinancialReportMetadata = extract_metadata(
            aoim_text=aoim_for_agent2,
            broker=payload.broker,
            cache=cache,
        )
    except Exception as e:
        return {"status": "error", "reason": f"metadata extraction failed: {e}", "message_id": payload.message_id}

    # Enforce broker from domain whitelist — authoritative over content extraction.
    # Digest emails mention many institutions; Agent 2 can extract the wrong one.
    # Only override when payload.broker is a real known value, not a fallback like "Unknown".
    if payload.broker not in ("Unknown", "", None) and metadata.broker != payload.broker:
        print(f"[ingest] Broker override: '{metadata.broker}' → '{payload.broker}' (domain mapping wins)")
        metadata = metadata.model_copy(update={"broker": payload.broker})

    # ── 8. Store to PostgreSQL ────────────────────────────────────────────────
    filename = f"email_{payload.message_id[:16]}.eml"
    sent_date = payload.report_date.date() if payload.report_date else None

    try:
        eps_pe_dict = metadata.eps_pe.model_dump(exclude_none=True) if metadata.eps_pe else None
        doc = db.add_document(
            filename=filename,
            file_path="email://inbox",
            total_pages=0,
            file_size_bytes=len(raw_aoim.encode()),
            file_hash=payload.message_id,
            sender_name=payload.sender,
            sender_company=payload.broker,
            sent_date=sent_date,
            written_date=metadata.report_date,
            tickers=metadata.tickers or [],
            broker=metadata.broker,
            broker_action=metadata.broker_action,
            rating=metadata.rating,
            target_price=metadata.target_price,
            eps_pe=eps_pe_dict,
            dense_summary=metadata.dense_summary,
            email_message_id=payload.message_id,
            ingest_source="email",
        )
    except Exception as e:
        if "unique" in str(e).lower():
            return {"status": "skipped", "reason": "duplicate", "message_id": payload.message_id}
        return {"status": "error", "reason": str(e), "message_id": payload.message_id}

    # ── 9. RAG chunks + content dedup + embeddings ────────────────────────────
    any_chunks_added = False
    seen_chunk_hashes: set = set()   # in-run dedup: skip identical paragraphs (boilerplate)

    for pdf_filename, pdf_bytes in payload.pdf_attachments:
        try:
            chunks = _cached_full_pdf_chunks(pdf_bytes, pdf_filename)

            unique_chunks = []
            for chunk in chunks:
                content_hash = hashlib.md5(
                    (chunk.get("raw_content") or "").encode()
                ).hexdigest()
                if content_hash not in seen_chunk_hashes:
                    seen_chunk_hashes.add(content_hash)
                    unique_chunks.append(chunk)

            skipped = len(chunks) - len(unique_chunks)
            if skipped:
                print(f"[ingest] Chunk dedup: skipped {skipped} duplicate paragraph(s) in {pdf_filename}")

            if unique_chunks:
                db.add_chunks(doc.id, unique_chunks)
                any_chunks_added = True

        except Exception as e:
            print(f"[ingest] RAG chunking failed ({pdf_filename}): {e}")

    # Fallback: no PDFs → chunk email HTML so digest content is searchable via RAG
    if not any_chunks_added and payload.html_body:
        try:
            email_chunks = email_html_to_rag_chunks(payload.html_body, broker=metadata.broker)
            unique_email_chunks = []
            for chunk in email_chunks:
                content_hash = hashlib.md5(
                    (chunk.get("raw_content") or "").encode()
                ).hexdigest()
                if content_hash not in seen_chunk_hashes:
                    seen_chunk_hashes.add(content_hash)
                    unique_email_chunks.append(chunk)
            if unique_email_chunks:
                db.add_chunks(doc.id, unique_email_chunks)
                any_chunks_added = True
                print(f"[ingest] Email HTML chunked: {len(unique_email_chunks)} section(s) for doc {doc.id}")
        except Exception as e:
            print(f"[ingest] Email HTML chunking failed for doc {doc.id}: {e}")

    if any_chunks_added:
        try:
            import sys as _sys
            _backend = str(_PDF_SUMMARIZER.parent / "backend")
            if _backend not in _sys.path:
                _sys.path.insert(0, _backend)
            from rag_gemini import GeminiRAGPipeline
            rag = GeminiRAGPipeline(db=db)
            embedded = rag.backfill_embeddings()
            print(f"[ingest] Embedded {embedded} chunk(s) for doc {doc.id}")
        except Exception as e:
            print(f"[ingest] Embedding failed for doc {doc.id}: {e}")

    # Log successful acceptance so this sender builds up a trust history.
    db.log_triage_decision(
        message_id=payload.message_id,
        sender_email=sender_email,
        sender_domain=payload.sender_domain,
        broker=metadata.broker,
        subject=payload.subject,
        decision="accepted",
        doc_id=doc.id,
    )

    return {
        "status": "success",
        "document_id": doc.id,
        "broker": metadata.broker,
        "broker_action": metadata.broker_action,
        "rating": metadata.rating,
        "target_price": metadata.target_price,
        "message_id": payload.message_id,
    }


# ── Celery tasks (only registered when celery is installed) ───────────────────

if _CELERY_AVAILABLE and celery_app is not None:
    @celery_app.task(name="PDF_summarizer.ingest.worker.ingest_email_batch", bind=True, max_retries=2)
    def ingest_email_batch(self):
        """Scheduled task: fetch new broker emails and dispatch per-email processing.

        Emails are sorted by priority before dispatch so high-value reports are picked
        up first by available workers. Celery task priority (0–9) mirrors the score.
        """
        try:
            emails = _fetch_new_emails()
            emails.sort(key=_email_priority_score, reverse=True)
            print(f"[ingest_email_batch] Dispatching {len(emails)} emails (priority-sorted)")
            for payload in emails:
                # Map float score (1.0–6.0) to Celery priority int (0=low, 9=high).
                raw_score = _email_priority_score(payload)
                celery_priority = min(9, int((raw_score - 1.0) / 5.0 * 9))
                process_email_task.apply_async(
                    args=[_payload_to_dict(payload)],
                    priority=celery_priority,
                )
            return {"fetched": len(emails)}
        except Exception as exc:
            raise self.retry(exc=exc, countdown=300)

    @celery_app.task(name="PDF_summarizer.ingest.worker.process_email_task", bind=True, max_retries=3)
    def process_email_task(self, payload_dict: dict):
        """Process a single email payload dict (serialized for Celery)."""
        try:
            payload = _dict_to_payload(payload_dict)
            return _process_email(payload)
        except Exception as exc:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


# ── Serialization helpers (Celery passes JSON, not Python objects) ─────────────

def _payload_to_dict(p: EmailPayload) -> dict:
    return {
        "message_id": p.message_id,
        "sender": p.sender,
        "sender_domain": p.sender_domain,
        "broker": p.broker,
        "report_date": p.report_date.isoformat() if p.report_date else None,
        "html_body": p.html_body,
        "text_body": p.text_body,
        "pdf_attachments": [
            {"filename": fn, "data_hex": data.hex()} for fn, data in p.pdf_attachments
        ],
    }


def _dict_to_payload(d: dict) -> EmailPayload:
    from ingest.email_fetcher import EmailPayload as EP
    return EP(
        message_id=d["message_id"],
        sender=d["sender"],
        sender_domain=d["sender_domain"],
        broker=d["broker"],
        report_date=datetime.fromisoformat(d["report_date"]) if d.get("report_date") else None,
        html_body=d.get("html_body"),
        text_body=d.get("text_body"),
        pdf_attachments=[
            (a["filename"], bytes.fromhex(a["data_hex"]))
            for a in d.get("pdf_attachments", [])
        ],
    )


# ── Synchronous entry point (no Celery / Redis required) ─────────────────────

def run_ingest_now(max_emails: int = 50) -> List[dict]:
    """
    Run the full ingestion pipeline synchronously (no Celery needed).

    Useful for: manual triggers via the API, testing, first-time setup.
    Emails are processed in priority order: high-tier brokers and recent
    upgrades/downgrades first, then lower-priority maintenance notes.
    """
    emails = _fetch_new_emails()
    emails.sort(key=_email_priority_score, reverse=True)
    print(f"[run_ingest_now] Processing {len(emails)} new emails (priority-sorted)")
    results = []
    for payload in emails[:max_emails]:
        result = _process_email(payload)
        results.append(result)
        print(f"  [{result['status']}] {payload.broker} — {payload.message_id[:12]}…")
    return results
