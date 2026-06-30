"""
Backfill html_body for pdf_documents rows that were ingested before the
html_body column existed.

For each doc missing html_body we hold its email_message_id (SHA-256 of the
raw RFC-822 bytes).  We iterate over Gmail messages, compute the same hash,
and when we get a hit we update the row in-place.

Usage:
    cd emailai/PDF_summarizer
    python scripts/backfill_html_body.py
"""

import base64
import hashlib
import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent.parent   # PDF_summarizer/
sys.path.insert(0, str(_here))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import psycopg
import email as _email

from ingest.email_fetcher import _extract_parts
from ingest.gmail_fetcher import _build_service

# ── config ────────────────────────────────────────────────────────────────────
DB_URL = os.environ["PDF_SUMMARIZER_DB_URL"].replace(
    "postgresql+psycopg://", "postgresql://"
).replace("postgresql+psycopg2://", "postgresql://")

# Search back 12 months to cover all historical docs
MONTHS_BACK = 12


def _gmail_query(months: int) -> str:
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=months * 30)
    return f"after:{cutoff.strftime('%Y/%m/%d')}"


def main():
    conn = psycopg.connect(DB_URL)

    # 1. Collect hashes of docs that need backfill
    rows = conn.execute(
        "SELECT id, email_message_id FROM pdf_documents WHERE html_body IS NULL AND email_message_id IS NOT NULL"
    ).fetchall()

    if not rows:
        print("Nothing to backfill — all docs already have html_body.")
        return

    need = {msg_id: doc_id for doc_id, msg_id in rows}
    print(f"Docs needing html_body backfill: {len(need)}")

    # 2. Walk Gmail messages and match by SHA-256 hash
    service = _build_service()
    query = _gmail_query(MONTHS_BACK)
    print(f"Searching Gmail: {query!r}")

    page_token = None
    fetched = 0
    updated = 0
    remaining = set(need.keys())

    while remaining:
        resp = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=500,
            pageToken=page_token,
        ).execute()

        stubs = resp.get("messages", [])
        if not stubs:
            break

        for stub in stubs:
            if not remaining:
                break
            gmail_id = stub["id"]
            try:
                raw_resp = service.users().messages().get(
                    userId="me", id=gmail_id, format="raw"
                ).execute()
                raw_bytes = base64.urlsafe_b64decode(raw_resp.get("raw", "") + "==")
            except Exception as e:
                print(f"  [warn] Failed to fetch {gmail_id}: {e}")
                continue

            fetched += 1
            msg_hash = hashlib.sha256(raw_bytes).hexdigest()

            if msg_hash not in remaining:
                continue

            # Match found — extract html_body
            doc_id = need[msg_hash]
            msg = _email.message_from_bytes(raw_bytes)
            html_body, _text, _pdfs = _extract_parts(msg)

            if html_body:
                conn.execute(
                    "UPDATE pdf_documents SET html_body = %s WHERE id = %s",
                    (html_body, doc_id),
                )
                conn.commit()
                updated += 1
                remaining.discard(msg_hash)
                print(f"  [backfill] doc={doc_id} updated ({updated}/{len(need)}, {len(remaining)} remaining)")
            else:
                # Email had no HTML part — mark with empty string so we don't retry
                conn.execute(
                    "UPDATE pdf_documents SET html_body = '' WHERE id = %s",
                    (doc_id,),
                )
                conn.commit()
                remaining.discard(msg_hash)
                print(f"  [backfill] doc={doc_id} — no HTML part in email (marked empty)")

            if fetched % 100 == 0:
                print(f"  [progress] {fetched} messages scanned, {updated} updated, {len(remaining)} still needed")

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    conn.close()
    print(f"\nDone. Scanned {fetched} Gmail messages, updated {updated} docs.")
    if remaining:
        print(f"Could not find source email for {len(remaining)} doc(s) — may be older than {MONTHS_BACK} months.")
        not_found_ids = [need[h] for h in remaining]
        print(f"  Doc IDs not resolved: {sorted(not_found_ids)}")


if __name__ == "__main__":
    main()
