"""
Gmail API fetcher — OAuth2 alternative to IMAP + app password.

Uses the Gmail API (read-only scope) with a one-time browser authorization
flow. After the first run, a refresh token is stored in token.json and
all subsequent runs are fully headless.

Setup (one time):
  1. Google Cloud Console → Enable Gmail API
  2. Create OAuth client ID (Desktop App) → download credentials.json
  3. Set GMAIL_CREDENTIALS_FILE=credentials.json in .env
  4. Run: python -m ingest.gmail_fetcher --setup
     (opens browser once to authorize, saves token.json)

Environment variables:
  GMAIL_CREDENTIALS_FILE  Path to credentials JSON from Cloud Console
  GMAIL_TOKEN_FILE        Path to store the refresh token (default: token.json)
  GMAIL_LABEL             Gmail label/folder to scan (default: INBOX)
  GMAIL_MAX_EMAILS        Max emails to fetch per run (default: 50)
"""

from __future__ import annotations

import base64
import email as _email
import hashlib
import os
from pathlib import Path
from typing import List, Optional

from ingest.email_fetcher import (
    EmailPayload,
    DOMAIN_TO_BROKER,
    _decode_str,
    _sender_domain,
    _resolve_broker,
    _extract_parts,
    _parse_date,
    load_config,
)

_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _credentials_path() -> Optional[Path]:
    raw = os.getenv("GMAIL_CREDENTIALS_FILE", "")
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(__file__).parent.parent.parent / raw  # relative to Rays_Intern/
    return p if p.exists() else None


def _token_path() -> Path:
    raw = os.getenv("GMAIL_TOKEN_FILE", "token.json")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(__file__).parent.parent.parent / raw
    return p


def _get_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_file = _token_path()
    creds = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), _SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json())
        return creds

    return None  # need interactive setup


def setup_oauth(open_browser: bool = True):
    """
    Run the one-time OAuth2 authorization flow.
    Opens a browser window for the user to sign in and grant access.
    Saves the refresh token to token.json for all future headless runs.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds_path = _credentials_path()
    if not creds_path:
        raise FileNotFoundError(
            "GMAIL_CREDENTIALS_FILE not set or file not found. "
            "Download OAuth client JSON from Google Cloud Console and set the path in .env"
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
    if open_browser:
        creds = flow.run_local_server(port=0)
    else:
        # Headless fallback: print URL, accept code via stdin
        creds = flow.run_console()

    token_file = _token_path()
    token_file.write_text(creds.to_json())
    print(f"[gmail] Authorization complete. Token saved to {token_file}")
    return creds


def _build_service():
    from googleapiclient.discovery import build

    creds = _get_credentials()
    if creds is None:
        raise RuntimeError(
            "Gmail not authorized. Run: python -m ingest.gmail_fetcher --setup"
        )
    return build("gmail", "v1", credentials=creds)


def _message_to_payload(
    raw_bytes: bytes,
    broker_domains: List[str],
) -> Optional[EmailPayload]:
    """Parse raw RFC-822 bytes into an EmailPayload, same as the IMAP path."""
    msg = _email.message_from_bytes(raw_bytes)
    from_hdr = _decode_str(msg.get("From", ""))
    domain = _sender_domain(from_hdr)
    if not domain:
        return None

    broker = _resolve_broker(domain, broker_domains)
    if not broker:
        return None

    html_body, text_body, pdfs = _extract_parts(msg)
    msg_hash = hashlib.sha256(raw_bytes).hexdigest()

    return EmailPayload(
        message_id=msg_hash,
        sender=from_hdr,
        sender_domain=domain,
        broker=broker,
        subject=_decode_str(msg.get("Subject", "")),
        report_date=_parse_date(msg),
        html_body=html_body,
        text_body=text_body,
        pdf_attachments=pdfs,
    )


def fetch_broker_emails_gmail(
    existing_ids: set,
    max_emails: int = 50,
    label: str = "INBOX",
) -> List[EmailPayload]:
    """
    Fetch unread broker emails via Gmail API.

    Drop-in replacement for fetch_broker_emails() from email_fetcher.py.
    No IMAP credentials required — uses OAuth2 token from token.json.

    Args:
        existing_ids: Set of SHA-256 hashes already in the database (dedup).
        max_emails:   Maximum number of messages to inspect per run.
        label:        Gmail label to scan (default: INBOX).

    Returns:
        List of EmailPayload objects ready for _process_email().
    """
    service = _build_service()
    cfg = load_config()
    broker_domains = cfg["broker_domains"]

    # Query: unread messages only
    query = "is:unread"
    response = service.users().messages().list(
        userId="me",
        q=query,
        labelIds=[label],
        maxResults=max_emails,
    ).execute()

    messages = response.get("messages", [])
    if not messages:
        return []

    print(f"[gmail] Found {len(messages)} unread message(s) to check")

    results: List[EmailPayload] = []
    for msg_stub in messages:
        msg_id = msg_stub["id"]
        try:
            # Fetch full RFC-822 raw bytes
            raw_resp = service.users().messages().get(
                userId="me",
                id=msg_id,
                format="raw",
            ).execute()
            raw_b64 = raw_resp.get("raw", "")
            raw_bytes = base64.urlsafe_b64decode(raw_b64 + "==")
        except Exception as e:
            print(f"[gmail] Failed to fetch message {msg_id}: {e}")
            continue

        # Quick dedup check before full parsing
        msg_hash = hashlib.sha256(raw_bytes).hexdigest()
        if msg_hash in existing_ids:
            continue

        payload = _message_to_payload(raw_bytes, broker_domains)
        if payload is None:
            continue  # not from a known broker domain

        results.append(payload)
        print(f"[gmail] Queued: {payload.sender[:60]} ({payload.broker})")

    print(f"[gmail] {len(results)} new broker email(s) ready for processing")
    return results


def is_available() -> bool:
    """Return True if Gmail OAuth is configured and authorized."""
    if not _credentials_path():
        return False
    try:
        creds = _get_credentials()
        return creds is not None
    except Exception:
        return False


def fetch_broker_emails_gmail_history(
    existing_ids: set,
    months: int = 3,
    broker_filter: Optional[str] = None,
    max_emails: int = 200,
    label: str = "INBOX",
) -> List[EmailPayload]:
    """
    Fetch ALL broker emails (read + unread) from the past `months` months.

    Unlike fetch_broker_emails_gmail() which only looks at unread messages,
    this searches across all mail so historical reports can be back-filled.

    Args:
        existing_ids:  SHA-256 hashes already in the database (dedup).
        months:        How far back to search (default 3 months).
        broker_filter: Optional broker name filter, e.g. "Morgan Stanley".
                       When set, only emails from that broker's domains are returned.
        max_emails:    Maximum results to inspect.
        label:         Gmail label to scan.
    """
    from datetime import date, timedelta

    service = _build_service()
    cfg = load_config()
    broker_domains = cfg["broker_domains"]

    cutoff = date.today() - timedelta(days=months * 30)
    # Gmail date format: YYYY/MM/DD
    after_str = cutoff.strftime("%Y/%m/%d")
    query = f"after:{after_str}"

    # Narrow to broker domains if a filter is specified
    if broker_filter:
        # Build "from:(domain1 OR domain2 ...)" clause
        matching_domains = [
            d for d, b in DOMAIN_TO_BROKER.items()
            if broker_filter.lower() in b.lower()
        ]
        if matching_domains:
            from_clause = " OR ".join(f"from:{d}" for d in matching_domains)
            query = f"({from_clause}) after:{after_str}"

    response = service.users().messages().list(
        userId="me",
        q=query,
        labelIds=[label],
        maxResults=max_emails,
    ).execute()

    messages = response.get("messages", [])
    if not messages:
        print(f"[gmail_history] No messages found matching query: {query!r}")
        return []

    print(f"[gmail_history] Found {len(messages)} message(s) in past {months}mo")

    results: List[EmailPayload] = []
    for msg_stub in messages:
        msg_id = msg_stub["id"]
        try:
            raw_resp = service.users().messages().get(
                userId="me", id=msg_id, format="raw"
            ).execute()
            raw_bytes = base64.urlsafe_b64decode(raw_resp.get("raw", "") + "==")
        except Exception as e:
            print(f"[gmail_history] Failed to fetch {msg_id}: {e}")
            continue

        msg_hash = hashlib.sha256(raw_bytes).hexdigest()
        if msg_hash in existing_ids:
            continue

        payload = _message_to_payload(raw_bytes, broker_domains)
        if payload is None:
            continue

        if broker_filter and (payload.broker or "").lower() != broker_filter.lower():
            continue

        results.append(payload)
        print(f"[gmail_history] Queued: {payload.sender[:60]} ({payload.broker})")

    print(f"[gmail_history] {len(results)} new broker email(s) ready for processing")
    return results


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup_oauth()
    elif "--check" in sys.argv:
        print("Gmail OAuth available:", is_available())
    else:
        print("Usage: python -m ingest.gmail_fetcher --setup | --check")
