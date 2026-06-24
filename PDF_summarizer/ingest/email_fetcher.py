"""
Phase 1: Inbound Gatekeeper — fetch unread broker emails via IMAP.

Connects to the configured mailbox, filters by trusted broker domains,
computes a SHA-256 deduplication hash over the full email payload, and
returns only emails whose hash is not already present in the database.
"""

import email
import hashlib
import imaplib
import os
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email.header import decode_header as _decode_header
from email.message import Message
from typing import List, Optional, Tuple


@dataclass
class EmailPayload:
    """One incoming broker email, pre-routed and deduplicated."""

    message_id: str           # SHA-256 of (body + all attachment bytes)
    sender: str               # Full From address
    sender_domain: str        # e.g. "morganstanley.com"
    broker: str               # Normalized institution derived from domain
    report_date: Optional[datetime]
    html_body: Optional[str]
    text_body: Optional[str]
    subject: str = ""         # Email subject line (used for triage)
    pdf_attachments: List[Tuple[str, bytes]] = field(default_factory=list)  # (filename, bytes)


# Domain → normalized broker name. Extend as needed.
DOMAIN_TO_BROKER: dict = {
    # US / Global bulge-bracket
    "gs.com": "Goldman Sachs",
    "goldmansachs.com": "Goldman Sachs",
    "morganstanley.com": "Morgan Stanley",
    "jpmorgan.com": "J.P. Morgan",
    "jpmchase.com": "J.P. Morgan",
    "ubs.com": "UBS",
    "barclays.com": "Barclays",
    "db.com": "Deutsche Bank",
    "deutschebank.com": "Deutsche Bank",
    "bofa.com": "Bank of America",
    "bofasecurities.com": "Bank of America",
    "ml.com": "Bank of America",
    "citi.com": "Citi",
    "credit-suisse.com": "Credit Suisse",
    "jefferies.com": "Jefferies",
    "wolferesearch.com": "Wolfe Research",
    "bernstein.com": "Bernstein",
    "btigresearch.com": "BTIG",
    # Asia / Japan brokers
    "daiwacm.com": "Daiwa Capital Markets",
    "daiwa.co.jp": "Daiwa Capital Markets",
    "nomura.com": "Nomura",
    "nomura.co.jp": "Nomura",
    "clsa.com": "CLSA",
    "macquarie.com": "Macquarie",
    "cgscimb.com": "CGS-CIMB",
    "boci.com.hk": "BOCI",
    "ccbis.com": "CCB International",
    "hsbc.com": "HSBC",
    "sc.com": "Standard Chartered",
    # Chinese brokers (A-share / fixed income)
    "tfzq.com": "天风证券",
    "citics.com": "中信证券",
    "gtja.com": "国泰君安",
    "htsec.com": "海通证券",
    "swsresearch.com": "申万宏源",
    "csc.com.cn": "中信建投",
    "gjzq.com.cn": "国金证券",
    "dfzq.com.cn": "东方证券",
    "cmschina.com.cn": "招商证券",
    "cgws.com": "长城证券",
    "essence.com.cn": "安信证券",
}


def _decode_str(value: str) -> str:
    parts = _decode_header(value)
    decoded = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            decoded.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(chunk)
    return "".join(decoded)


def _sender_domain(from_addr: str) -> Optional[str]:
    if "@" not in from_addr:
        return None
    return from_addr.split("@")[-1].strip().strip(">").lower()


def _resolve_broker(domain: str, allowed_domains: List[str]) -> Optional[str]:
    """
    Match a sender domain against the whitelist, handling subdomains.

    'research.morganstanley.com' matches whitelist entry 'morganstanley.com'.
    Returns the canonical broker name if matched, else None.
    """
    # Exact match first
    if domain in DOMAIN_TO_BROKER:
        return DOMAIN_TO_BROKER[domain]

    # Subdomain match: does any whitelisted domain appear as a suffix?
    for allowed in allowed_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return DOMAIN_TO_BROKER.get(allowed, allowed.split(".")[0].title())

    return None


def _message_hash(msg: Message) -> str:
    """SHA-256 over the entire raw email message for deduplication."""
    raw = msg.as_bytes()
    return hashlib.sha256(raw).hexdigest()


def _extract_parts(msg: Message) -> Tuple[Optional[str], Optional[str], List[Tuple[str, bytes]]]:
    """Walk the MIME tree and return (html_body, text_body, pdf_attachments)."""
    html_body: Optional[str] = None
    text_body: Optional[str] = None
    pdfs: List[Tuple[str, bytes]] = []

    for part in msg.walk():
        ct = part.get_content_type()
        cd = part.get("Content-Disposition", "")

        if part.get_filename():
            fname = _decode_str(part.get_filename())
            if fname.lower().endswith(".pdf"):
                payload = part.get_payload(decode=True)
                if payload:
                    pdfs.append((fname, payload))
            continue

        if ct == "text/html" and not html_body:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                html_body = payload.decode(charset, errors="replace")
        elif ct == "text/plain" and not text_body:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                text_body = payload.decode(charset, errors="replace")

    return html_body, text_body, pdfs


def _parse_date(msg: Message) -> Optional[datetime]:
    date_str = msg.get("Date")
    if not date_str:
        return None
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def fetch_broker_emails(
    host: str,
    port: int,
    username: str,
    password: str,
    broker_domains: List[str],
    existing_ids: set,
    inbox: str = "INBOX",
    max_emails: int = 100,
) -> List[EmailPayload]:
    """
    Connect to IMAP, fetch UNSEEN messages from trusted broker domains,
    skip any whose hash is in existing_ids, and return parsed EmailPayload objects.

    Args:
        broker_domains: Whitelist of sender domains (e.g. ["gs.com", "morganstanley.com"]).
        existing_ids: Set of SHA-256 hashes already present in the database.
    """
    ctx = ssl.create_default_context()
    results: List[EmailPayload] = []

    with imaplib.IMAP4_SSL(host, port, ssl_context=ctx) as imap:
        imap.login(username, password)
        imap.select(inbox)

        # Fetch all UNSEEN; we domain-filter in Python to avoid complex IMAP SEARCH syntax
        status, data = imap.search(None, "UNSEEN")
        if status != "OK" or not data[0]:
            return results

        uids = data[0].split()[-max_emails:]  # most-recent cap

        for uid in uids:
            status, raw = imap.fetch(uid, "(RFC822)")
            if status != "OK" or not raw or not raw[0]:
                continue
            raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else raw[0]
            msg = email.message_from_bytes(raw_bytes)

            from_hdr = _decode_str(msg.get("From", ""))
            domain = _sender_domain(from_hdr)
            if not domain:
                continue

            broker = _resolve_broker(domain, broker_domains)
            if not broker:
                continue

            msg_hash = _message_hash(msg)
            if msg_hash in existing_ids:
                continue

            html_body, text_body, pdfs = _extract_parts(msg)

            results.append(EmailPayload(
                message_id=msg_hash,
                sender=from_hdr,
                sender_domain=domain,
                broker=broker,
                subject=_decode_str(msg.get("Subject", "")),
                report_date=_parse_date(msg),
                html_body=html_body,
                text_body=text_body,
                pdf_attachments=pdfs,
            ))

    return results


def load_config() -> dict:
    """Load IMAP + broker domain config from environment."""
    # DOMAIN_TO_BROKER is always the authoritative source. BROKER_DOMAINS in .env
    # can add extra domains (e.g., enterprise subdomains) but never overrides the dict.
    base_domains = set(DOMAIN_TO_BROKER.keys())
    env_extra = os.getenv("BROKER_DOMAINS", "")
    extra = {d.strip() for d in env_extra.split(",") if d.strip()}
    all_domains = sorted(base_domains | extra)
    return {
        "host": os.getenv("IMAP_HOST", "imap.gmail.com"),
        "port": int(os.getenv("IMAP_PORT", "993")),
        "username": os.getenv("IMAP_USER", ""),
        "password": os.getenv("IMAP_PASSWORD", ""),
        "inbox": os.getenv("IMAP_INBOX", "INBOX"),
        "broker_domains": all_domains,
    }


def parse_eml_bytes(
    raw: bytes,
    broker_override: Optional[str] = None,
) -> EmailPayload:
    """
    Parse raw .eml bytes into an EmailPayload, exactly as the IMAP fetcher does.

    Args:
        raw: The full contents of a .eml file.
        broker_override: If provided, skip domain resolution and use this broker name.
                         Useful when testing with internal/forwarded emails whose
                         sender domain isn't in DOMAIN_TO_BROKER.
    """
    msg = email.message_from_bytes(raw)
    from_hdr = _decode_str(msg.get("From", ""))
    domain = _sender_domain(from_hdr) or ""

    cfg = load_config()
    broker = broker_override or _resolve_broker(domain, cfg["broker_domains"]) or domain.split(".")[0].title() or "Unknown"

    html_body, text_body, pdfs = _extract_parts(msg)

    return EmailPayload(
        message_id=_message_hash(msg),
        sender=from_hdr,
        sender_domain=domain,
        broker=broker,
        subject=_decode_str(msg.get("Subject", "")),
        report_date=_parse_date(msg),
        html_body=html_body,
        text_body=text_body,
        pdf_attachments=pdfs,
    )


def test_connection() -> dict:
    """
    Verify IMAP credentials and preview what's in the inbox.

    Returns a status dict with connection health, message counts, and
    a sample of broker-matching senders — useful for the /ingest/test-email endpoint.
    """
    cfg = load_config()
    if not cfg["username"] or not cfg["password"]:
        return {
            "ok": False,
            "error": "IMAP_USER or IMAP_PASSWORD not set in .env",
            "config": {"host": cfg["host"], "user": cfg["username"]},
        }

    ctx = ssl.create_default_context()
    try:
        with imaplib.IMAP4_SSL(cfg["host"], cfg["port"], ssl_context=ctx) as imap:
            imap.login(cfg["username"], cfg["password"])
            imap.select(cfg["inbox"])

            # Count UNSEEN
            _, unseen_data = imap.search(None, "UNSEEN")
            unseen_uids = unseen_data[0].split() if unseen_data[0] else []

            # Sample the 20 most-recent UNSEEN headers only (no body download)
            broker_matches = []
            for uid in unseen_uids[-20:]:
                _, hdr = imap.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
                if not hdr or not hdr[0]:
                    continue
                raw_hdr = hdr[0][1] if isinstance(hdr[0], tuple) else hdr[0]
                msg = email.message_from_bytes(raw_hdr)
                from_hdr = _decode_str(msg.get("From", ""))
                domain = _sender_domain(from_hdr)
                if domain:
                    broker = _resolve_broker(domain, cfg["broker_domains"])
                    if broker:
                        broker_matches.append({
                            "sender": from_hdr,
                            "broker": broker,
                            "subject": _decode_str(msg.get("Subject", "")),
                        })

            return {
                "ok": True,
                "host": cfg["host"],
                "user": cfg["username"],
                "inbox": cfg["inbox"],
                "unseen_total": len(unseen_uids),
                "broker_domains_configured": cfg["broker_domains"],
                "broker_matches_in_unseen": broker_matches,
            }
    except imaplib.IMAP4.error as e:
        return {"ok": False, "error": f"IMAP auth failed: {e}", "config": {"host": cfg["host"], "user": cfg["username"]}}
    except Exception as e:
        return {"ok": False, "error": str(e), "config": {"host": cfg["host"], "user": cfg["username"]}}
