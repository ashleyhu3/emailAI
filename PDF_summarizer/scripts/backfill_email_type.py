"""
Backfill email_type and report_type for existing docs using their dense_summary.

Batches 20 docs per LLM call to keep cost low (~14 API calls for 268 docs).
Uses gemini-3.1-pro-preview for accuracy.

Usage:
    cd emailai/PDF_summarizer
    python scripts/backfill_email_type.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import psycopg
from google import genai
from google.genai import types

DB_URL = os.environ["PDF_SUMMARIZER_DB_URL"].replace(
    "postgresql+psycopg://", "postgresql://"
).replace("postgresql+psycopg2://", "postgresql://")

_MODEL = "models/gemini-3.5-flash"
BATCH_SIZE = 20

_SYSTEM = """\
You are a financial research classifier. For each broker email entry provided, output a
JSON array with one object per entry in the same order.

Each object must have exactly these fields:
  "id": integer (copy from input)
  "email_type": "sales" or "analyst"
    sales = digest/round-up covering multiple companies or topics in one email
            (morning notes, weekly wrap, top-ideas lists, sector digests, marketing calendars)
    analyst = focused note on ONE company, ticker, or a single narrow theme
  "report_subtype": one of the strings below, or null
    Only set for analyst emails:
      "formal_report"     - full initiation or comprehensive coverage note
      "model_update"      - primarily an EPS/TP revision with brief commentary
      "earnings"          - any earnings-related note (pre-results, post-results, or both)
      "brief"             - short color, quick take, or commentary (<1 page equivalent)
    Always null for sales emails.

Return ONLY the JSON array, no prose."""


def _classify_batch(client, docs: list[dict]) -> list[dict]:
    """Send one batch to the LLM, return list of {id, email_type, report_subtype}."""
    entries = "\n\n".join(
        f"ID {d['id']} | broker={d['broker']} | subject={d['subject']}\n{d['summary']}"
        for d in docs
    )
    prompt = f"Classify each entry:\n\n{entries}"

    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (response.text or "").strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def main():
    conn = psycopg.connect(DB_URL)

    rows = conn.execute("""
        SELECT id, broker, filename, dense_summary
        FROM pdf_documents
        WHERE email_type IS NULL AND dense_summary IS NOT NULL
        ORDER BY id
    """).fetchall()

    if not rows:
        print("Nothing to backfill.")
        return

    print(f"Classifying {len(rows)} docs in batches of {BATCH_SIZE}...")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    total_updated = 0
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches, 1):
        docs = [
            {
                "id": r[0],
                "broker": r[1] or "unknown",
                "subject": (r[2] or "").replace("email_", "").replace(".eml", ""),
                "summary": (r[3] or "")[:400],
            }
            for r in batch
        ]

        try:
            results = _classify_batch(client, docs)
        except Exception as e:
            print(f"  [batch {batch_num}] LLM error: {e} — retrying once...")
            time.sleep(3)
            try:
                results = _classify_batch(client, docs)
            except Exception as e2:
                print(f"  [batch {batch_num}] Failed again: {e2} — skipping batch")
                continue

        for item in results:
            doc_id = item.get("id")
            email_type = item.get("email_type")
            report_subtype = item.get("report_subtype")

            if email_type not in ("sales", "analyst"):
                print(f"  [warn] doc={doc_id} unexpected email_type={email_type!r}, defaulting to analyst")
                email_type = "analyst"

            valid_subtypes = {"formal_report", "model_update", "earnings", "brief", None}
            if report_subtype not in valid_subtypes:
                report_subtype = None

            conn.execute(
                "UPDATE pdf_documents SET email_type = %s, report_type = %s WHERE id = %s",
                (email_type, report_subtype, doc_id),
            )
            total_updated += 1

        conn.commit()
        print(f"  [batch {batch_num}/{len(batches)}] {len(results)} docs classified, {total_updated} total so far")
        if batch_num < len(batches):
            time.sleep(1)  # gentle rate limiting

    conn.close()

    # Print distribution
    conn2 = psycopg.connect(DB_URL)
    dist = conn2.execute("""
        SELECT email_type, report_type, COUNT(*)
        FROM pdf_documents
        GROUP BY email_type, report_type
        ORDER BY email_type, report_type
    """).fetchall()
    conn2.close()

    print(f"\nDone. {total_updated} docs updated. Distribution:")
    for email_type, report_type, count in dist:
        print(f"  {email_type or 'NULL'} / {report_type or 'NULL'}: {count}")


if __name__ == "__main__":
    main()
