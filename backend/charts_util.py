"""Chart generation utilities for broker research risk-reward charts."""
from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

# Ensure PDF_summarizer is importable
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_pdf_summarizer_dir = os.path.join(os.path.dirname(_backend_dir), "PDF_summarizer")
if _pdf_summarizer_dir not in sys.path:
    sys.path.insert(0, _pdf_summarizer_dir)


# ── Ticker resolution ─────────────────────────────────────────────────────────

_ticker_cache: Dict[str, Optional[str]] = {}


def _extract_ticker_from_dense_summary(dense_summary: str) -> Optional[str]:
    """Pull bare ticker symbol from dense_summary if present as (TICKER)."""
    # Dense summaries often contain things like "Qualcomm Inc. (IDEA)" where IDEA is
    # the report type, not the ticker.  Skip 3-4 char ALL-CAPS words that are report
    # type abbreviations (IDEA, OW, EW, UW) when they follow a company name paren.
    _REPORT_TYPE_ABBREVS = {"IDEA", "OW", "EW", "UW", "UOW", "NA"}
    for m in re.finditer(r"\(([A-Z]{1,5})\)", dense_summary):
        candidate = m.group(1)
        if candidate not in _REPORT_TYPE_ABBREVS:
            return candidate
    return None


def _extract_company_name(dense_summary: str) -> Optional[str]:
    """Extract the primary company name from a dense summary."""
    if not dense_summary:
        return None
    m = re.search(
        r"Morgan Stanley (?:upgraded|downgraded|initiated|reiterated|rates|maintained|"
        r"cuts|raises|reaffirms|starts|resumes)\s+([A-Za-z][A-Za-z0-9 ,\.\-&']+?)"
        r"(?:\s*\([A-Z]+\)|\s+to\s|\s+at\s|\s+from\s|\s*,)",
        dense_summary,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(",")
    # Fallback: first capitalized multi-word segment
    m2 = re.search(r"([A-Z][a-z]+ (?:[A-Z][a-z]+ ?){1,3})", dense_summary)
    return m2.group(1).strip() if m2 else None


def resolve_ticker(company_name: str, existing_tickers: Optional[list] = None) -> Optional[str]:
    """Return a primary ticker for a company, using cache + yfinance Search."""
    key = (company_name or "").lower().strip()
    if key in _ticker_cache:
        return _ticker_cache[key]

    # Prefer tickers extracted during ingestion
    if existing_tickers:
        valid = [t for t in existing_tickers if isinstance(t, str) and 1 < len(t) <= 5]
        if valid:
            _ticker_cache[key] = valid[0]
            return valid[0]

    # yfinance search
    try:
        import yfinance as yf
        results = yf.Search(company_name, max_results=5).quotes
        for r in results:
            sym = r.get("symbol", "")
            exchange = r.get("exchange", "")
            # Prefer US-listed plain tickers (no dots for foreign exchanges)
            if sym and "." not in sym and exchange in ("NMS", "NYQ", "NGM", "PCX", ""):
                _ticker_cache[key] = sym
                return sym
        # Fallback: take first result
        if results:
            sym = results[0].get("symbol")
            if sym:
                _ticker_cache[key] = sym
                return sym
    except Exception:
        pass

    _ticker_cache[key] = None
    return None


# ── DB query ──────────────────────────────────────────────────────────────────

def _query_ms_docs(days: int) -> List[dict]:
    """Return MS research docs from the past N days from the DB."""
    import psycopg
    db_url = os.environ.get("PDF_SUMMARIZER_DB_URL", "")
    conn_str = db_url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    cutoff = date.today() - timedelta(days=days)
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, broker_action, rating, target_price,
                   written_date, tickers, dense_summary, html_body IS NOT NULL AS has_html
            FROM pdf_documents
            WHERE broker ILIKE %s
              AND written_date >= %s
            ORDER BY written_date DESC
            """,
            ("%morgan stanley%", cutoff),
        ).fetchall()
        cols = ["id", "filename", "broker_action", "rating", "target_price",
                "written_date", "tickers", "dense_summary", "has_html"]
        return [dict(zip(cols, r)) for r in rows]


# ── Chart generation ──────────────────────────────────────────────────────────

_ACTION_COLOR = {
    "u":  "#22c55e",   # green  — upgrade
    "d":  "#ef4444",   # red    — downgrade
    "id": "#3b82f6",   # blue   — initiation
    "m":  "#94a3b8",   # gray   — maintain
}
_ACTION_LABEL = {"u": "U", "d": "D", "id": "ID", "m": "M"}


def generate_ms_research_chart(days: int = 90) -> Tuple[str, int]:
    """
    Build an interactive Plotly risk-reward chart for all Morgan Stanley research
    in the past `days` days.  Returns (html_string, company_count).
    Price and volume are on independent axes: volume on the left, price on the right.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf
    from datetime import datetime, timezone

    docs = _query_ms_docs(days)
    if not docs:
        return "<p style='font-family:sans-serif;color:#6b7280'>No Morgan Stanley research found for this period.</p>", 0

    # Group by company
    companies: Dict[str, List[dict]] = {}
    ticker_for: Dict[str, str] = {}

    for doc in docs:
        company = _extract_company_name(doc["dense_summary"] or "")
        if not company:
            continue
        ticker = resolve_ticker(company, doc.get("tickers") or [])
        if not ticker:
            continue
        if company not in companies:
            companies[company] = []
            ticker_for[company] = ticker
        companies[company].append(doc)

    if not companies:
        return "<p style='font-family:sans-serif;color:#6b7280'>Could not resolve tickers for any Morgan Stanley research in this period.</p>", 0

    n = len(companies)

    # Each subplot row gets a secondary y-axis: left=volume, right=price
    fig = make_subplots(
        rows=n, cols=1,
        specs=[[{"secondary_y": True}]] * n,
        subplot_titles=[f"{c} ({ticker_for[c]})" for c in companies],
        vertical_spacing=max(0.04, 0.12 / n),
        row_heights=[280] * n,
    )

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    for row_idx, (company, reports) in enumerate(companies.items(), start=1):
        ticker = ticker_for[company]
        try:
            hist = yf.Ticker(ticker).history(start=start_dt, end=end_dt)
        except Exception:
            hist = pd.DataFrame()

        if hist.empty:
            continue

        vol_max = float(hist["Volume"].max()) or 1.0
        p_min = float(hist["Close"].min())
        p_max = float(hist["Close"].max())
        p_margin = max((p_max - p_min) * 0.15, p_max * 0.05)

        # ── Volume bars (left / primary axis) ────────────────────────────────
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                marker_color="rgba(148,163,184,0.25)",
                name="Volume",
                showlegend=False,
                hovertemplate="%{x|%b %d}<br>Vol %{y:,.0f}<extra></extra>",
            ),
            row=row_idx, col=1,
            secondary_y=False,
        )
        # Scale volume axis so bars occupy only the bottom ~30% of the panel
        fig.update_yaxes(
            range=[0, vol_max * 3.5],
            tickformat=".2s",
            secondary_y=False,
            row=row_idx, col=1,
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#94a3b8", size=9),
            title_text="VOL",
            title_font=dict(color="#94a3b8", size=9),
            title_standoff=4,
        )

        # ── Price line (right / secondary axis) ───────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                line=dict(color="#2563eb", width=1.5),
                name=ticker,
                showlegend=False,
                hovertemplate="%{x|%b %d}<br>%{y:.2f}<extra></extra>",
            ),
            row=row_idx, col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            range=[max(0, p_min - p_margin), p_max + p_margin * 2],
            secondary_y=True,
            row=row_idx, col=1,
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            tickfont=dict(size=9),
        )

        # ── Research event markers (on price axis) ────────────────────────────
        for doc in reports:
            event_date = doc["written_date"]
            if not event_date:
                continue
            action = (doc["broker_action"] or "m").lower()
            color = _ACTION_COLOR.get(action, "#94a3b8")
            label = _ACTION_LABEL.get(action, "?")

            ts = pd.Timestamp(event_date)
            if hist.index.tz is not None:
                ts = ts.tz_localize(hist.index.tz)
            idx = int(hist.index.searchsorted(ts))
            idx = min(idx, len(hist) - 1)
            price_at_event = float(hist["Close"].iloc[idx])

            # First sentence of dense_summary for the tooltip
            summary = doc["dense_summary"] or ""
            first_sentence = re.split(r"(?<=[.!?])\s+", summary.strip())[0] if summary else ""
            if len(first_sentence) > 120:
                first_sentence = first_sentence[:117] + "…"

            tp_line = f"PT: ${doc['target_price']:.0f}<br>" if doc["target_price"] else ""
            open_hint = "🔗 Click to open report" if doc.get("has_html") else "📄 Click to view report"
            hover = (
                f"<b>{label} — {doc['rating'] or 'N/A'}</b><br>"
                f"{str(event_date)}<br>"
                f"{tp_line}"
                f"{first_sentence}<br>"
                f"<i style='color:#93c5fd'>{open_hint}</i>"
            )
            fig.add_trace(
                go.Scatter(
                    x=[event_date],
                    y=[price_at_event * 1.02],
                    mode="markers+text",
                    marker=dict(size=26, color=color, symbol="square",
                                line=dict(width=1.5, color="white")),
                    text=[label],
                    textfont=dict(color="white", size=9, family="Arial Black"),
                    textposition="middle center",
                    showlegend=False,
                    customdata=[[doc["id"], int(bool(doc.get("has_html")))]],
                    hovertemplate=hover + "<extra></extra>",
                ),
                row=row_idx, col=1,
                secondary_y=True,
            )

            if doc["target_price"]:
                fig.add_trace(
                    go.Scatter(
                        x=[event_date],
                        y=[doc["target_price"]],
                        mode="markers",
                        marker=dict(size=10, color="#1e3a5f", symbol="diamond",
                                    line=dict(width=1, color="white")),
                        showlegend=False,
                        customdata=[[doc["id"], int(bool(doc.get("has_html")))]],
                        hovertemplate=f"Price Target: ${doc['target_price']:.0f} — <i>click to open report</i><extra></extra>",
                    ),
                    row=row_idx, col=1,
                    secondary_y=True,
                )

    total_height = max(420, n * 320)
    fig.update_layout(
        height=total_height,
        template="plotly_white",
        margin=dict(l=55, r=70, t=50, b=30),
        font=dict(family="Inter, system-ui, sans-serif", size=11),
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)

    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                              config={"displayModeBar": False})

    if days <= 7:
        time_label = "past week"
    elif days <= 14:
        time_label = "past 2 weeks"
    elif days % 30 == 0:
        nm = days // 30
        time_label = f"past {nm} month{'s' if nm != 1 else ''}"
    else:
        time_label = f"past {days} days"

    return _wrap_chart(chart_html, time_label, n), n


def _wrap_chart(chart_html: str, time_label: str, n_companies: int) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Inter, system-ui, sans-serif; background: #fff; }}
  .chart-area {{ padding: 16px 8px; }}
  .header {{ font-size: 11px; font-weight: 600; letter-spacing: .08em;
             text-transform: uppercase; color: #374151; margin-bottom: 12px; }}
  .legend {{ display: flex; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #6b7280; }}
  .legend-dot {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="chart-area">
  <div class="header">Morgan Stanley Research · {time_label.title()} · {n_companies} compan{"ies" if n_companies!=1 else "y"}</div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Initiation (ID)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>Upgrade (U)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Downgrade (D)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>Maintain (M)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#1e3a5f;clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%)"></div>Price Target</div>
  </div>
  {chart_html}
</div>
<script>
(function() {{
  // Wait for Plotly to be ready, then wire up click-to-open-report
  function attachClickHandler() {{
    var divs = document.querySelectorAll('.plotly-graph-div');
    if (!divs.length) {{ setTimeout(attachClickHandler, 200); return; }}
    divs.forEach(function(div) {{
      div.on('plotly_click', function(data) {{
        if (!data || !data.points || !data.points.length) return;
        var pt = data.points[0];
        var cd = pt.customdata;
        if (!cd) return;
        var docId = cd[0];
        if (!docId) return;
        // Try HTML endpoint first; fall back to content viewer
        var htmlUrl = '/api/documents/' + docId + '/html';
        var contentUrl = '/api/documents/' + docId + '/content';
        fetch(htmlUrl, {{ method: 'HEAD' }})
          .then(function(r) {{
            window.open(r.ok ? htmlUrl : contentUrl, '_blank', 'noopener');
          }})
          .catch(function() {{
            window.open(contentUrl, '_blank', 'noopener');
          }});
      }});
    }});
  }}
  attachClickHandler();
}})();
</script>
</body>
</html>"""
