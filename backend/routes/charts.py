"""Chart routes — broker research risk-reward charts."""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/ms-reports", response_class=HTMLResponse)
def ms_research_chart(months: int = Query(default=3, ge=1, le=24)):
    """
    Generate an interactive risk-reward chart for all Morgan Stanley research
    in the past `months` months.  Returns a self-contained HTML page.
    """
    from charts_util import generate_ms_research_chart
    html, _ = generate_ms_research_chart(months=months)
    return HTMLResponse(content=html)
