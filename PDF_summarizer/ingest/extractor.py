"""
Deterministic extraction and pre-processing utilities.

These run BEFORE any LLM call to:
  1. Extract fields with high confidence from text patterns alone (zero token cost).
  2. Target Agent-2's input to only high-signal AOIM sections (~60% input token reduction).
  3. Pre-filter Docling figures already captured as text (skip Agent-1 for those).
  4. Triage emails that are not research reports before invoking the heavy pipeline.
  5. Repair obvious field-level formatting errors post-LLM (eliminate many Agent-3 calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set


# ── Date parsing ──────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

_DATE_ISO = re.compile(r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b')
_DATE_WORDY = re.compile(
    r'\b(?:(\d{1,2})\s+)?'
    r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'(?:,?\s+(\d{1,2}))?,?\s+(20\d{2})\b',
    re.I,
)
_DATE_SLASH = re.compile(r'\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b')
# Chinese date format: 2024年6月24日 or 2024年06月24日
_DATE_CJK = re.compile(r'(20\d{2})\s*年\s*(1[0-2]|0?[1-9])\s*月\s*(3[01]|[12]\d|0?[1-9])\s*日')


def _try_parse_date(text: str) -> Optional[date]:
    """Try multiple date formats in text, return first successful parse."""
    m = _DATE_ISO.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _DATE_CJK.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _DATE_WORDY.search(text)
    if m:
        day_pre, month_str, day_post, year = m.group(1), m.group(2), m.group(3), m.group(4)
        day = int(day_pre or day_post or 1)
        month = _MONTH_MAP.get(month_str.lower()[:3])
        if month:
            try:
                return date(int(year), month, day)
            except ValueError:
                pass

    m = _DATE_SLASH.search(text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    return None


# ── Action classification ─────────────────────────────────────────────────────

_ACTION_REGEXES: Dict[str, list] = {
    "u": [
        re.compile(r'\bupgrad', re.I),
        re.compile(r'\braised?\s+(?:rating\s+)?to\b', re.I),
        re.compile(r'\bbuy\s+from\s+(?:neutral|hold)\b', re.I),
        re.compile(r'\blifted?\s+to\b', re.I),
        # Chinese: 上调 (upgrade), 升级评级, 由...上调至
        re.compile(r'上调(?:评级|至)?'),
        re.compile(r'升级(?:评级)?'),
    ],
    "d": [
        re.compile(r'\bdowngrad', re.I),
        re.compile(r'\bcut(?:ting)?\s+(?:rating\s+)?to\b', re.I),
        re.compile(r'\breduced?\s+to\b', re.I),
        re.compile(r'\bsell\s+from\s+(?:neutral|buy)\b', re.I),
        # Chinese: 下调 (downgrade), 降级评级
        re.compile(r'下调(?:评级|至)?'),
        re.compile(r'降级(?:评级)?'),
    ],
    "id": [
        re.compile(r'\binitiati', re.I),
        re.compile(r'\bcommence[sd]?\s+coverage\b', re.I),
        re.compile(r'\bstart(?:ing|ed)?\s+coverage\b', re.I),
        # Chinese: 首次覆盖 (initiate coverage), 开始覆盖
        re.compile(r'首次(?:覆盖|评级)'),
        re.compile(r'开始覆盖'),
        re.compile(r'启动覆盖'),
    ],
    "m": [
        re.compile(r'\bmaintain', re.I),
        re.compile(r'\breiterat', re.I),
        re.compile(r'\breaffirm', re.I),
        re.compile(r'\baffirm', re.I),
        # Chinese: 维持 (maintain), 重申
        re.compile(r'维持(?:评级|买入|增持|中性|减持|卖出)?'),
        re.compile(r'重申(?:评级)?'),
    ],
}


def _detect_action(text: str) -> Optional[str]:
    for action, patterns in _ACTION_REGEXES.items():
        if any(p.search(text) for p in patterns):
            return action
    return None


# ── Rating classification ─────────────────────────────────────────────────────

_RATING_RULES = [
    (re.compile(r'\boverweight\b', re.I), "Overweight"),
    (re.compile(r'\bstrong\s+buy\b', re.I), "Overweight"),
    (re.compile(r'\boutperform\b', re.I), "Overweight"),
    (re.compile(r'\bbuy\b', re.I), "Overweight"),
    (re.compile(r'\bequal.weight\b', re.I), "Equal-weight"),
    (re.compile(r'\bequal\s+weight\b', re.I), "Equal-weight"),
    (re.compile(r'\bmarket.perform\b', re.I), "Equal-weight"),
    (re.compile(r'\bneutral\b', re.I), "Equal-weight"),
    (re.compile(r'\bhold\b', re.I), "Equal-weight"),
    (re.compile(r'\bunderweight\b', re.I), "Underweight"),
    (re.compile(r'\bunderperform\b', re.I), "Underweight"),
    (re.compile(r'\bsell\b', re.I), "Underweight"),
    # Chinese ratings
    # Overweight: 买入 (buy), 增持 (accumulate/overweight), 强烈推荐 (strong buy)
    (re.compile(r'(?:^|[\s,，])买入(?:$|[\s,，\)）])'), "Overweight"),
    (re.compile(r'增持'), "Overweight"),
    (re.compile(r'强烈推荐'), "Overweight"),
    (re.compile(r'推荐'), "Overweight"),
    # Equal-weight: 中性 (neutral), 持有 (hold), 观望
    (re.compile(r'中性'), "Equal-weight"),
    (re.compile(r'持有'), "Equal-weight"),
    (re.compile(r'观望'), "Equal-weight"),
    # Underweight: 减持 (reduce/underweight), 卖出 (sell), 回避 (avoid)
    (re.compile(r'减持'), "Underweight"),
    (re.compile(r'卖出'), "Underweight"),
    (re.compile(r'回避'), "Underweight"),
]


def _detect_rating(text: str) -> Optional[str]:
    for pattern, rating in _RATING_RULES:
        if pattern.search(text):
            return rating
    return None


# ── Target price extraction ───────────────────────────────────────────────────

_TP_PATTERNS = [
    re.compile(r'(?:price\s+target|PT)\s*[:=]?\s*[$£¥]?\s*([\d,]+(?:\.\d+)?)', re.I),
    re.compile(r'target\s+(?:price|of)\s*[$£¥]?\s*([\d,]+(?:\.\d+)?)', re.I),
    re.compile(r'[$£¥]\s*([\d,]+(?:\.\d+)?)\s*(?:price\s+target|PT)', re.I),
    re.compile(r'(?:raises?|lowers?|cuts?|maintains?)\s+(?:PT|target)\s+(?:to\s+)?[$£¥]?\s*([\d,]+(?:\.\d+)?)', re.I),
    # Chinese: 目标价 (target price), 目标价格
    re.compile(r'目标(?:价格?|股价)\s*[：:为]?\s*(?:港元|人民币|元|¥|HK\$|CNY|RMB)?\s*([\d,]+(?:\.\d+)?)'),
    re.compile(r'(?:港元|人民币|元|HK\$|CNY|RMB)\s*([\d,]+(?:\.\d+)?)\s*(?:目标价?|TP)'),
]


def _detect_target_price(text: str) -> Optional[float]:
    for pattern in _TP_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


# ── Ticker / company extraction ───────────────────────────────────────────────

# Exchange-qualified tickers: "AAPL US", "2330 TT", "005930 KS", "TSMC TW"
_TICKER_EXCHANGE_RE = re.compile(
    r'\b([A-Z0-9]{1,6})\s+(?:US|HK|TW|TT|KS|JP|LN|GR|FP|AU|SP|IN|CN|SG)\b'
)

# RIC-style tickers: "2330.TW", "7203.T", "TSMC.TW", "005930.KS"
_TICKER_RIC_RE = re.compile(
    r'\b([A-Z0-9]{1,6})\.(TW|HK|T|KS|L|DE|PA|AX|SI|NS|BO|SS|SZ)\b'
)

# Subject-line company name: "Reliance Industries: Upgrade to Buy" or "TSMC — OW"
# Captures everything before ":", "—", "|" (stripping email prefixes).
_SUBJECT_COMPANY_RE = re.compile(
    r'^(?:(?:fw|fwd|re|re:|nt)\s*:\s*)*([^:|\—\–\|]{4,60})[:|\—\–\|]',
    re.IGNORECASE,
)

# Words that look like tickers but are always English words — don't extract these.
_TICKER_STOPWORDS: frozenset = frozenset({
    "THE", "AND", "FOR", "ARE", "BUY", "SELL", "HOLD", "NOTE", "RATE", "FROM", "INTO",
    "WITH", "OVER", "ALSO", "BEEN", "HAVE", "WILL", "WHAT", "WHEN", "THEY", "BOTH",
    "EACH", "JUST", "SOME", "YOUR", "MANY", "MUCH", "NEED", "TAKE", "MAKE", "GIVE",
    "COME", "MADE", "LIKE", "USED", "DAYS", "WERE", "THAN", "FORM", "SHOW", "FALL",
    "RISE", "FLAT", "WEAK", "SOFT", "AMID", "NEAR", "EARN", "BEAT", "MISS", "HALF",
    "GROW", "ZERO", "BULL", "BEAR", "STAY", "ADDS", "CUTS", "SETS", "SEES", "HITS",
    "TOPS", "PAYS", "SAYS", "GDP", "YOY", "QOQ", "LTM", "NTM", "EPS", "DPS", "NAV",
    "ROE", "ROA", "IRR", "NPV", "DCF", "IPO", "SPO", "M&A", "LBO", "ESG", "ESS",
    "ETF", "AUM", "AUC", "RHS", "LHS", "FY", "YE", "CY", "CAGR", "CAPEX", "OPEX",
    "EBIT", "CEO", "CFO", "COO", "CTO", "MOM", "YOY", "USD", "HKD", "JPY", "CNY",
    "KRW", "TWD", "EUR", "GBP", "AUD", "SGD", "INR", "BRL", "EMEA", "APAC", "LATAM",
})


def _extract_tickers(text: str) -> List[str]:
    """Extract high-confidence ticker symbols from text."""
    found: list = []
    seen: set = set()

    for m in _TICKER_EXCHANGE_RE.finditer(text):
        t = m.group(1).upper()
        if t not in _TICKER_STOPWORDS and t not in seen:
            found.append(f"{t} {m.group(0).split()[-1]}")  # keep exchange suffix
            seen.add(t)

    for m in _TICKER_RIC_RE.finditer(text):
        t = m.group(1).upper()
        if t not in _TICKER_STOPWORDS and t not in seen:
            found.append(m.group(0))
            seen.add(t)

    return found[:8]  # cap at 8 to stay concise


def _extract_company_from_subject(subject: str) -> Optional[str]:
    """Extract company/stock name from structured email subject lines."""
    if not subject:
        return None
    m = _SUBJECT_COMPANY_RE.match(subject.strip())
    if not m:
        return None
    company = m.group(1).strip().strip("[](){}")
    # Reject if it looks like boilerplate (too short, all caps and > 6 chars is likely ticker)
    if len(company) < 3:
        return None
    if company.isupper() and len(company) > 6:
        return None  # looks like an all-caps title, not a company name
    return company


# ── Main deterministic extractor ──────────────────────────────────────────────

@dataclass
class PartialExtraction:
    """Fields extracted deterministically before any LLM call."""
    report_date: Optional[date] = None
    broker_action: Optional[str] = None
    rating: Optional[str] = None
    target_price: Optional[float] = None
    tickers: List[str] = field(default_factory=list)
    company_name: Optional[str] = None
    confident_fields: Set[str] = field(default_factory=set)

    @property
    def overall_confidence(self) -> float:
        return len(self.confident_fields) / 6.0

    def as_hint_block(self) -> str:
        """Format as a hint block to prepend to AOIM text for Agent 2."""
        if not self.confident_fields:
            return ""
        lines = ["=== PRE-EXTRACTED FIELDS (HIGH CONFIDENCE — CONFIRM OR CORRECT) ==="]
        if self.report_date:
            lines.append(f"report_date: {self.report_date.isoformat()}")
        if self.broker_action:
            lines.append(f"broker_action: {self.broker_action}")
        if self.rating:
            lines.append(f"rating: {self.rating}")
        if self.target_price is not None:
            lines.append(f"target_price: {self.target_price}")
        if self.tickers:
            lines.append(f"tickers: {self.tickers}")
        if self.company_name:
            lines.append(f"company_name: {self.company_name}")
        lines.append("=== END PRE-EXTRACTED FIELDS ===")
        return "\n".join(lines)


def extract_fields_deterministically(text: str, subject: str = "") -> PartialExtraction:
    """
    Regex-based extraction pass — zero LLM cost.

    Run on the email text/subject before any PDF slicing or agent call.
    High-confidence fields are passed as hints to Agent 2 to reduce its token workload.

    Args:
        text: Combined email body (text + HTML prefix) to scan.
        subject: Email subject line for company/ticker extraction.
    """
    result = PartialExtraction()

    d = _try_parse_date(text)
    if d:
        result.report_date = d
        result.confident_fields.add("report_date")

    action = _detect_action(text)
    if action:
        result.broker_action = action
        result.confident_fields.add("broker_action")

    rating = _detect_rating(text)
    if rating:
        result.rating = rating
        result.confident_fields.add("rating")

    tp = _detect_target_price(text)
    if tp is not None:
        result.target_price = tp
        result.confident_fields.add("target_price")

    # Ticker extraction (exchange-qualified and RIC formats only — high precision)
    tickers = _extract_tickers(f"{subject} {text[:2000]}")
    if tickers:
        result.tickers = tickers
        result.confident_fields.add("tickers")

    # Company name from structured subject lines ("Reliance Industries: Upgrade to Buy")
    company = _extract_company_from_subject(subject)
    if company:
        result.company_name = company
        result.confident_fields.add("company_name")

    return result


# ── AOIM section targeting ────────────────────────────────────────────────────

_SIGNAL_RE = [
    re.compile(r'(?:rating|overweight|underweight|equal.weight|buy|sell|neutral)', re.I),
    re.compile(r'(?:target|price\s+target|\bpt\b)', re.I),
    re.compile(r'(?:upgrade|downgrade|initiat|maintain|reiterat)', re.I),
    re.compile(r'(?:\beps\b|\bp/e\b|earnings\s+per)', re.I),
    re.compile(r'(?:summary|recommendation|investment\s+(?:view|thesis|case))', re.I),
    # Chinese signal keywords for AOIM section targeting
    re.compile(r'评级|目标价|买入|增持|减持|卖出|中性|持有|上调|下调|维持'),
    re.compile(r'盈利预测|净利润|每股收益|市盈率|EPS|PE'),
    re.compile(r'投资建议|研究结论|核心观点|风险提示'),
]


def extract_relevant_aoim_sections(aoim_text: str, max_chars: int = 3000) -> str:
    """
    Extract high-signal sections from AOIM to reduce Agent-2 input tokens by ~60%.

    Always includes the first 30 lines (title, header, first paragraph).
    Also pulls lines near rating/action/TP signal words with ±2/+5 line context.
    Falls back to uniform truncation for very short or very dense texts.
    """
    if len(aoim_text) <= max_chars:
        return aoim_text

    lines = aoim_text.split("\n")
    header_end = min(30, len(lines))
    included: Dict[int, bool] = {i: True for i in range(header_end)}

    for i, line in enumerate(lines):
        if any(p.search(line) for p in _SIGNAL_RE):
            for j in range(max(0, i - 2), min(len(lines), i + 6)):
                included[j] = True

    selected = [lines[i] for i in sorted(included)]
    result = "\n".join(selected)

    if len(result) > max_chars:
        result = result[:max_chars] + "\n…[truncated for token efficiency]"

    # If the targeted extraction is suspiciously short, fall back to uniform truncation
    if len(result) < 500:
        return aoim_text[:max_chars] + "\n…[truncated]"

    return result


# ── Agent-1 figure pre-filter ─────────────────────────────────────────────────

_TABLE_PIPE_RE = re.compile(r'\|.+\|', re.M)
_NUMERIC_RE = re.compile(r'\d+\.?\d*')


def figure_needs_vision(caption: str, surrounding_text: str) -> bool:
    """
    Return True if this figure requires Agent-1 (vision model call).

    Return False when the figure's data is already captured as structured text
    in the surrounding AOIM (Docling reconstructed it as a Markdown table),
    making Agent-1 redundant for this figure.

    Saves ~50-80% of Agent-1 vision calls on reports with text-based tables.
    """
    # Docling already reconstructed a pipe-delimited table
    if _TABLE_PIPE_RE.search(surrounding_text):
        return False

    # High ratio of numbers in surrounding text → already extracted
    words = surrounding_text.split()
    if len(words) > 15:
        numeric_count = len(_NUMERIC_RE.findall(surrounding_text))
        if numeric_count / len(words) > 0.35:
            return False

    return True


# ── Post-processing rule engine ───────────────────────────────────────────────

_RATING_SYNONYMS = {
    "buy": "Overweight", "strong buy": "Overweight", "outperform": "Overweight",
    "market outperform": "Overweight", "op": "Overweight",
    "hold": "Equal-weight", "neutral": "Equal-weight", "market perform": "Equal-weight",
    "in-line": "Equal-weight", "peer perform": "Equal-weight",
    "sell": "Underweight", "underperform": "Underweight", "reduce": "Underweight",
}

_ACTION_SYNONYMS = {
    "upgrade": "u", "upgraded": "u", "up": "u",
    "downgrade": "d", "downgraded": "d", "down": "d",
    "initiation": "id", "initiate": "id", "initiation of coverage": "id", "ioc": "id",
    "maintenance": "m", "maintain": "m", "reiterate": "m", "reaffirm": "m",
    "neutral": "m", "hold": "m", "unchanged": "m",
}


def repair_metadata_fields(data: dict) -> dict:
    """
    Apply cheap regex / lookup fixes to LLM output BEFORE considering it failed.

    Handles:
    - Date strings in non-ISO format → YYYY-MM-DD
    - Target price as string "$150" or "150 USD" → float
    - Rating synonyms → canonical Overweight / Equal-weight / Underweight
    - broker_action synonyms → canonical u / d / id / m

    Returns a copy with fixes applied. Raises nothing — caller validates afterward.
    """
    data = dict(data)

    # Fix report_date
    if isinstance(data.get("report_date"), str):
        parsed = _try_parse_date(data["report_date"])
        if parsed:
            data["report_date"] = parsed.isoformat()

    # Fix target_price
    if isinstance(data.get("target_price"), str):
        m = re.search(r"[\d,]+(?:\.\d+)?", data["target_price"])
        if m:
            try:
                data["target_price"] = float(m.group().replace(",", ""))
            except ValueError:
                data["target_price"] = None

    # Fix rating
    raw_rating = (data.get("rating") or "").strip().lower()
    if raw_rating and raw_rating not in {"overweight", "equal-weight", "underweight"}:
        data["rating"] = _RATING_SYNONYMS.get(raw_rating, data.get("rating"))

    # Fix broker_action
    raw_action = (data.get("broker_action") or "").strip().lower()
    if raw_action and raw_action not in {"u", "d", "id", "m"}:
        data["broker_action"] = _ACTION_SYNONYMS.get(raw_action, data.get("broker_action"))

    # Ensure dense_summary is a non-empty string
    if not data.get("dense_summary"):
        data["dense_summary"] = f"Broker report from {data.get('broker', 'unknown broker')}."

    return data


# ── Report relevance triage ───────────────────────────────────────────────────

_REPORT_RE = re.compile(
    r'(?:'
    # English keywords
    r'\b(?:research|analyst|rating|target\s+price|initiat|upgrade|downgrade|'
    r'recommend|report|note|coverage|outlook|forecast|preview|earnings\s+(?:call|report)|'
    r'overweight|underweight|neutral|buy|sell|hold|investment\s+(?:view|thesis))\b'
    # Chinese keywords: 研究报告, 分析师, 评级, 目标价, 买入, 增持, 减持, 卖出, 中性, 持有,
    #   首次覆盖, 上调, 下调, 维持, 盈利预测, 投资建议
    r'|研究(?:报告|员)|分析师|评级|目标价|买入|增持|减持|卖出|中性持有'
    r'|首次覆盖|上调评级|下调评级|维持评级|盈利预测|投资建议|研报'
    r')',
    re.I,
)

_NON_REPORT_RE = re.compile(
    r'\b(?:confirm(?:ation)?|receipt|invoice|statement|unsubscribe|'
    r'calendar|reminder|invitation|scheduling|account|password|login|'
    r'morning\s+(?:brief|wrap|digest)|close\s+of\s+(?:play|business)|'
    r'trade\s+confirm|settlement|execution\s+report)\b',
    re.I,
)

# Hard block: applies even to emails from known broker domains.
_HARD_NON_REPORT_RE = re.compile(
    r'\b(?:trade\s+confirm(?:ation)?|settlement\s+notice|execution\s+report|'
    r'account\s+statement|password\s+reset|login\s+alert|'
    r'invoice\s+(?:attached|enclosed)|order\s+(?:receipt|confirmation)|'
    r'sales\s+and\s+trading\s+(?:note|department)|'   # S&T notes are explicitly not research
    r'not\s+a\s+product\s+of\s+(?:the\s+)?jefferies\s+research)\b',
    re.I,
)

# Pure event/invitation signals — apply to known brokers when NO research signal is present.
# These are high-precision patterns that never appear in actual research notes.
_INVITATION_RE = re.compile(
    r'\b(?:'
    r'you(?:\'re|\s+are)\s+invited|'
    r'join\s+us\s+(?:for|at|in)\b|'
    r'register\s+(?:now|today|here|to\s+attend|for\s+(?:the|our|free))|'
    r'rsvp\b|'
    r'save\s+the\s+date\b|'
    r'(?:book|secure|reserve)\s+(?:your\s+)?(?:place|spot|seat)\b|'
    r'webinar\s+invitation|'
    r'conference\s+invitation|'
    r'event\s+invitation|'
    r'invitation\s+to\s+(?:join|attend|register)|'
    r'you\s+have\s+been\s+invited|'
    r'attend\s+(?:our|the|this)\s+(?:webinar|event|conference|seminar|forum|roundtable)|'
    r'lunch(?:eon)?\s+invitation|'
    r'dinner\s+invitation|'
    r'roadshow\s+invitation|'
    r'please\s+(?:join|register|confirm\s+your\s+attendance)|'
    r'limited\s+(?:seats?|spaces?|spots?)\s+available|'
    r'click\s+(?:here\s+)?to\s+register|'
    # Corporate access invitation patterns (Daiwa, MS Japan, etc.)
    r'we\s+(?:have\s+)?invited\s+.{0,80}\s+to\s+(?:join|present|speak)|'
    r'kindly\s+let\s+(?:me|us)\s+know\s+if\s+you\s+(?:are\s+)?interested|'
    r'please\s+(?:add\s+(?:it|this)\s+(?:in|to)\s+(?:your\s+)?calendar|register\s+in\s+advance)|'
    r'virtual\s+group\s+meeting|'
    r'top\s+management\s+(?:group\s+)?meeting|'
    r'corporate\s+access\s+(?:event|meeting|call)|'
    r'ms\.?\s*japan\s+corporate\s+access'
    r')\b',
    re.I,
)

# Invitation signals strong enough to block even when _REPORT_RE fires in the body.
# Applied to SUBJECT ONLY — broker email footers always contain "Analyst"/"RESEARCH",
# so the normal invitation check (which requires absence of _REPORT_RE) never fires.
_HARD_INVITATION_SUBJECT_RE = re.compile(
    r'(?:'
    r'\bsave\s+the\s+date\b|'
    r'\bJEF\s+Access[:\s]|'                       # Jefferies event series
    r'Jefferies\s+Hosts?\s*[|:]|'                 # Jefferies NDR/conference host
    r'\bfireside\s+chat\b|'                       # any broker fireside chat
    r'\bbull[/\\]bear\s+(?:debate|panel)\b|'      # debate webinars
    r'\bnon[-\s]deal\s+roadshow\b|'               # NDR invitations
    r'\bdaily\s+summ(?:ary|aries)\b|'             # corporate access digest
    r'\bglobal\s+access\s+daily\b|'               # e.g. "Asia Pacific Global Access Daily"
    r'\bweekly\s+(?:download|update|digest)\b|'  # webinar series
    r"What[’']s\s+Next\?|"                    # webinar series title (straight or curly apostrophe)
    r'\bwebinar\b|'                               # any webinar mention in subject
    r'\bpre[-\s]deal\s+investor\s+education\b|'  # IPO roadshow PDIE
    r'\bmight\s+this\s+be\s+of\s+interest\b'     # Jefferies NDR/IPO solicitation phrase
    r')',
    re.I,
)

# Compliance / regulatory disclosure PDF patterns — applied to PDF *content*, not email body.
_COMPLIANCE_PDF_RE = re.compile(
    r'(?:'
    r'\b(?:'
    r'compliance\s+disclosure|'
    r'regulatory\s+(?:disclosure|filing|notice)|'
    r'form\s+adv\b|'
    r'CA[-\s]AB[-\s]1305|'
    r'privacy\s+(?:notice|policy|statement)\b|'
    r'terms\s+(?:of\s+service|and\s+conditions)\b|'
    r'annual\s+report\s+to\s+(?:shareholders|investors)\b|'
    r'proxy\s+statement\b|'
    r'material\s+changes?\s+(?:to|in)\s+(?:our|the)\s+(?:firm|business|services)\b'
    r')\b'
    # Chinese compliance/legal document signals (免责声明-only docs, regulatory filings)
    r'|招股(?:说明)?书'      # prospectus
    r'|年度报告(?:全文)?$'   # annual report (standalone)
    r'|(?:信息披露|信披)公告' # regulatory disclosure notice
    r'|隐私(?:政策|声明)'    # privacy policy
    r')',
    re.I,
)

# Filename patterns for non-research PDFs fetched via link extractor.
_COMPLIANCE_FILENAME_RE = re.compile(
    r'(?:'
    r'compliance[-_\s]disclosure|'
    r'regulatory[-_\s](?:disclosure|notice|filing)|'
    r'privacy[-_\s](?:notice|policy)|'
    r'terms[-_\s](?:of[-_\s]service|and[-_\s]conditions)|'
    r'ca[-_]ab[-_]\d{4}|'          # California AB bills
    r'form[-_\s]adv|'
    r'annual[-_\s]report[-_\s](?:to|for)|'
    r'proxy[-_\s]statement|'
    r'disclaimer[-_\s]only'
    r')',
    re.I,
)


def is_likely_research_report(
    subject: str,
    text_body: Optional[str],
    has_pdf: bool,
    from_known_broker: bool = False,
) -> tuple:
    """
    Lightweight non-LLM triage: is this email likely a broker research report?

    Returns (passed: bool, reason: str). reason is empty string when passed=True.

    Design principle: false negatives (dropping real research) are far more costly
    than false positives (processing non-research). Be conservative with exclusions.

    - has_pdf → True, UNLESS the email is a pure invitation with no research signal
    - from_known_broker → only block on hard non-research content or pure invitations
    - unknown sender → require at least one positive research signal to proceed
    """
    # Scan subject + generous body window for all signal checks
    text = f"{subject or ''} {(text_body or '')[:3000]}"

    # Hard block: trade confirmations, account statements, etc. — reject regardless
    if _HARD_NON_REPORT_RE.search(text):
        return False, "hard_block"

    # Subject-level hard invitation block: fires even when _REPORT_RE matches in the body.
    # Broker email footers always contain "Analyst"/"RESEARCH"/"Note", which normally
    # neutralises the invitation check below. Checking the subject alone avoids that trap.
    if _HARD_INVITATION_SUBJECT_RE.search(subject or ''):
        return False, "invitation_subject"

    # Pure event/invitation: reject even from known brokers when there's no research signal.
    # E.g. "You're invited to our Asia Macro Forum" with no ratings/targets.
    if _INVITATION_RE.search(text) and not _REPORT_RE.search(text):
        return False, "invitation_pattern"

    # PDF attachment present and no hard block → accept (most broker PDFs are reports)
    if has_pdf:
        return True, ""

    # Known broker domain: trust it unless blocked above
    if from_known_broker:
        return True, ""

    # Unknown sender: require at least one positive research signal
    if _REPORT_RE.search(text):
        return True, ""

    if _NON_REPORT_RE.search(text):
        return False, "non_report_pattern"

    return True, ""


def is_research_pdf_content(first_page_text: str, filename: str = "") -> bool:
    """
    Return False if a downloaded PDF is a compliance/regulatory document rather
    than a broker research report. Called after Docling extracts page 1 text.

    Args:
        first_page_text: Text from the PDF's first page (from slice_pdf_pages_to_aoim).
        filename: The PDF filename (from Content-Disposition or URL).
    """
    # Filename-based fast check
    if filename and _COMPLIANCE_FILENAME_RE.search(filename):
        return False

    # Content-based check on first page
    check_text = first_page_text[:1500]
    if _COMPLIANCE_PDF_RE.search(check_text):
        # One more gate: if the PDF also contains research signals, allow it
        # (some research reports have compliance sections on page 1)
        if not _REPORT_RE.search(check_text):
            return False

    return True
