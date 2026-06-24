"""
Shared Gemini API rate limiter.

Enforces two limits:
  - RPM: minimum gap between consecutive calls so we stay under requests/minute
  - RPD: daily request cap persisted to disk so restarts don't reset the count

Limits default to 80% of the free-tier ceiling (15 RPM / 1500 RPD) for a
safety margin. Override with env vars GEMINI_RPM_LIMIT / GEMINI_RPD_LIMIT
if you are on a paid plan with higher quotas.
"""
import datetime
import json
import os
import time
from pathlib import Path

_STATE_FILE = Path(__file__).parent / ".gemini_rate_state.json"

RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "10"))   # calls per minute
RPD_LIMIT = int(os.getenv("GEMINI_RPD_LIMIT", "1400"))  # calls per day


class GeminiRateLimiter:
    def __init__(self, rpm: int = RPM_LIMIT, rpd: int = RPD_LIMIT):
        self.rpm = rpm
        self.rpd = rpd
        self._min_interval = 60.0 / rpm  # seconds between calls
        self._last_call = 0.0
        self._state = self._load()

    def _load(self) -> dict:
        today = datetime.date.today().isoformat()
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                if data.get("date") == today:
                    return data
            except Exception:
                pass
        return {"date": today, "count": 0}

    def _save(self) -> None:
        try:
            _STATE_FILE.write_text(json.dumps(self._state))
        except Exception:
            pass

    def wait(self) -> None:
        """
        Block until it is safe to make the next Gemini API call, then record it.
        Raises RuntimeError if the daily cap has already been reached.
        """
        today = datetime.date.today().isoformat()
        if self._state["date"] != today:
            self._state = {"date": today, "count": 0}

        if self._state["count"] >= self.rpd:
            raise RuntimeError(
                f"Daily Gemini quota ({self.rpd} requests/day) exhausted. "
                "Quota resets at midnight Pacific Time — try again tomorrow."
            )

        gap = self._min_interval - (time.time() - self._last_call)
        if gap > 0:
            time.sleep(gap)

        self._last_call = time.time()
        self._state["count"] += 1
        self._save()

        remaining = self.rpd - self._state["count"]
        print(
            f"   [quota] call {self._state['count']}/{self.rpd} today  "
            f"({remaining} remaining)",
            flush=True,
        )


# Module-level singleton — shared across all imports within one process.
limiter = GeminiRateLimiter()
