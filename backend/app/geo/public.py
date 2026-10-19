"""Public-demo mode (Phase 3.5). Off by default: locally the console binds 127.0.0.1 only.

Turned on by environment variables on the host (Render sets PORT and RENDER_EXTERNAL_HOSTNAME itself):

  SAFEUAV_PUBLIC=1                 switch on
  SAFEUAV_PUBLIC_HOST=<name>       the public host name (default: RENDER_EXTERNAL_HOSTNAME)
  PORT                             the port to listen on (set by the host)
  SAFEUAV_MAX_DRONES=20            drones (browser tabs) at once
  SAFEUAV_CHAT_PER_MIN=6           chat messages per drone per minute
  SAFEUAV_CHAT_PER_IP_MIN=40       chat messages per client address per minute
  SAFEUAV_STARTS_PER_HOUR=60       flights started per client address per hour
  SAFEUAV_AI_DAILY_BUDGET=200      AI calls per day for everyone; after that the basic parser answers
  GEMINI_API_KEY                   the AI key. Set it as a secret on the host. It is never printed or stored.

Nothing here can touch the trusted safety constants.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from .llm import LLMReply


class PublicConfigError(Exception):
    pass


@dataclass(frozen=True)
class PublicConfig:
    enabled: bool = False
    host: str = ""
    bind: str = "127.0.0.1"
    port: int | None = None
    max_drones: int = 16
    chat_per_min: int = 6
    chat_per_ip_min: int = 40
    starts_per_hour: int = 60
    ai_daily_budget: int = 200


def _int(env: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise PublicConfigError(f"{name} must be a whole number.") from None
    if not lo <= v <= hi:
        raise PublicConfigError(f"{name} must be between {lo} and {hi}.")
    return v


def from_env(env=None) -> PublicConfig:
    env = os.environ if env is None else env
    if env.get("SAFEUAV_PUBLIC") not in ("1", "true", "yes"):
        return PublicConfig()
    host = (env.get("SAFEUAV_PUBLIC_HOST") or env.get("RENDER_EXTERNAL_HOSTNAME") or "").strip().lower()
    if not host or "/" in host or " " in host or ":" in host:
        raise PublicConfigError("Public mode needs SAFEUAV_PUBLIC_HOST (a host name such as demo.onrender.com).")
    return PublicConfig(
        enabled=True, host=host, bind="0.0.0.0", port=_int(env, "PORT", 10000, 1024, 65535),
        max_drones=_int(env, "SAFEUAV_MAX_DRONES", 20, 1, 200),
        chat_per_min=_int(env, "SAFEUAV_CHAT_PER_MIN", 6, 1, 60),
        chat_per_ip_min=_int(env, "SAFEUAV_CHAT_PER_IP_MIN", 40, 1, 600),
        starts_per_hour=_int(env, "SAFEUAV_STARTS_PER_HOUR", 60, 1, 1000),
        ai_daily_budget=_int(env, "SAFEUAV_AI_DAILY_BUDGET", 200, 0, 100000))


class Limiter:
    """Sliding-window counters, one per (kind, who). Thread-safe, and old keys are dropped."""

    def __init__(self):
        self._hits: dict[tuple[str, str], deque] = {}
        self._lock = threading.Lock()

    def allow(self, kind: str, who: str, limit: int, window_s: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            dq = self._hits.setdefault((kind, who), deque())
            while dq and now - dq[0] > window_s:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            if len(self._hits) > 5000:               # bound the memory a flood of addresses can use
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] > window_s][:2500]:
                    self._hits.pop(k, None)
            return True


class AiBudget:
    """Calls to the AI per UTC day, for everyone together. Protects the free API quota."""

    def __init__(self, per_day: int, clock=time.time):
        self.per_day, self._clock = per_day, clock
        self._day, self._used = self._today(), 0
        self._lock = threading.Lock()

    def _today(self) -> int:
        return int(self._clock() // 86400)

    def take(self) -> bool:
        with self._lock:
            if self._today() != self._day:
                self._day, self._used = self._today(), 0
            if self._used >= self.per_day:
                return False
            self._used += 1
            return True

    def left(self) -> int:
        with self._lock:
            if self._today() != self._day:
                return self.per_day
            return max(0, self.per_day - self._used)


class BudgetLLM:
    """Wraps a real model client. When the daily budget is spent it answers with an error, and the session
    falls back to the basic parser (the chat says so)."""

    def __init__(self, inner, budget: AiBudget):
        self.inner, self.budget = inner, budget
        self.provider = getattr(inner, "provider", "ai")
        self.model = getattr(inner, "model", "")

    def ask(self, system, user, schema, seed=0, max_tokens=8192):
        if not self.budget.take():
            return LLMReply(self.provider, self.model, None, None, 0.0, "the daily AI limit for this demo is used up")
        return self.inner.ask(system, user, schema, seed=seed, max_tokens=max_tokens)
