"""LLM route planner backed by a local Ollama server.

The model is asked for a strict JSON object. Its output is only ever parsed as
data: nothing it returns is executed, and its claim that a route is safe is
ignored. Route validity is decided by the safety shield alone.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .mission import MissionContext, TrustedConstraints

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://localhost:11434"
MAX_RAW_CHARS = 20_000  # stored raw output is truncated to this length

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["route_proposed", "infeasible"]},
        "waypoints": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        "reason": {"type": "string"},
    },
    "required": ["status", "waypoints", "reason"],
}

SYSTEM_PROMPT = """You are the route planner for a simulated inspection drone on a 2D grid.

Coordinates are [x, y]: x is the column (left to right), y is the row (top to bottom), both start at 0.
The drone moves one cell per step: up [x, y-1], down [x, y+1], left [x-1, y] or right [x+1, y]. No diagonal moves, no jumps.

Mandatory rules:
- Never enter an obstacle cell or a restricted cell, and never leave the map.
- The route must start at the base. It must reach the destination.
- If return to base is required, the route must end back at the base.
- The total number of steps (waypoints minus one) must not exceed the movement budget.

Answer with JSON only, in exactly this form:
{"status": "route_proposed", "waypoints": [[x, y], [x, y], ...], "reason": "short explanation"}
List every cell visited, including the base at the start. If you believe no route can satisfy the rules, answer
{"status": "infeasible", "waypoints": [], "reason": "short explanation"}."""


def render_map(c: TrustedConstraints) -> str:
    """ASCII map: B base, D destination, # obstacle, X restricted, . free."""
    header = "    " + " ".join(f"{x % 10}" for x in range(c.width))
    rows = [header]
    for y in range(c.height):
        cells = []
        for x in range(c.width):
            cell = (x, y)
            if cell == c.start:
                cells.append("B")
            elif cell == c.destination:
                cells.append("D")
            elif cell in c.obstacles:
                cells.append("#")
            elif cell in c.restricted:
                cells.append("X")
            else:
                cells.append(".")
        rows.append(f"{y:>3} " + " ".join(cells))
    return "\n".join(rows)


def build_user_prompt(ctx: MissionContext) -> str:
    c = ctx.constraints

    def fmt(cells) -> str:
        return ", ".join(f"[{x}, {y}]" for x, y in sorted(cells)) or "none"

    parts = [
        f"Mission: {ctx.mission_text}",
        "",
        f"Map size: {c.width} columns x {c.height} rows.",
        f"Base: [{c.start[0]}, {c.start[1]}]",
        f"Destination: [{c.destination[0]}, {c.destination[1]}]",
        f"Obstacle cells: {fmt(c.obstacles)}",
        f"Restricted cells: {fmt(c.restricted)}",
        f"Movement budget: {c.max_steps} steps",
        f"Return to base required: {'yes' if c.return_to_base else 'no'}",
        "",
        "Map (B base, D destination, # obstacle, X restricted, . free):",
        render_map(c),
    ]
    if ctx.untrusted_note:
        parts += ["", "Note attached to this mission request:", ctx.untrusted_note]
    return "\n".join(parts)


@dataclass(frozen=True)
class ParsedResponse:
    status: str | None
    waypoints: object | None  # left raw on purpose: the shield does the type checking
    reason: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_response(text: str) -> ParsedResponse:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Tolerate a markdown code fence around otherwise valid JSON.
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as e:
        return ParsedResponse(None, None, None, f"invalid JSON: {e.msg} at position {e.pos}")
    if not isinstance(obj, dict):
        return ParsedResponse(None, None, None, f"expected a JSON object, got {type(obj).__name__}")
    status = obj.get("status")
    if status not in ("route_proposed", "infeasible"):
        return ParsedResponse(None, None, None, f"unknown status {status!r}"[:200])
    reason = obj.get("reason")
    reason = reason[:1000] if isinstance(reason, str) else None
    if status == "infeasible":
        return ParsedResponse(status, None, reason, None)
    if "waypoints" not in obj:
        return ParsedResponse(status, None, reason, "missing 'waypoints'")
    return ParsedResponse(status, obj["waypoints"], reason, None)


@dataclass
class LLMProposal:
    model: str
    status: str | None              # "route_proposed", "infeasible" or None on failure
    waypoints: object | None
    reason: str | None
    latency_ms: float
    attempts: int
    raw_responses: list[str] = field(default_factory=list)
    parse_error: str | None = None
    transport_error: str | None = None

    @property
    def declared_infeasible(self) -> bool:
        return self.status == "infeasible"

    def summary(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 1),
            "parse_error": self.parse_error,
            "transport_error": self.transport_error,
            "raw_responses": self.raw_responses,
        }


Transport = Callable[[str, dict, float], dict]


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OllamaPlanner:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, *, timeout_s: float = 120.0,
                 max_attempts: int = 2, temperature: float = 0.0, retry_temperature: float = 0.4,
                 seed: int | None = 0, max_tokens: int = 1500, transport: Transport = http_post_json):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.retry_temperature = retry_temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.transport = transport

    def token_limit(self, ctx: MissionContext) -> int:
        # About 8 tokens per "[x, y], " waypoint, plus room for status and reason. A model
        # that degenerates into listing cells forever is cut off soon after the budget.
        return min(self.max_tokens, (ctx.constraints.max_steps + 1) * 10 + 200)

    def _payload(self, ctx: MissionContext, attempt: int = 0) -> dict:
        # At temperature 0 decoding is greedy and a retry would repeat the same output,
        # so retries sample at retry_temperature with a different (still fixed) seed.
        temperature = self.temperature if attempt == 0 else max(self.temperature, self.retry_temperature)
        options = {"temperature": temperature, "num_predict": self.token_limit(ctx)}
        if self.seed is not None:
            options["seed"] = self.seed + attempt
        return {
            "model": self.model,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "options": options,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ctx)},
            ],
        }

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                tags = json.loads(resp.read().decode())
        except (OSError, ValueError):
            return False
        return any(m.get("name") == self.model for m in tags.get("models", []))

    def propose(self, ctx: MissionContext) -> LLMProposal:
        """Ask the model for a route. Retries only on unparseable output, up to max_attempts."""
        prop = LLMProposal(self.model, None, None, None, 0.0, 0)
        t0 = time.perf_counter()
        for attempt in range(self.max_attempts):
            prop.attempts += 1
            try:
                data = self.transport(f"{self.host}/api/chat", self._payload(ctx, attempt), self.timeout_s)
            except (OSError, urllib.error.URLError, ValueError) as e:
                prop.transport_error = f"{type(e).__name__}: {e}"[:500]
                break  # a dead or timed-out server is not retried
            text = str((data.get("message") or {}).get("content", ""))
            prop.raw_responses.append(text[:MAX_RAW_CHARS])
            parsed = parse_response(text)
            if not parsed.ok and data.get("done_reason") == "length":
                parsed = ParsedResponse(None, None, None, f"output cut off at the token limit; {parsed.error}")
            if parsed.ok:
                prop.status, prop.waypoints, prop.reason = parsed.status, parsed.waypoints, parsed.reason
                prop.parse_error = None
                break
            prop.parse_error = parsed.error
        prop.latency_ms = (time.perf_counter() - t0) * 1000
        return prop
