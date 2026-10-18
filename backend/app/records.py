"""Flat, JSON-serialisable run records. One record = one (scenario, seed, method) run.

Analysis scripts read these records back from JSONL files; nothing in a results
table should be typed by hand.
"""

from __future__ import annotations

import json
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .mission import MissionContext
from .runner import RunResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except OSError:
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except OSError:
        return None


def _ollama_version(host: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=3) as resp:
            return json.loads(resp.read().decode()).get("version")
    except (OSError, ValueError):
        return None


def environment(model: str | None, host: str = "http://localhost:11434") -> dict:
    """Software and hardware the runs were produced on."""
    return {
        "software_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or None,
        "gpu": _gpu_name(),
        "model": model,
        "ollama_version": _ollama_version(host) if model else None,
    }


def to_record(r: RunResult, ctx: MissionContext, env: dict) -> dict:
    v = r.proposal_validation

    def cells(route):
        return None if route is None else [list(p) for p in route]

    def length(route):
        return None if route is None else len(route) - 1

    llm = r.llm or {}
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": r.run_id,
        "scenario_id": r.scenario_id,
        "condition": ctx.condition.value,
        "method": r.method.value,
        "seed": r.seed,
        "status": r.status.value,
        "ground_truth_feasible": r.ground_truth_feasible,
        "optimal_steps": r.optimal_steps,
        "max_steps": ctx.constraints.max_steps,
        "llm_status": llm.get("status"),
        "proposal_valid": None if v is None or v.malformed else v.valid,
        "proposal_unsafe": r.proposal_unsafe,
        "proposal_mission_invalid": r.proposal_mission_invalid,
        "violation_codes": [] if v is None else v.code_list,
        "validation_reason": None if v is None else v.reason,
        "swept_forbidden": [] if v is None else [list(p) for p in v.swept_forbidden],
        "false_infeasible": r.false_infeasible,
        "rejection_cause": None if r.rejection_cause is None else r.rejection_cause.value,
        "shield_intervened": r.shield_intervened,
        "fallback_used": r.fallback_used,
        "proposed_route_length": length(v.route) if v is not None and v.route is not None else None,
        "flown_route_length": length(r.flown_route),
        "executed_violations": r.executed_violations,
        "route_efficiency": (r.optimal_steps / length(r.flown_route)
                             if r.flown_route and r.optimal_steps and length(r.flown_route) else None),
        "proposal_latency_ms": r.proposal_latency_ms,
        "decision_latency_ms": r.decision_latency_ms,
        "total_latency_ms": r.total_latency_ms,
        "llm_attempts": llm.get("attempts"),
        "llm_attempt_log": llm.get("attempt_log"),
        "llm_parse_error": llm.get("parse_error"),
        "llm_transport_error": llm.get("transport_error"),
        "llm_reason": llm.get("reason"),
        "llm_raw_responses": llm.get("raw_responses"),
        "proposal": r.proposal if isinstance(r.proposal, (list, dict, str, type(None))) else repr(r.proposal),
        "flown_route": cells(r.flown_route),
        "hypothetical_route": cells(r.hypothetical_route),
        "events": [{"t": e.timestamp, "ms": e.elapsed_ms, "kind": e.kind, "message": e.message, "data": e.data}
                   for e in r.events],
        "env": env,
    }


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
