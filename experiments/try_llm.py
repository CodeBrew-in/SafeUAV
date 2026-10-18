"""Run every demo scenario through all three methods with the live local model and print what happened.

Usage (from the repo root):
    .venv/Scripts/python experiments/try_llm.py [--model qwen2.5:3b] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.llm_planner import DEFAULT_MODEL, OllamaPlanner  # noqa: E402
from app.mission import build_mission  # noqa: E402
from app.runner import Method, run_astar, run_llm  # noqa: E402
from app.scenarios import load_scenarios  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", type=Path, help="also save full run records here")
    args = ap.parse_args()

    planner = OllamaPlanner(model=args.model)
    if not planner.is_available():
        print(f"Ollama is not running or model {args.model} is not installed.", file=sys.stderr)
        return 1

    records = []
    print(f"{'scenario':<20} {'method':<11} {'status':<24} {'proposal':<10} {'steps':>5} {'llm s':>6}  reason")
    for s in load_scenarios().values():
        ctx = build_mission(s)
        for method in Method:
            r = run_astar(ctx) if method is Method.ASTAR else run_llm(ctx, method, planner)
            v = r.proposal_validation
            if method is Method.ASTAR:
                proposal = "-"
            elif r.llm and r.llm["status"] == "infeasible":
                proposal = "infeasible"
            elif r.llm and (r.llm["parse_error"] or r.llm["transport_error"]):
                proposal = "no route"
            else:
                proposal = "valid" if v and v.valid else "invalid"
            steps = len(r.executed_trail) - 1 if r.executed_trail else "-"
            llm_s = f"{r.proposal_latency_ms / 1000:.1f}" if r.proposal_latency_ms else "-"
            llm = r.llm or {}
            if llm.get("parse_error") or llm.get("transport_error"):
                why = llm.get("parse_error") or llm.get("transport_error")
            elif v and not v.valid:
                why = v.reason
            else:
                why = llm.get("reason") or ""
            why = why[:90]
            print(f"{s.id:<20} {method.value:<11} {r.status.value:<24} {proposal:<10} {steps:>5} {llm_s:>6}  {why}")
            records.append(asdict(r))

    if args.json:
        args.json.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved {len(records)} run records to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
