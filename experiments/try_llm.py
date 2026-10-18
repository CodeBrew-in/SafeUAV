"""Run every demo scenario through all methods with the live local model and save every run.

Each (scenario, seed) gets ONE model call, scored under llm_only, llm_shield and
llm_shield_no_fallback (paired design). A* runs once per scenario (it is deterministic).
Records go to a JSONL file; the summary table is then printed FROM that file.

Usage (from the repo root):
    .venv/Scripts/python experiments/try_llm.py [--model qwen2.5:3b] [--seeds 3] [--out results/demo_runs.jsonl]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "experiments"))

from app.llm_planner import DEFAULT_MODEL, OllamaPlanner  # noqa: E402
from app.mission import build_mission  # noqa: E402
from app.records import append_jsonl, environment, to_record  # noqa: E402
from app.runner import run_astar, run_llm_arms  # noqa: E402
from app.scenarios import load_scenarios  # noqa: E402
from summarize_runs import print_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, default=1, help="number of draws per scenario (seeds 0..N-1)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / f"demo_runs_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    args = ap.parse_args()

    planner = OllamaPlanner(model=args.model)
    if not planner.is_available():
        print(f"Ollama is not running or model {args.model} is not installed.", file=sys.stderr)
        return 1
    env = environment(args.model, planner.host)
    print(f"model={args.model} ollama={env['ollama_version']} gpu={env['gpu']} commit={env['git_commit']}")

    for s in load_scenarios().values():
        ctx = build_mission(s)
        append_jsonl(args.out, [to_record(run_astar(ctx), ctx, env)])
        for seed in range(args.seeds):
            arms = run_llm_arms(ctx, planner, seed=seed)
            append_jsonl(args.out, [to_record(r, ctx, env) for r in arms.values()])
            any_r = next(iter(arms.values()))
            print(f"  {s.id:<20} seed {seed}: llm {any_r.llm['status'] or 'no output'}, "
                  f"{(any_r.proposal_latency_ms or 0) / 1000:.1f}s")

    print(f"\nSaved run records to {args.out}\n")
    print_summary(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
