"""Run the 60-scenario benchmark: A* once per scenario, and one model call per (scenario, seed)
scored under llm_only, llm_shield and llm_shield_no_fallback.

Resumable: (scenario, seed) pairs already in the output file are skipped, so the run can be
stopped with Ctrl-C and started again with the same command.

Usage (repo root):
    .venv/Scripts/python experiments/run_benchmark.py --seeds 3 --out results/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.llm_planner import DEFAULT_MODEL, OllamaPlanner  # noqa: E402
from app.mission import build_mission  # noqa: E402
from app.records import append_jsonl, environment, read_jsonl, to_record  # noqa: E402
from app.runner import run_astar, run_llm_arms  # noqa: E402
from app.scenarios import load_scenarios  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenarios", type=Path, default=ROOT / "benchmark" / "scenarios.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark.jsonl")
    args = ap.parse_args()

    planner = OllamaPlanner(model=args.model)
    if not planner.is_available():
        print(f"Ollama is not running or model {args.model} is not installed.", file=sys.stderr)
        return 1
    env = environment(args.model, planner.host)
    if env["git_commit"] and env["git_commit"].endswith("-dirty"):
        print("WARNING: uncommitted changes; records will be marked -dirty", file=sys.stderr)

    done_llm, done_astar = set(), set()
    if args.out.exists():
        for r in read_jsonl(args.out):
            if r["method"] == "astar":
                done_astar.add(r["scenario_id"])
            else:
                done_llm.add((r["scenario_id"], r["seed"]))

    scenarios = list(load_scenarios(args.scenarios).values())
    total = len(scenarios) * args.seeds
    n = len(done_llm)
    print(f"model={args.model} gpu={env['gpu']} commit={env['git_commit']}  "
          f"{len(scenarios)} scenarios x {args.seeds} seeds = {total} model calls ({n} already done)")
    t_start = time.perf_counter()
    for s in scenarios:
        ctx = build_mission(s)
        if s.id not in done_astar:
            append_jsonl(args.out, [to_record(run_astar(ctx), ctx, env)])
        for seed in range(args.seeds):
            if (s.id, seed) in done_llm:
                continue
            arms = run_llm_arms(ctx, planner, seed=seed)
            append_jsonl(args.out, [to_record(r, ctx, env) for r in arms.values()])
            n += 1
            only = next(r for r in arms.values() if r.method.value == "llm_only")
            elapsed = time.perf_counter() - t_start
            print(f"[{n:>3}/{total}] {s.id:<24} seed {seed}  {only.status.value:<26} "
                  f"{(only.proposal_latency_ms or 0) / 1000:5.1f}s  (elapsed {elapsed / 60:4.1f} min)", flush=True)
    print(f"\nDone. Records in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
