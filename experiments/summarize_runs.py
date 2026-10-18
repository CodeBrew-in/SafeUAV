"""Print a results table from a JSONL file of run records.

Usage: .venv/Scripts/python experiments/summarize_runs.py results/demo_runs_XXXX.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.records import read_jsonl  # noqa: E402


def print_summary(path: Path) -> None:
    rows = read_jsonl(path)
    env = rows[0]["env"] if rows else {}
    print(f"Source: {path}  ({len(rows)} records)")
    print(f"Model {env.get('model')}, Ollama {env.get('ollama_version')}, GPU {env.get('gpu')}, "
          f"commit {env.get('git_commit')}\n")
    header = (f"{'scenario':<20} {'seed':>4} {'method':<23} {'status':<25} {'cause':<32} "
              f"{'flown':>5} {'viol':>4} {'llm s':>6} {'try':>3}")
    print(header)
    print("-" * len(header))
    for r in rows:
        seed = "-" if r["seed"] is None else r["seed"]
        flown = "-" if r["flown_route_length"] is None else r["flown_route_length"]
        viol = "-" if r["executed_violations"] is None else r["executed_violations"]
        llm_s = "-" if r["proposal_latency_ms"] is None else f"{r['proposal_latency_ms'] / 1000:.1f}"
        tries = r["llm_attempts"] if r["llm_attempts"] else "-"
        print(f"{r['scenario_id']:<20} {seed:>4} {r['method']:<23} {r['status']:<25} "
              f"{r['rejection_cause'] or '':<32} {flown:>5} {viol:>4} {llm_s:>6} {tries:>3}")
    print("\nflown = steps actually flown; viol = violations measured on the flown trail; "
          "llm_only routes are never flown.")


if __name__ == "__main__":
    print_summary(Path(sys.argv[1]))
