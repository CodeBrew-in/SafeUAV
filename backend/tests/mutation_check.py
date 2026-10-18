"""Mutation check: plant known bugs in the core and confirm the stress tests catch each one.

Run from backend/:  ../.venv/Scripts/python tests/mutation_check.py
Every line should say CAUGHT. Output is also saved to docs/mutation_check_output.txt. A line saying MISSED means the tests have a blind spot.
Source files are always restored, even if the run is interrupted.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MUTANTS = [
    ("shield ignores restricted cells", "app/safety_shield.py", "if cell in c.restricted:", "if False:"),
    ("shield ignores obstacles", "app/safety_shield.py", "if cell in c.obstacles:", "if False:"),
    ("shield ignores map edges", "app/safety_shield.py", "if not c.in_bounds(cell):", "if False:"),
    ("shield allows diagonal moves", "app/safety_shield.py", "abs(a[1] - b[1]) == 1", "abs(a[1] - b[1]) <= 2"),
    ("shield budget off by one", "app/safety_shield.py", "if steps > c.max_steps:", "if steps > c.max_steps + 1:"),
    ("shield skips return-to-base", "app/safety_shield.py", "if cells[-1] != end:", "if False:"),
    ("shield skips start check", "app/safety_shield.py", "if cells[0] != c.start:", "if False:"),
    ("shield skips destination check", "app/safety_shield.py", "if c.destination not in cells:", "if False:"),
    ("unsafe ignores budget overrun", "app/safety_shield.py",
     "return bool(self.codes & SAFETY_VIOLATIONS)",
     "return bool(self.codes & (SAFETY_VIOLATIONS - {ViolationCode.BUDGET_EXCEEDED}))"),
    ("jumps not swept for forbidden cells", "app/safety_shield.py", "swept += [", "_unused = ["),
    ("runner counts parse failure as shield rejection", "app/runner.py",
     "res.shield_intervened = cause in SHIELD_CAUSES", "res.shield_intervened = True"),
    ("runner misses false infeasible", "app/runner.py",
     "res.false_infeasible = bool(res.ground_truth_feasible)", "res.false_infeasible = False"),
    ("A* forgets return leg", "app/astar.py",
     "route = outbound + outbound[-2::-1] if c.return_to_base else outbound", "route = outbound"),
    ("A* budget off by one", "app/astar.py", "if steps > c.max_steps:", "if steps > c.max_steps + 1:"),
    ("traversable ignores restricted", "app/mission.py",
     "return cell in self.obstacles or cell in self.restricted", "return cell in self.obstacles"),
]


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "mutation_check_output.txt"


def main() -> int:
    lines: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    missed = 0
    for name, path, old, new in MUTANTS:
        f = Path(path)
        original = f.read_bytes().decode()
        if old not in original:
            say(f"SKIPPED  {name} (pattern not found)")
            missed += 1
            continue
        try:
            f.write_bytes(original.replace(old, new, 1).encode())
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                               capture_output=True, text=True)
        finally:
            f.write_bytes(original.encode())  # bytes: keep line endings exactly
        caught = r.returncode != 0
        missed += not caught
        say(f"{'CAUGHT ' if caught else 'MISSED '}  {name}")
    say(f"\n{len(MUTANTS) - missed}/{len(MUTANTS)} planted bugs caught")
    OUTPUT.write_text(f"Mutation check run {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
                      + "\n".join(lines) + "\n", encoding="utf-8")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
