# SafeUAV-Lite

A research prototype that tests whether a deterministic **safety shield** can stop an
LLM-based drone mission planner from executing unsafe routes in a simulated 12×12 grid
world, without needlessly reducing mission completion.

> Everything runs in simulation. There is no physical drone control.

## Status

| Milestone | Contents | State |
|---|---|---|
| 1 | Grid simulator, A\*, safety shield, run pipeline, tests | done |
| 2 | Local LLM planner (Ollama, `qwen2.5:3b`) | next |
| 3 | Shield/fallback wired to live LLM output | — |
| 4 | Website (React + TypeScript) | — |
| 5 | 60-scenario benchmark, 3 methods, results CSV | — |
| 6 | Methodology, limitations, report | — |

## How it works

```
mission ──► planner proposal ──► safety shield ──► simulator ──► recorded run
            (LLM or A*)            │ reject
                                   └──► A* fallback ──► shield again ──► simulator
```

* **Trusted constraints** (map, obstacles, restricted cells, budget, return-to-base) come
  only from the scenario config. The untrusted note goes to the LLM prompt and nowhere
  else: the shield, A\* and the simulator never see it.
* **The shield** (`backend/app/safety_shield.py`) checks the whole route before
  execution: shape of the output, start cell, map bounds, obstacles, restricted cells,
  single up/down/left/right moves, reaching the destination, returning to base, and the
  full round-trip budget. It reports every violation, not just the first.
* **The simulator** (`backend/app/simulator.py`) only accepts an `ApprovedRoute`,
  re-validates it, and re-checks every move before applying it.
* **Methods** (`backend/app/runner.py`): `astar`, `llm_only` (proposal is measured and
  shown as a *hypothetical* trajectory, never executed), and `llm_shield`.

Coordinates are `(x, y)`: x is the column, y is the row, both starting at 0.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

## Run the tests

```bash
cd backend
../.venv/Scripts/python -m pytest -q
```

## Layout

```
backend/app/       mission model, A*, shield, simulator, run pipeline
backend/tests/     pytest suite
benchmark/         scenario definitions (demo set for now)
```

## Inspiration

Motivated by *LLM and AI Agents for Autonomous Systems: A Survey of Applications,
Datasets, and Security Challenges*. SafeUAV-Lite is our own implementation, inspired by
its discussion of prompt injection, agent-level security and closed-loop evaluation.
