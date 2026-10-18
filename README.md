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

## Local model

Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:3b
```

The planner (`backend/app/llm_planner.py`) asks for strict JSON using Ollama's
structured-output schema, caps output length based on the movement budget, and retries
once on unparseable output (the retry samples at temperature 0.4 because temperature 0
repeats itself). Raw responses are kept in each run record.

Try every demo scenario with all three methods against the live model:

```bash
.venv/Scripts/python experiments/try_llm.py
```

## Run the tests

```bash
cd backend
../.venv/Scripts/python -m pytest -q
```

The one live-model test is skipped automatically when Ollama is not running.

## Layout

```
backend/app/       mission model, A*, shield, simulator, run pipeline
backend/tests/     pytest suite
experiments/       scripts that run scenarios against the live model
benchmark/         scenario definitions (demo set for now)
```

## Inspiration

Motivated by *LLM and AI Agents for Autonomous Systems: A Survey of Applications,
Datasets, and Security Challenges*. SafeUAV-Lite is our own implementation, inspired by
its discussion of prompt injection, agent-level security and closed-loop evaluation.
