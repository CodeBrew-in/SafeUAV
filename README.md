# SafeUAV-Lite

Type a drone order in plain English. An AI turns it into a small, checkable mission. A deterministic
**safety shield** checks every route before the drone flies it. Nothing you type can lift the safety rules.

> **Simulation only.** The drone, wind and battery are simulated on the real Thapar campus map
> (OpenStreetMap). This is a research prototype, not flight software.

![Console: a mission in flight, with a drawn area and a restricted building](docs/img/console-flight.jpg)

![An attack message refused, the drone keeps its plan](docs/img/console-attack-refused.jpg)

## What it does

- **Map:** a clean vector map of the campus (roads, buildings, grass, lake). Pan, zoom, click.
- **Plan by clicking or chatting:** pick targets and restricted buildings, or draw a round area
  (press **Area**, drag outward). Type orders such as "inspect the library, then the directorate, and stay away from E Hostel".
- **Nothing starts by itself:** press **Start**. Then watch the drone, its planned route (dashed) and where it has flown.
- **Change the plan in flight:** clicks and chat both work. The drone replans from where it is. If the new order
  is impossible, it drops the old route, lands on a pad and waits.
- **Always in control:** Pause, speed (1×, 2×, 5×), **Return now**, **Land now**, **Land here** (click a pad), **End mission**.
  These work even while the AI is thinking.
- **Download your flight:** an HTML report, a JSONL audit log and a track CSV (Log tab).
- **No accounts, one drone per browser tab.**

## Quick start (Windows)

Needs Python 3.11 or newer.

```powershell
.\run.ps1
```

It creates `.venv`, installs the pinned packages, starts the console and opens http://127.0.0.1:8000.
Options: `-Port 8001`, `-NoBrowser`. Settings (port, default model, battery, wind, speed) go in
[`safeuav.toml`](safeuav.toml); every line is optional.

**The AI is optional.** Pick a model in the Mission tab:

| Mode / model | Needs |
|---|---|
| `Classic: no AI` | nothing. A basic parser understands simple orders ("inspect the library and the directorate"). |
| `gemini-3.5-flash-lite` | a Google AI Studio key in the `GEMINI_API_KEY` environment variable (on Windows, also read from your user environment). The key is never written to files, logs or the screen. |
| `qwen2.5:7b` / `qwen2.5:3b` | [Ollama](https://ollama.com) running locally with that model pulled (`ollama pull qwen2.5:3b`). |

If the AI is down or slow, the basic parser takes over and the chat says so. It understands much less than the AI.

## Try this (5 minutes)

[docs/demo_script.md](docs/demo_script.md) walks through a flight, a mid-flight change, an attack message,
an impossible order, and the downloads.

## Put it on the internet (free)

A public demo link on Render: see [docs/deploy_render.md](docs/deploy_render.md). It uses your Gemini key as a host secret, shares a daily
AI budget between visitors, rate-limits chat and flights, and falls back to the basic parser when the budget is used up.

## How it stays safe

```
words ─► AI ─► mission (targets, restricted areas, return?) ─► validate ─► A* route ─► shield ─► drone
```

- The AI only writes a small mission. It never writes coordinates and never touches the control loop.
- The trusted rules are constants in code: the permanent no-fly zone (Central Jail Patiala plus buffer), the
  flight-area fence, the battery reserve, the list of landing pads. The mission format has no field for them,
  so no message can lift them.
- Every route (including mid-flight replans and "Land here") passes the same shield before it replaces the current one.
- **The shield enforces the permanent rules, not the operator's intent.** If the AI drops a restriction you asked for,
  the drone may legally fly through it. Read the mission in the Mission tab before pressing Start.

Details: [docs/architecture.md](docs/architecture.md), [docs/threat_model.md](docs/threat_model.md).

## Results (Phase 2 benchmark)

On 20 held-out orders, 10 mid-flight changes and 15 attacks: Gemini-lite understood 100% of orders end to end, the
local 3b model 75%, the no-AI parser 35%. No shielded flight entered the no-fly zone (0 of 17-18 flights per arm;
the 95% upper bound is about 18%, so the property tests and the mutation check carry the safety argument).
Small sample sizes: treat differences under about 20 points as unproven. Report: [docs/geo_report.md](docs/geo_report.md),
tables: [docs/geo_results.md](docs/geo_results.md). The earlier 12×12 grid-world study is described in
[docs/review_packet.md](docs/review_packet.md).

## Limits

- Simulation: flat 2D world, simple wind and battery models, no GPS error, no building heights, no other aircraft.
- Single user per tab, in-memory only (a restart clears everything), plain HTTP on 127.0.0.1, polling not websockets.
  Not a production server.
- The token that protects the API is in the page, so it guards against other web pages, not other programs on your computer.
- Place names come from an OpenStreetMap snapshot (`benchmark/geo/thapar_osm.xml`); some buildings are missing or duplicated.
- Latency numbers come from one laptop. The AI models change over time; the Gemini rows cannot be re-run bit-for-bit.

## Developing

```powershell
.\scripts\check.ps1          # JavaScript syntax check + the whole test suite (about 3 minutes)
.\scripts\check.ps1 -Fast    # skips the two slowest tests
```

```
backend/app/geo/         console: server, session, controller, shield (flight.py), AI layer, map, exports
backend/app/geo/static/  the single-page console (plain JavaScript and CSS, no build step)
backend/tests/           pytest suite (property tests, mutation check, session and HTTP tests)
benchmark/geo/           frozen missions and the pinned OSM snapshot
results/                 saved benchmark runs
docs/                    report, results, architecture, threat model, demo script
backend/app/             the original grid-world study (phase 1)
```

Pinned packages: `requirements.txt` (run) and `requirements-dev.txt` (tests, plots). Tested on Python 3.12, Windows 11.

## Credits

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL).
Motivated by *LLM and AI Agents for Autonomous Systems: A Survey of Applications, Datasets, and Security Challenges*.
