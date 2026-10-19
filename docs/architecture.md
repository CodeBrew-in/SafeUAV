# Architecture

```
browser tab (plain JS, canvas map)
   │  polls /api/state every 250 ms (delta feed); sends clicks, chat, controls
   ▼
server.py  ── Host / Origin / token / content-type / size checks, one Session per X-Drone-Id
   ▼
session.py ── draft before Start, one "desired" mission, pause / speed / hold, drawn areas, safety commands
   │   chat ─► operator.py + llm.py (AI or basic parser) ─► a mission diff, merged into the current mission
   ▼
controller.py ── state machine (PREFLIGHT, CRUISE, ESCAPE, TO_PAD, LANDED_WAIT, EMERGENCY, DONE, ...)
   │   spec.py validate ─► flight.py A* planner ─► preflight shield ─► route
   ▼
simulated drone (1 s steps, wind, battery)
```

## Modules (`backend/app/geo/`)

| File | Job |
|---|---|
| `campus.py`, `basemap.py` | Places, pads, no-fly zone and fence from the pinned OSM snapshot; the drawing data for the map |
| `spec.py` | The mission format (targets, restricted areas with optional time windows, stay at end) and `validate_spec` |
| `flight.py` | Airspace (zones + buffers), grid A*, shortcutting, energy and wind model, the preflight shield |
| `controller.py` | The flight state machine, replanning, runtime battery monitor, safe-hold, `land_plan` (land at a chosen pad) |
| `operator.py`, `llm.py` | Turn text into a mission: strict keywords, the AI (Gemini or Ollama), the basic parser |
| `session.py` | One operator's draft and flight, all rules for editing them, drawn areas, downloads' data |
| `exports.py` | JSONL audit log, track CSV, HTML report (fixed template, escaped values) |
| `server.py` | The HTTP server and the per-tab drone registry; `settings.py` reads `safeuav.toml` |
| `static/` | The console |

## Rules that hold everywhere

1. **The AI's output is data.** It is validated, and only a valid mission reaches the planner.
2. **Trusted constants live in code** (`campus.py`, `flight.py`): no-fly zone and buffer, fence, reserve fraction, pad list. No request, setting
   or mission field can change them.
3. **Every route is shield-checked** before it replaces the current one: mission replans, return, landing at the nearest or a chosen pad.
4. **An invalid order never changes the flight.** The current valid route stays.
5. **Safety commands are never held up.** Land, Return and End are not rate-limited, work while the AI is thinking, and cancel any AI reply
   still on its way.
6. **One desired mission.** Clicks are patches; chat replies are diffs against the mission the AI was shown, applied to the current mission, so a
   slow reply cannot erase a click.

## Data flow of one chat message

1. `handle()` first checks whole-message commands ("land now", "end mission", "land at <pad>"). They never reach the AI.
2. Otherwise the drone holds position (at most 20 simulated seconds, at hover battery cost) while the AI reads the message.
3. The AI reply becomes a mission diff, merged into the current mission, validated, then submitted to the controller.
4. Any failure (AI down, slow, malformed) falls back to the basic parser, and the chat says so.

## Downloads

`/api/logs/audit.jsonl`, `/api/logs/track.csv`, `/api/logs/report.html` return the caller's own drone only, as attachments. The report is built from
recorded facts (summary, mission, settings, SVG track, events, chat) with a simulation-only notice. No generated text is added.
