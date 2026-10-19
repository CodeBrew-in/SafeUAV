# Changelog

## 0.3.5 - Phase 3.5: public demo

- Public mode (`SAFEUAV_PUBLIC=1`): public host only, HTTPS origins, HSTS, listens on the host's `PORT`, `/healthz`.
- Rate limits (chat per drone and per address, flight starts per address, drone cap) and one daily AI budget with an automatic fall-back
  to the basic parser. The page shows an AI status chip and a "public demo" tag.
- Only the hosted AI is offered in public mode; local models are not.
- Render blueprint (`render.yaml`) and a deploy guide (`docs/deploy_render.md`).
- Fixes from hand testing: AI-read spelling in "land at <place>", "divert to X", "don't return home", End mission drops unreached targets,
  ragged building footprints are clickable, Space works after a button click.

## 0.3.0 - Phase 3: the console

- A local web console: vector map of the Thapar campus (OpenStreetMap), pan/zoom, click cards, drawn round areas,
  Start / Pause / Reset, speed 1×/2×/5×, chat column, mission and log tabs.
- Session layer: a draft before Start, one desired mission, chat replies merged as diffs, hold while the AI reads,
  late AI replies dropped.
- Operator controls that work without the AI: Return now, Land now, **Land here** (a chosen pad, shield-checked), End mission.
  They are never rate-limited and never wait for the AI.
- Downloads: HTML flight report, JSONL audit log, track CSV.
- One drone per browser tab. No accounts.
- A secured local server: Host, Origin and token checks, strict JSON, framing protection, exclusive port, connection limits.
- Packaging: `run.ps1`, pinned `requirements.txt` / `requirements-dev.txt`, optional `safeuav.toml`, `scripts/check.ps1`.
- Docs: README rewrite with screenshots, architecture, demo script, threat-model rows for the console, corrected Phase 2 report wording.
- Reviewed by seven independent review passes (front end, backend stress, live UX, security, product, logic, QA) and a final live pass; the
  findings that mattered are fixed and covered by tests (about 290 tests).

Known gaps: midnight time windows on restrictions; the pad-route and runtime-monitor margins differ slightly; the basic parser misreads
"avoiding" and "don't inspect"; the configured wind does not feed the preflight forecast; the OSM snapshot hash is not checked at startup;
no front-end automated tests; no LICENSE file yet.

## 0.2.0 - Phase 2: real campus, benchmark and report

Plain-English orders on the real campus, mid-flight changes, safe-hold, attacks, a frozen benchmark and the written report.

## 0.1.0 - Phase 1: grid world

The 12×12 grid-world study with a safety shield and a local LLM planner.
