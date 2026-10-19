# SafeUAV Phase 2: plain-English drone missions on a real campus

**One line:** the AI understands the order, the algorithm plans it, the shield guarantees safety, and no chat message can change the safety rules.

> 2D simulation on the real Thapar campus map (OpenStreetMap). Wind and battery are simulated. The jail is used as an illustrative no-fly zone, not an official airspace claim.

## What it does

```
operator text ──► LLM ──► MissionSpec ──► validate ──► A* planner ──► preflight shield ──► controller ──► drone
 (chat, any time)        (targets, avoid zones      (names, geometry,   (10 m grid,     (no-fly, geofence,   (state machine,
                          + time windows, stay)      ambiguity)          shortcutting)    order, energy+wind)   runtime monitor)
```

- **The LLM's only job:** turn words into a small, checkable spec. It never writes coordinates and never touches the control loop.
- **Trusted rules are code constants:** the permanent no-fly zone, the geofence and the battery reserve. The spec schema has no field for them, so no message can lift them. See [threat_model.md](threat_model.md).
- **Mid-flight changes:** the drone replans from where it is. A new route replaces the old one only after the shield passes it.
- **If the new order is impossible,** the drone does **not** keep the old route (it may break the new rules). It escapes any forbidden zone, lands on the nearest approved pad, waits (bounded, idle battery drain), and resumes when a feasible order arrives.
- **Runtime monitor:** every 5 s it re-checks the remaining route against the *measured* wind. If the battery won't last, it diverts to a pad.
- **"Return now" / "land now"** work by keyword, without the LLM. Everything is logged to a replayable audit log.

## Results

Full tables with 95% CIs: [geo_results.md](geo_results.md). Generated only from `results/geo_benchmark*.jsonl`.

Held-out test set: 20 orders, 10 mid-flight changes, 15 attacks.

| | regex (no AI) | qwen2.5:3b (ours) | qwen2.5:7b (ours) | Gemini 3.5 Flash-Lite (ours) | gold spec (ceiling) |
|---|---|---|---|---|---|
| Order understood (strict) | 35% | 65% | 45% (60%†) | **100%** | 100% |
| Mission done correctly, end to end | 35% | 75% | 70% | **100%** | 100% |
| Mid-flight change handled | 40% | 90% | 100% | **100%** | 100% |
| Attacks handled as intended | 53% | 53% | 80% | **100%** | – |
| **Flights entering the no-fly zone** (0 of 17-18 flights per column) | **0** | **0** | **0** | **0** | **0** |
| Median AI latency | – | 3.8 s | 6.6 s | 1.3 s | – |

† 7b often adds a redundant "avoid the jail" rule. This column ignores it (post-hoc, flight unchanged).

**Key findings**

1. **The shield held in every run.** Every shielded run, for every parser, model and attack, had 0 flights into the no-fly zone, out of the geofence, or below the reserve. With only 17-18 flights per column, "0" still allows a true failure rate up to about 18% (95% upper bound). The stronger evidence is the property tests and the mutation check, which test the shield on thousands of random maps and planted bugs.
   - Without the shield, 33–40% of attacks on the small/no-AI parsers would have sent a trusting planner into the jail.
   - In direct mode (the AI writes waypoints with no shield), 7b broke a rule in 4/12 attacks.
2. **The AI helps on this set.** Gemini-lite beats the regex baseline 13–0 on paired orders (p = 0.0002, n = 20, so the intervals are wide).
   - 3b beats it 7–1 (p = 0.07, not significant at n = 20).
   - 7b ties with regex on the strict score.
   - On hard phrasing, regex gets 10%; 3b gets 70%; Gemini gets 100%.
3. **Specs beat waypoints.** When the same models write waypoints directly (no shield):
   - 3b gets 15% of missions right, mostly because it runs out of tokens writing coordinates.
   - 7b gets 35%.
   - Gemini-lite gets 85%, but 2/18 of its routes cut through zones the operator said to avoid.
   - Through the spec pipeline, the same Gemini model scores 100% with 0 violations.
4. **Safe-hold works.** On every impossible mid-flight change (2/2 for each AI model), the drone dropped the old route, landed on a pad and waited.
5. **The runtime monitor matters only when the forecast is wrong.** The forecast was 3 m/s on a 20 Wh battery. With 7–7.5 m/s real wind, 16–21 of 68 flights ran out of battery without the monitor, against **0/68** with it.
6. **The limit to know about:** the shield enforces permanent rules, not the operator's intent.
   - If the AI *drops* an avoid zone the operator asked for, the drone can legally fly through it.
   - This happened in 1/20 orders for 3b and 7b, 0 for Gemini, and 4/20 for regex.
   - Mitigation for industry: read the spec back to the operator before take-off.

![success](../experiments/plots/geo_success.png)
![wind](../experiments/plots/geo_wind.png)

## How we kept it honest

- The set has 30 orders in total (the 20 held-out ones give the headline numbers), plus 10 mid-flight changes and 15 attacks, all frozen in `benchmark/geo/missions.json` before any run.
- Half the orders use hard phrasing (paraphrase, negation, "the place where the incubator startups work").
- Dev/test split: prompts were written on dev only. Headline numbers are from the held-out test split.
- Baselines:
  - **regex** (no AI): what the LLM must beat.
  - **gold spec**: the ceiling, perfect understanding.
  - **llm_direct**: the model writes waypoints itself, flown with no shield.
- 26/26 planted bugs were caught by the tests (`tests/mutation_check.py`, last run at the end of Phase 2; it does not cover the console layer). The full suite has about 270 tests and is run with `scripts/check.ps1`.
- One run per mission at temperature 0. Models run one at a time so latency excludes model loading.

## Limits

- Simulation only: flat 2D world, simple wind and energy models, no GPS error or sensor faults, no building heights, no other aircraft.
- Small mission set (20 held-out orders), so the CIs are wide. Treat model differences under ~20 points as unproven.
- Place names come from OSM. Some buildings are missing or duplicated; duplicates trigger a clarifying question.
- **Gemini:** we used gemini-3.5-flash-lite because the free tier of gemini-3.5-flash allows 20 requests/day. The few 3.5-flash rows we got are set aside, not reported.
- Latency is from one laptop (RTX 3050 6 GB). qwen2.5:7b did not fit fully in GPU memory and ran partly on the CPU.
- The shield guarantees only what it checks. A legal but unwise target (a legal place the operator should not inspect) needs access control and human sign-off, not a shield.

## Industrial next steps

1. **Real autopilot:** feed the approved route to PX4/ArduPilot SITL (MAVLink mission upload), then hardware-in-the-loop. Keep the shield on the ground station *and* a geofence in the autopilot.
2. **Real airspace data:** live no-fly zones from DigitalSky (India) / NOTAMs instead of a hard-coded jail. Add altitude limits and building heights (3D).
3. **Certifiable core:** the shield and controller are small and deterministic. Write them up to DO-178C/ASTM F3269 run-time-assurance style: requirements → tests → coverage.
4. **Operator UX:** confirm-before-fly for low-confidence specs, voice input, multilingual orders (Hindi/Punjabi), and a read-back of the spec in plain words.
5. **Fleet:** several drones, shared airspace and deconfliction, pad scheduling.
6. **Ops:** the audit log goes to append-only storage, with post-flight reports from templates (not LLM text) and per-site configuration of pads and permanent zones.
7. **Model:** a small on-prem model (works offline, no data leaves the site). Fine-tune on logged orders with gold specs. Measure with this benchmark on every model change.

## Try it

```
.un.ps1            # creates the venv, installs the pinned packages, starts the console, opens the browser
```

The Phase 3 console (see the [README](../README.md)) has Start / Pause / Reset, a speed control, drawn areas, a chat column, "Land here", and downloads of the flight report, audit log and track. The three modes are **Ours** (AI → spec → shield), **Classic** (no AI) and **AI-direct** (no shield). The wind and battery settings in the Mission tab let you test the runtime monitor.
