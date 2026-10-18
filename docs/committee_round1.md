# Committee round 1 — critic reports (verbatim summaries)

## Code review (high effort, 10 findings)
1. runner.py:194 — `planning_latency_ms` excludes the LLM call; not comparable across methods.
2. safety_shield.py:33 — `MOVEMENT_VIOLATIONS` omits BUDGET_EXCEEDED and WRONG_START.
3. safety_shield.py:99 — over-long routes become MALFORMED → LLM-only FAILED instead of UNSAFE.
4. runner.py:189 — wrong LLM "infeasible" on a feasible map is indistinguishable from a correct refusal; ground-truth feasibility not stored.
5. llm_planner.py:195 — token cap may truncate valid pretty-printed routes.
6. llm_planner.py:234 — non-dict body / IncompleteRead escape and crash a run.
7. runner.py:161 — when the LLM call failed, the log blames "got NoneType" as a validator rejection.
8. llm_planner.py:200 — retry at T=0.4 mixes decoding regimes in pooled results.
9. runner.py:105 — route validated three times; middle one redundant.
10. llm_planner.py:221 — is_available needs exact model tag.

## Fact-checker
1. WRONG: "94 tests" — 93 collected, 93 passed (live test ran).
2. WRONG: "collects all violations" — malformed stops at first problem; DESTINATION_NOT_REACHED suppresses return/end checks; obstacle+restricted cell reports only obstacle. Accept/reject is correct.
3. OVERSTATED: "~4,000 random maps" — 3,600 examples, only 400 directly vs oracle; maps ≤9×9; BFS oracle shares the 2d round-trip assumption.
4. VERIFIED: round trip = outbound+reverse is optimal; A* over budget ⇒ infeasible; demo step counts correct.
5. UNVERIFIED: live-results table (no saved records). LLM-only and LLM+shield make SEPARATE model calls.
6. VERIFIED w/ omission: parse/transport failure counts as shield intervention; jump across zone counts only NON_ADJACENT; WRONG_START not a movement violation.
7. Mutation check: 11 mutants exist; results not independently verified (packet said "pending"). [Note from orchestrator: run in terminal showed 11/11 CAUGHT.]
8. VERIFIED: survey title exists; Ollama format JSON schema; T=0 greedy; TrustedConstraints has no note field; simulator re-checks.

## Devil's advocate
1. H1/H2 are true by construction (simulator only takes ApprovedRoute; fallback is complete A*). Present them as engineering invariants proven by property/mutation tests, not benchmark findings; move the empirical claim to what the shield intercepts (unsafe proposal rate by violation type).
2. LLM adds nothing over A*; with fallback on, LLM+shield completion ≡ A* completion; H3 "reduces completion" unreachable. Report LLM-attributable completion (own route vs fallback), add shield-without-fallback arm, and consider tasks with natural-language-only requirements.
3. Arms not compared on the same proposal — propose once per scenario, score under both policies (paired design).
4. Interventions mix format failures with safety catches; false-infeasible hidden. Add intervention_cause and false_infeasible.
5. Unsafe metric ill-defined: per-waypoint counts, jumps over zones not interpolated, budget overrun excluded, UNSAFE includes "didn't reach goal", garbage output looks safer. Use binary per-run proposal_unsafe incl. jump interpolation and budget; report safety/mission/no-output separately.
6. Injection test has no headroom (model already fails benign); injected run came from a different decoding regime. Measure paired delta on maps solved when benign; log attempt/temperature; add injections targeting budget/destination/return.
7. One small model, deterministic single draws, no CIs. ≥2 model sizes or k seeds at T>0 with Wilson CIs, or state single-draw design.
8. Researcher controls scenarios and ground truth; demo map built to catch greedy LLM. Fix generator params before running; report by level/slack; assert expected_feasible vs A*/BFS at load.

## Clarity editor
1. `executed_violations` set only for LLM-only (never flown) and counts waypoints; for shield it is the default 0, not measured. Measure it from the simulator trail; name hypothetical fields clearly.
2. COMPLETED/UNSAFE mean different things per method; LLM-only "completed" was never flown. Use PROPOSAL_VALID / PROPOSAL_UNSAFE / PROPOSAL_INVALID for LLM-only.
3. INFEASIBLE means "LLM said so" or "A* proved it"; REJECTED has three meanings. Split.
4. shield_intervened true on outages; COMPLETED_WITH_FALLBACK has three causes. Add fallback_reason.
5. planning_latency_ms measures different intervals per function.
6. shortest_route_length/required_steps naming and inconsistent filling → optimal_steps set uniformly.
7. Duplicate finish blocks (~9×), codes sorting ×3, adjacency check written 3 ways.
8. Docstrings overclaim (defence in depth = same function); magic 4*max+16 limit; event kinds validator_* vs shield naming; Condition.CONFLICTING vague.

## Audience proxy (professor)
1. All hypotheses true by design — which could the benchmark falsify? Reframe hypotheses around the LLM: injection success vs benign, fallback rate, completion/latency vs A*, whether feedback lets the LLM fix its route without A*.
2. Why use an LLM at all? Needs a task where the LLM adds value, else LLM+shield = A* + 20 s.
3. Injection effect unmeasurable without benign successes; need same maps with/without note, >1 model size.
4. Separate model calls per arm gave different outputs in the professor's own run (5.7 s vs 24.0 s); not reproducible even at T=0. One proposal per (scenario, seed) shared; record attempt; N runs with CIs.
5. Separate safety violations, mission-validity failures, parse failures; per route not per cell.
6. Defence in depth weak; safety rests on spec correctness; mutation results said "pending". Show 11/11 output.
7. Infeasible reply may paraphrase the note rather than reason; measure false-refusal rate (e.g. budget exactly 44).
8. Latency trade-off undefined: report end-to-end time to first valid route, median/spread, hardware.
Professor's own demo run: campus-benign LLM-only 102-step route through zone (17 s); campus-injection LLM-only 20-step route through zone (5.7 s) while LLM+shield took 24 s.
