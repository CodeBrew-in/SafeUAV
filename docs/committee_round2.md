# Committee round 2 — judge report (summary)

Verdict: **PASS, 8/10** (round 1: REVISE, 5/10). Judged on commit 2701dd0 plus the round-2 packet.

## Round-1 must-fix items

| # | Item | Round-2 status |
|---|---|---|
| 1 | Paired proposals, no-fallback arm | DONE |
| 2 | Separate rejection causes; shield_intervened only for shield rejections | DONE (cosmetic leftover: null waypoints → fixed in 4096f1c) |
| 3 | Binary unsafe metric (budget, wrong start, swept cells, long routes) | DONE |
| 4 | Separate status vocabularies, split infeasible/rejected | DONE |
| 5 | Ground truth, optimal_steps, false_infeasible, loader check | DONE |
| 6 | executed_violations measured on the flown trail | DONE |
| 7 | Comparable latency, per-attempt logging | PARTIAL → fixed in 4096f1c (A* timed only when the fallback runs; tested) |
| 8 | Planner robustness, token cap | DONE |
| 9 | Corrected claims | DONE |
| 10 | Reproducible JSONL records, mutation output saved | PARTIAL → fixed: records regenerated on clean commit 4096f1c, mutation output re-saved, both committed |

## Round-2 must-fix (all applied in 4096f1c and the follow-up commit)

1. Make `decision_latency_ms` consistent across arms. Done: ground truth is outside the timed region; A* is
   timed only on fallback; `test_decision_latency_counts_astar_only_when_falling_back`.
2. Regenerate demo records on a clean commit and commit them with the mutation output. Done:
   `results/demo_runs_4096f1c.jsonl`, `docs/mutation_check_output.txt`.
3. `summarize_runs.py` prints proposal length, violation codes and the producing attempt's temperature.
   Done.
4. Benchmark analysis rules (first-attempt vs all proposals, feasible-only completion denominators). Done:
   "Analysis rules" section in `docs/review_packet.md`.

## Judge's remaining notes for the benchmark phase

- Report E1 by producing attempt/temperature; never pool retried proposals silently.
- No-fallback completion must use feasible scenarios as the denominator.
