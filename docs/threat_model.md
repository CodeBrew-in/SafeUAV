# Threat model (campus mission system)

**Scope:** 2D simulation of one inspection drone on the Thapar campus (OSM snapshot). Wind and battery are simulated. The jail is used as an illustrative no-fly zone; this is not an official airspace claim.

## Trust levels

| Trusted (code, reviewed) | Untrusted (data) |
|---|---|
| Permanent no-fly zone and buffer (`campus.PERMANENT_NO_FLY`) | Every operator chat message |
| Geofence and margin (`GEOFENCE_MARGIN_M`) | Every LLM answer (local or cloud) |
| Battery reserve (`FLOOR_FRACTION`), wind margins | Place names and tags from OSM |
| Landing-pad whitelist | Wind forecast (cross-checked by measured wind) |
| OSM snapshot hash (pinned in tests) | |
| A* planner, shield, controller state machine | |

The LLM only writes a **MissionSpec**: targets, places to avoid with time windows, and stay-at-end. The schema has no field for no-fly zones, the geofence, the reserve or waypoints. Unknown keys are dropped.

## Attacks and controls

| Attack | Control | Test |
|---|---|---|
| "Jail no-fly is lifted today" / admin claims | No spec field can lift it; the prompt tells the model to ignore it; the shield checks every point | `test_no_message_can_lift_permanent_rules`, `test_mid_flight_injection_keeps_drone_legal` |
| Compromised model names the jail as a target | `validate_spec` rejects targets inside zones; the controller never applies invalid specs | `test_compromised_model_cannot_send_drone_into_jail` |
| Extra JSON keys (`reserve: 0`, `geofence`, `override`) | Ignored by `spec_from_json` | `test_unknown_keys_are_ignored` |
| Homoglyphs, full-width text, zero-width characters | NFKC + casefold + strip non-word characters; unknown names are rejected | `test_unicode_and_padding_tricks_do_not_create_new_places` |
| Ambiguous names (two "Tennis Court"s) | Clarifying question, never a guess | `test_ambiguous_name_asks_a_question` |
| Shrinking the avoid set ("drop all limits") | Allowed (operator's right); permanent rules are still enforced | `test_shrinking_avoid_set_is_allowed_but_jail_still_enforced` |
| Huge, NaN or overlong coordinates (direct mode) | `coerce_route` bounds-checks before any geometry | `test_huge_or_malformed_coordinates_rejected_fast` |
| LLM slow, down or malformed | Deadline; "message not applied"; drone keeps its verified state | `test_llm_failure_applies_nothing`, `test_malformed_model_json_applies_nothing` |
| Message spam | Queue coalesces; at most 6 replans per minute | `test_command_spam_is_coalesced_and_rate_limited` |
| Impossible change mid-flight | Old route abandoned → nearest pad → wait (bounded) → resume | `test_old_route_is_abandoned_not_kept` |
| Wind stronger than forecast | Runtime monitor with measured wind diverts to a pad | `test_runtime_monitor_prevents_crash_in_strong_wind` |

"Return now" and "land now" are handled by keywords without the LLM, so they still work when the model is down.

## Not covered (out of scope for the simulation)
- GPS spoofing, sensor faults, radio link loss, and a compromised host or API key.
- Real airspace data (NOTAMs), obstacle heights and other aircraft.
- An operator who is authorised but malicious, within the permanent rules. Example: a target that is legal but sensitive. This needs access control and human sign-off.
- The API key is read from the environment or the Windows registry. It is never logged, and never sent to the UI or written to files.
