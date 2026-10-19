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

## Console layer (Phase 3)

The console is a local web server (`backend/app/geo/server.py`) bound to 127.0.0.1 only. It has no
accounts. Each browser tab gets its own drone (`X-Drone-Id`, kept in `sessionStorage`).

| Attack or failure | Control | Test |
|---|---|---|
| DNS rebinding / foreign Host header | Host must be `127.0.0.1:PORT` or `localhost:PORT` | `test_bad_host_is_refused_even_for_the_page` |
| Cross-site request (CSRF) | Origin must be the same origin; POST must be `application/json`; every `/api/*` call needs the per-process token | `test_foreign_origin_is_refused`, `test_post_needs_json_and_sane_length`, `test_missing_or_wrong_token_is_refused` |
| Clickjacking (a hostile page frames the console) | `frame-ancestors 'none'` and `X-Frame-Options: DENY` | `test_review_hardening_headers_nan_deep_json_and_strict_values` |
| XSS through OSM names, chat, AI text, audit lines | All text is inserted with `textContent` / canvas text, never `innerHTML`. CSP `default-src 'self'`. The report escapes every value | `test_the_downloads_are_valid_and_inert` |
| Path traversal under `/static/` | File-name whitelist | `test_bad_inputs_get_400_not_500` |
| Oversized, malformed, NaN / Infinity, deeply nested JSON | 20 KB cap, strict Content-Length, `parse_constant` rejects NaN, `RecursionError` is a 400, state is serialised with `allow_nan=False` | `test_review_hardening_headers_nan_deep_json_and_strict_values` |
| Strings or floats where booleans or integers are expected | Strict type checks in `configure`, `edit`, `set_speed` | `test_non_finite_and_non_boolean_settings_are_refused` |
| Another process binds the same port (Windows `SO_REUSEADDR`) | `SO_EXCLUSIVEADDRUSE`, reuse disabled | (manual check) |
| Connection flood, slow clients | Per-request slots (32), connection cap (128), 10 s socket timeout, bounded listen queue | (stress-tested by hand in review) |
| Memory growth | Chat capped at 400 lines; the browser caps its trail, audit and chat copies | `test_session` (chat offsets) |
| Operator's Land / Return / End delayed by the AI or the replan limit | Safety commands are never rate-limited, act while the AI is thinking, and a late AI reply is dropped | `test_land_acts_while_the_ai_is_still_thinking_and_a_late_reply_cannot_relaunch`, `test_the_replan_rate_limit_never_delays_land` |
| A longer order containing "land" or "return" is taken as a command | Only whole-message commands count ("land now", "return home") | `test_longer_orders_containing_safety_words_are_not_hijacked` |
| Landing at a pad that is unsafe or too far | "Land at a pad" goes through the same checks as every route and is refused with a reason | `test_land_at_a_pad_that_fails_the_shield_is_refused_and_the_drone_flies_on` |
| Drawn areas used to bypass rules | A drawn area is only a place: restricting or targeting it goes through `validate_spec`; radius 20-300 m, centre inside the flight area, at most 8 | `test_drawn_areas_over_http`, `test_a_restricted_drawn_area_is_flown_around_and_the_cap_holds` |
| Download files carry spreadsheet formulas or HTML | Cells starting with `= + - @` are neutralised; the HTML report escapes everything and is served as an attachment | `test_csv_cells_cannot_become_formulas` |
| Settings file used to lower the safety rules | `safeuav.toml` only accepts a fixed list of keys and types; the safety constants are not settings | `test_settings_file_is_validated_and_cannot_touch_safety_constants` |
| API key leaks | Read only from the environment (or the Windows user environment); never written to files, logs, state or error text; `/api/health` reports only whether a key exists | `test_state_never_contains_the_token_or_a_key` |

**Accepted limits of the console layer**

- The token is in the page HTML, so it only defends against other web pages, not against other programs
  on the same computer or other users of it. Run it on a computer you trust.
- The OSM snapshot's hash is pinned in tests but not checked when the server starts.
- A user can restrict every pad (or the whole area) and ground their own drone. This is allowed by design;
  the drone then fails to start or ends up safely grounded.
- The AI's free text (reasons, questions) is shown to the operator. A manipulated model could write
  misleading advice there; it cannot change the flight rules.
- The server is a single-process teaching server, not a production service.
