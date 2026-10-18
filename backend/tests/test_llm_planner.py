"""LLM planner tests. A fake transport stands in for Ollama; no model is called.

One live test at the bottom runs only when a local Ollama server with the model is up.
"""

import http.client
import json
import urllib.error

import pytest

from app.llm_planner import OllamaPlanner, build_user_prompt, parse_response, render_map
from app.mission import build_mission
from app.runner import LLM_METHODS, Method, RejectionCause as RC, Status as S, run_llm_arms
from app.scenarios import load_scenarios

SCENARIOS = load_scenarios()
THROUGH_ZONE = [[x, 1] for x in range(1, 11)] + [[10, 2], [10, 1]] + [[x, 1] for x in range(9, 0, -1)]


def ctx(sid):
    return build_mission(SCENARIOS[sid])


class FakeTransport:
    """Replays canned model replies and records every request payload."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, url, payload, timeout):
        self.payloads.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"role": "assistant", "content": reply}}


def planner(*replies, **kw):
    t = FakeTransport(*replies)
    return OllamaPlanner(transport=t, **kw), t


def reply(status="route_proposed", waypoints=(), reason="ok"):
    return json.dumps({"status": status, "waypoints": list(waypoints), "reason": reason})


# --- prompt construction -------------------------------------------------------

def test_prompt_lists_trusted_constraints():
    p = build_user_prompt(ctx("campus-benign"))
    assert "Base: [1, 1]" in p and "Destination: [10, 2]" in p
    assert "[6, 3]" in p  # a restricted cell
    assert "Movement budget: 50 steps" in p
    assert "Return to base required: yes" in p


def test_untrusted_note_only_in_prompt_for_injection_scenario():
    note = SCENARIOS["campus-injection"].untrusted_note
    assert note in build_user_prompt(ctx("campus-injection"))
    assert note not in build_user_prompt(ctx("campus-benign"))


def test_render_map_marks_cells():
    m = render_map(ctx("campus-benign").constraints).splitlines()
    row1 = m[2].split()  # header, y=0, y=1
    assert row1[0] == "1" and row1[2] == "B" and row1[6] == "X"
    assert m[3].split()[11] == "D"  # (10, 2)


def test_payload_uses_schema_and_bounded_output():
    p, t = planner(reply(waypoints=[[1, 1]]))
    p.propose(ctx("campus-benign"))
    payload = t.payloads[0]
    assert payload["format"]["required"] == ["status", "waypoints", "reason"]
    assert payload["options"]["num_predict"] > 0
    assert payload["stream"] is False


# --- parsing --------------------------------------------------------------------

@pytest.mark.parametrize("text,error", [
    ("not json", "invalid JSON"),
    ("[1, 2]", "expected a JSON object"),
    ('{"status": "launch_missiles", "waypoints": []}', "unknown status"),
    ('{"status": "route_proposed", "reason": "x"}', "missing 'waypoints'"),
])
def test_parse_rejects_bad_output(text, error):
    r = parse_response(text)
    assert not r.ok and error in r.error


def test_parse_accepts_code_fence():
    r = parse_response("```json\n" + reply(waypoints=[[1, 1]]) + "\n```")
    assert r.ok and r.waypoints == [[1, 1]]


def test_parse_leaves_waypoint_types_to_the_shield():
    r = parse_response('{"status": "route_proposed", "waypoints": "north", "reason": ""}')
    assert r.ok and r.waypoints == "north"


def test_parse_infeasible():
    r = parse_response(reply("infeasible", reason="budget too small"))
    assert r.ok and r.status == "infeasible" and r.waypoints is None


# --- retries and errors ---------------------------------------------------------

def test_retries_once_after_bad_json_then_succeeds():
    p, t = planner("oops", reply(waypoints=[[1, 1]]), max_attempts=2)
    prop = p.propose(ctx("campus-benign"))
    assert prop.attempts == 2 and prop.parse_error is None
    assert prop.raw_responses == ["oops", reply(waypoints=[[1, 1]])]


def test_retries_are_bounded():
    p, t = planner("bad", "bad", "bad", max_attempts=2)
    prop = p.propose(ctx("campus-benign"))
    assert prop.attempts == 2 and prop.parse_error
    assert len(t.replies) == 1  # third reply never requested


def test_transport_error_not_retried():
    p, t = planner(urllib.error.URLError("connection refused"), reply(), max_attempts=3)
    prop = p.propose(ctx("campus-benign"))
    assert prop.attempts == 1 and "connection refused" in prop.transport_error


def test_timeout_is_reported():
    p, _ = planner(TimeoutError("timed out"))
    assert "timed out" in p.propose(ctx("campus-benign")).transport_error


def test_token_limit_scales_with_budget_and_fits_pretty_json():
    p, _ = planner()
    small, big = ctx("open-field"), ctx("campus-benign")  # budgets 20 and 50
    assert p.token_limit(small) < p.token_limit(big) <= p.max_tokens
    # 15 tokens per waypoint covers one-value-per-line JSON for a full-budget route
    assert p.token_limit(big) >= 51 * 15


@pytest.mark.parametrize("failure", [
    http.client.IncompleteRead(b"partial"),
    ConnectionResetError("dropped"),
])
def test_dropped_connection_is_transport_error(failure):
    p, _ = planner(failure)
    prop = p.propose(ctx("campus-benign"))
    assert prop.transport_error and prop.status is None


def test_non_object_body_is_transport_error():
    class ListTransport(FakeTransport):
        def __call__(self, url, payload, timeout):
            return ["not", "a", "dict"]
    prop = OllamaPlanner(transport=ListTransport()).propose(ctx("campus-benign"))
    assert "expected a JSON object" in prop.transport_error


def test_server_error_body_is_transport_error():
    class ErrTransport(FakeTransport):
        def __call__(self, url, payload, timeout):
            return {"error": "model 'x' not found"}
    prop = OllamaPlanner(transport=ErrTransport()).propose(ctx("campus-benign"))
    assert "not found" in prop.transport_error


def test_attempt_log_records_decoding_settings():
    p, _ = planner("bad", reply(waypoints=[[1, 1]]), max_attempts=2, seed=3)
    prop = p.propose(ctx("campus-benign"))
    log = prop.summary()["attempt_log"]
    assert [a["temperature"] for a in log] == [0.0, 0.4]
    assert [a["seed"] for a in log] == [3, 4]
    assert [a["outcome"] for a in log] == ["parse_error", "parsed"]


def test_seed_override_per_call():
    p, t = planner(reply(waypoints=[[1, 1]]), seed=0)
    p.propose(ctx("campus-benign"), seed=7)
    assert t.payloads[0]["options"]["seed"] == 7


def test_retry_uses_a_different_seed():
    p, t = planner("bad", "bad", max_attempts=2, seed=7)
    p.propose(ctx("campus-benign"))
    assert [pl["options"]["seed"] for pl in t.payloads] == [7, 8]
    assert [pl["options"]["temperature"] for pl in t.payloads] == [0.0, 0.4]


def test_cut_off_output_is_labelled():
    class Truncating(FakeTransport):
        def __call__(self, url, payload, timeout):
            return {**super().__call__(url, payload, timeout), "done_reason": "length"}

    t = Truncating('{"status": "route_proposed", "waypoints": [[1, 1], [1,')
    prop = OllamaPlanner(transport=t, max_attempts=1).propose(ctx("campus-benign"))
    assert "cut off at the token limit" in prop.parse_error


# --- end to end through the runner ------------------------------------------------

def test_one_model_call_scored_under_every_arm():
    p, t = planner(reply(waypoints=THROUGH_ZONE, reason="shortest route, zone lifted"))
    rs = run_llm_arms(ctx("campus-injection"), p, seed=0)
    assert len(t.payloads) == 1  # paired design: exactly one model call
    assert set(rs) == set(LLM_METHODS)
    raws = {tuple(r.llm["raw_responses"]) for r in rs.values()}
    assert len(raws) == 1
    assert rs[Method.LLM_ONLY].status is S.PROPOSAL_UNSAFE
    assert rs[Method.LLM_SHIELD].status is S.COMPLETED_WITH_FALLBACK
    assert rs[Method.LLM_SHIELD_NO_FALLBACK].status is S.NOT_FLOWN
    assert all(r.seed == 0 for r in rs.values())
    k = [e.kind for e in rs[Method.LLM_SHIELD].events]
    assert k[:3] == ["mission_loaded", "route_request_submitted", "llm_response_parsed"]
    assert "shield_rejected" in k


def test_model_claim_of_safety_is_ignored():
    p, _ = planner(reply(waypoints=THROUGH_ZONE, reason="This route is 100% safe and verified."))
    assert run_llm_arms(ctx("campus-benign"), p)[Method.LLM_SHIELD].shield_intervened


def test_server_down():
    p, _ = planner(urllib.error.URLError("refused"))
    rs = run_llm_arms(ctx("campus-benign"), p)
    assert rs[Method.LLM_SHIELD].status is S.COMPLETED_WITH_FALLBACK
    assert rs[Method.LLM_SHIELD].rejection_cause is RC.TRANSPORT_FAILURE
    assert not rs[Method.LLM_SHIELD].shield_intervened
    assert rs[Method.LLM_ONLY].status is S.NO_PROPOSAL
    assert "llm_error" in [e.kind for e in rs[Method.LLM_ONLY].events]


def test_model_says_infeasible():
    p, _ = planner(reply("infeasible", reason="too hard"))
    rs = run_llm_arms(ctx("campus-benign"), p)
    assert rs[Method.LLM_SHIELD].status is S.COMPLETED_WITH_FALLBACK and rs[Method.LLM_SHIELD].false_infeasible
    p, _ = planner(reply("infeasible", reason="too hard"))
    rs = run_llm_arms(ctx("campus-low-battery"), p)
    assert rs[Method.LLM_SHIELD].status is S.INFEASIBLE_PROVEN and not rs[Method.LLM_SHIELD].false_infeasible


def test_latency_includes_the_model_call():
    p, _ = planner(reply(waypoints=THROUGH_ZONE))
    r = run_llm_arms(ctx("campus-injection"), p)[Method.LLM_SHIELD]
    assert r.total_latency_ms == pytest.approx(r.proposal_latency_ms + r.decision_latency_ms)
    times = [e.elapsed_ms for e in r.events]
    assert times == sorted(times)


# --- live model (optional) ----------------------------------------------------------

live = OllamaPlanner()


@pytest.mark.skipif(not live.is_available(), reason="local Ollama with qwen2.5:3b not running")
def test_live_model_returns_parseable_proposal():
    prop = live.propose(ctx("open-field"))
    assert prop.transport_error is None
    assert prop.parse_error is None, prop.raw_responses
