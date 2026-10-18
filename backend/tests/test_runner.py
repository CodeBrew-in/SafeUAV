"""End-to-end checks of the three methods on the demo scenarios.

The 'proposals' here are hand-written test fixtures, not model output.
"""

import pytest

from app.astar import plan_mission
from app.mission import build_mission
from app.runner import Method, MissionStatus as S, run_astar, run_shielded, run_unshielded
from app.scenarios import load_scenarios

SCENARIOS = load_scenarios()

# Scripted unsafe proposal: straight through the restricted block at x=5..7 and back.
THROUGH_ZONE = [[x, 1] for x in range(1, 11)] + [[10, 2], [10, 1]] + [[x, 1] for x in range(9, 0, -1)]


def ctx(sid):
    return build_mission(SCENARIOS[sid])


def kinds(result):
    return [e.kind for e in result.events]


def test_demo_scenarios_match_expected_feasibility():
    for s in SCENARIOS.values():
        assert plan_mission(build_mission(s).constraints).feasible == s.expected_feasible, s.id


def test_untrusted_note_does_not_reach_constraints():
    c = ctx("campus-injection")
    benign = ctx("campus-benign")
    assert c.untrusted_note
    assert c.constraints == benign.constraints


def test_astar_completes_benign():
    r = run_astar(ctx("campus-benign"))
    assert r.status is S.COMPLETED
    assert r.executed_trail[0] == r.executed_trail[-1] == (1, 1)
    assert r.executed_violations == 0
    assert len(r.executed_trail) - 1 == r.shortest_route_length == 44


def test_astar_reports_infeasible():
    r = run_astar(ctx("campus-low-battery"))
    assert r.status is S.INFEASIBLE
    assert r.executed_trail is None


def test_shielded_accepts_valid_proposal():
    c = ctx("campus-benign")
    proposal = [list(p) for p in plan_mission(c.constraints).route]
    r = run_shielded(c, proposal)
    assert r.status is S.COMPLETED
    assert not r.shield_intervened and not r.fallback_used


def test_shielded_rejects_unsafe_proposal_and_falls_back():
    r = run_shielded(ctx("campus-injection"), THROUGH_ZONE)
    assert r.shield_intervened and r.fallback_used
    assert r.status is S.COMPLETED_WITH_FALLBACK
    assert "restricted zone" in r.proposal_validation.reason
    restricted = ctx("campus-injection").constraints.restricted
    assert not any(cell in restricted for cell in r.executed_trail)
    assert kinds(r).index("validator_rejected") < kinds(r).index("fallback_requested") \
        < kinds(r).index("fallback_accepted") < kinds(r).index("execution_started")


def test_shielded_infeasible_does_not_claim_success():
    r = run_shielded(ctx("campus-low-battery"), THROUGH_ZONE)
    assert r.status is S.INFEASIBLE
    assert r.executed_trail is None
    assert "execution_started" not in kinds(r)


def test_shielded_without_fallback_rejects():
    r = run_shielded(ctx("campus-injection"), THROUGH_ZONE, allow_fallback=False)
    assert r.status is S.REJECTED
    assert r.executed_trail is None


@pytest.mark.parametrize("bad", [None, "not json", {"oops": 1}, [[1, 1], "x"]])
def test_shielded_handles_malformed_output(bad):
    r = run_shielded(ctx("campus-benign"), bad)
    assert r.status is S.COMPLETED_WITH_FALLBACK
    assert r.shield_intervened


def test_shielded_declared_infeasible_checked_by_astar():
    r = run_shielded(ctx("campus-benign"), None, declared_infeasible=True)
    assert r.status is S.COMPLETED_WITH_FALLBACK
    r = run_shielded(ctx("campus-low-battery"), None, declared_infeasible=True)
    assert r.status is S.INFEASIBLE


def test_unshielded_unsafe_proposal_is_hypothetical_only():
    r = run_unshielded(ctx("campus-injection"), THROUGH_ZONE)
    assert r.method is Method.LLM_ONLY
    assert r.status is S.UNSAFE
    assert r.trail_is_hypothetical
    assert r.executed_violations == 6  # (5,1),(6,1),(7,1) outbound and back
    assert "execution_started" not in kinds(r)


def test_unshielded_malformed_is_failed():
    r = run_unshielded(ctx("campus-benign"), "garbage")
    assert r.status is S.FAILED


def test_every_run_has_status_and_timed_events():
    c = ctx("campus-injection")
    for r in (run_astar(c), run_shielded(c, THROUGH_ZONE), run_unshielded(c, THROUGH_ZONE)):
        assert isinstance(r.status, S)
        assert r.events and r.events[0].kind == "mission_loaded"
        times = [e.elapsed_ms for e in r.events]
        assert times == sorted(times)
        assert r.planning_latency_ms is not None


def test_runs_do_not_share_state():
    c = ctx("campus-benign")
    a, b = run_astar(c), run_astar(c)
    assert a.run_id != b.run_id
    a.executed_trail.clear()
    assert b.executed_trail
