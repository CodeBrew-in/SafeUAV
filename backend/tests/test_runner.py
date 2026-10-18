"""End-to-end checks of every method on the demo scenarios.

The proposals here are hand-written fixtures (model="scripted"), not model output.
"""

import pytest

from app.astar import plan_mission
from app.llm_planner import scripted_proposal
from app.mission import build_mission
from app.runner import (FLOWN_STATUSES, Method, RejectionCause as RC, Status as S, run_astar, run_shielded,
                        run_unshielded)
from app.scenarios import load_scenarios

SCENARIOS = load_scenarios()

# Straight through the restricted block at x=5..7 and back.
THROUGH_ZONE = [[x, 1] for x in range(1, 11)] + [[10, 2], [10, 1]] + [[x, 1] for x in range(9, 0, -1)]


def ctx(sid):
    return build_mission(SCENARIOS[sid])


def kinds(result):
    return [e.kind for e in result.events]


def safe_route(sid):
    return [list(p) for p in plan_mission(ctx(sid).constraints).route]


def test_demo_scenarios_match_expected_feasibility():
    for s in SCENARIOS.values():
        assert plan_mission(build_mission(s).constraints).feasible == s.expected_feasible, s.id


def test_mislabelled_scenario_refuses_to_load(tmp_path):
    import json
    from app.scenarios import DEMO_SCENARIOS, load_scenarios as load
    records = json.loads(DEMO_SCENARIOS.read_text())
    records[0]["expected_feasible"] = False  # campus-benign is feasible
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="expected_feasible"):
        load(bad)


def test_untrusted_note_does_not_reach_constraints():
    assert ctx("campus-injection").untrusted_note
    assert ctx("campus-injection").constraints == ctx("campus-benign").constraints


# --- astar ------------------------------------------------------------------------------

def test_astar_completes_benign():
    r = run_astar(ctx("campus-benign"))
    assert r.status is S.COMPLETED
    assert r.flown_route[0] == r.flown_route[-1] == (1, 1)
    assert r.executed_violations == 0  # measured on the trail, not a default
    assert len(r.flown_route) - 1 == r.optimal_steps == 44
    assert r.ground_truth_feasible


def test_astar_reports_infeasible_proven():
    r = run_astar(ctx("campus-low-battery"))
    assert r.status is S.INFEASIBLE_PROVEN
    assert r.flown_route is None and r.executed_violations is None
    assert r.optimal_steps == 44 and not r.ground_truth_feasible


# --- llm + shield -------------------------------------------------------------------------

def test_shielded_accepts_valid_proposal():
    r = run_shielded(ctx("campus-benign"), safe_route("campus-benign"))
    assert r.status is S.COMPLETED
    assert not r.shield_intervened and not r.fallback_used and r.rejection_cause is None
    assert r.executed_violations == 0


def test_shielded_rejects_unsafe_proposal_and_falls_back():
    r = run_shielded(ctx("campus-injection"), THROUGH_ZONE)
    assert r.status is S.COMPLETED_WITH_FALLBACK
    assert r.rejection_cause is RC.SHIELD_REJECTED_UNSAFE
    assert r.shield_intervened and r.fallback_used and r.proposal_unsafe
    assert "restricted zone" in r.proposal_validation.reason
    restricted = ctx("campus-injection").constraints.restricted
    assert not any(cell in restricted for cell in r.flown_route)
    assert r.executed_violations == 0
    k = kinds(r)
    assert k.index("shield_rejected") < k.index("fallback_requested") < k.index("fallback_accepted") \
        < k.index("execution_started")


def test_shielded_mission_invalid_is_its_own_cause():
    route = safe_route("campus-benign")[:10]  # safe cells, never reaches the destination
    r = run_shielded(ctx("campus-benign"), route)
    assert r.rejection_cause is RC.SHIELD_REJECTED_MISSION_INVALID
    assert r.proposal_unsafe is False and r.proposal_mission_invalid


def test_shielded_infeasible_does_not_claim_success():
    r = run_shielded(ctx("campus-low-battery"), THROUGH_ZONE)
    assert r.status is S.INFEASIBLE_PROVEN
    assert r.flown_route is None
    assert "execution_started" not in kinds(r)


def test_no_fallback_arm_flies_nothing_on_rejection():
    r = run_shielded(ctx("campus-injection"), THROUGH_ZONE, allow_fallback=False)
    assert r.method is Method.LLM_SHIELD_NO_FALLBACK
    assert r.status is S.NOT_FLOWN and r.flown_route is None
    assert r.rejection_cause is RC.SHIELD_REJECTED_UNSAFE


def test_no_fallback_arm_completes_with_llm_route():
    r = run_shielded(ctx("campus-benign"), safe_route("campus-benign"), allow_fallback=False)
    assert r.status is S.COMPLETED


@pytest.mark.parametrize("bad", ["not json", {"oops": 1}, [[1, 1], "x"], None])
def test_unusable_output_is_parse_failure_not_shield_intervention(bad):
    r = run_shielded(ctx("campus-benign"), bad)
    assert r.status is S.COMPLETED_WITH_FALLBACK
    assert r.rejection_cause is RC.PARSE_FAILURE
    assert not r.shield_intervened
    assert "shield_rejected" not in kinds(r)


def test_transport_failure_is_its_own_cause():
    p = scripted_proposal(transport_error="URLError: connection refused")
    r = run_shielded(ctx("campus-benign"), p)
    assert r.rejection_cause is RC.TRANSPORT_FAILURE and not r.shield_intervened
    assert r.status is S.COMPLETED_WITH_FALLBACK
    assert run_unshielded(ctx("campus-benign"), p).status is S.NO_PROPOSAL


def test_false_infeasible_is_flagged():
    p = scripted_proposal(status="infeasible")
    r = run_shielded(ctx("campus-benign"), p)
    assert r.false_infeasible and r.rejection_cause is RC.DECLARED_INFEASIBLE
    assert r.status is S.COMPLETED_WITH_FALLBACK and not r.shield_intervened
    u = run_unshielded(ctx("campus-benign"), p)
    assert u.status is S.DECLARED_INFEASIBLE and u.false_infeasible


def test_correct_infeasible_is_not_flagged():
    p = scripted_proposal(status="infeasible")
    r = run_shielded(ctx("campus-low-battery"), p)
    assert r.status is S.INFEASIBLE_PROVEN and not r.false_infeasible


# --- llm-only (hypothetical) ------------------------------------------------------------------

def test_unshielded_unsafe_proposal_is_hypothetical_only():
    r = run_unshielded(ctx("campus-injection"), THROUGH_ZONE)
    assert r.method is Method.LLM_ONLY
    assert r.status is S.PROPOSAL_UNSAFE and r.proposal_unsafe
    assert r.hypothetical_route and r.flown_route is None and r.executed_violations is None
    assert "execution_started" not in kinds(r)


def test_unshielded_statuses_never_overlap_flown_statuses():
    c = ctx("campus-benign")
    for proposal in (THROUGH_ZONE, safe_route("campus-benign"), safe_route("campus-benign")[:5], "x",
                     scripted_proposal(status="infeasible")):
        assert run_unshielded(c, proposal).status not in FLOWN_STATUSES


def test_jump_over_restricted_zone_is_unsafe():
    # (4,1) -> (8,1) skips over restricted cells (5..7, 1) without landing on them
    route = [[1, 1], [2, 1], [3, 1], [4, 1], [8, 1], [9, 1], [10, 1], [10, 2], [10, 1], [9, 1], [8, 1],
             [4, 1], [3, 1], [2, 1], [1, 1]]
    r = run_unshielded(ctx("campus-benign"), route)
    assert r.status is S.PROPOSAL_UNSAFE
    assert (5, 1) in r.proposal_validation.swept_forbidden


def test_over_budget_is_unsafe():
    c = ctx("open-field")  # budget 20, optimal 18
    route = safe_route("open-field")
    detour = route[:1] + [[1, 0], [0, 0]] + route[1:]  # 20 steps: still within budget
    assert run_unshielded(c, detour).status is S.PROPOSAL_VALID
    over = route[:1] + [[1, 0], [0, 0], [1, 0], [0, 0]] + route[1:]  # 22 steps
    assert run_unshielded(c, over).status is S.PROPOSAL_UNSAFE


def test_very_long_route_is_unsafe_not_malformed():
    route = [[1, 1], [1, 2]] * 150 + [[1, 1]]
    r = run_unshielded(ctx("campus-benign"), route)
    assert r.status is S.PROPOSAL_UNSAFE


# --- shared properties -----------------------------------------------------------------------

def test_every_run_has_status_and_ordered_events():
    c = ctx("campus-injection")
    for r in (run_astar(c), run_shielded(c, THROUGH_ZONE), run_unshielded(c, THROUGH_ZONE),
              run_shielded(c, THROUGH_ZONE, allow_fallback=False)):
        assert isinstance(r.status, S)
        assert r.events[0].kind == "mission_loaded" and r.events[-1].kind == "mission_finished"
        times = [e.elapsed_ms for e in r.events]
        assert times == sorted(times)
        assert r.decision_latency_ms is not None and r.total_latency_ms is not None
        assert r.ground_truth_feasible is not None


def test_runs_do_not_share_state():
    c = ctx("campus-benign")
    a, b = run_astar(c), run_astar(c)
    assert a.run_id != b.run_id
    a.flown_route.clear()
    assert b.flown_route
