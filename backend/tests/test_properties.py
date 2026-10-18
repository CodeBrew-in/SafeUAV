"""Property-based stress tests for the core safety logic.

Each property compares the production code against a separate, deliberately
simple oracle written here (plain BFS and a direct rule check), on thousands of
randomly generated maps and routes. A bug in the shield or A* would have to be
duplicated in the oracle to go unnoticed.
"""

from collections import deque

from hypothesis import HealthCheck, given, settings, strategies as st

from app.astar import PlanOutcome, plan_mission
from app.mission import TrustedConstraints, build_mission
from app.runner import RejectionCause as RC, Status as S, run_astar, run_shielded, run_unshielded
from app.safety_shield import approve, validate_route
from app.simulator import Simulator

SETTINGS = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --- oracles --------------------------------------------------------------------

def bfs_distance(c: TrustedConstraints, a, b):
    def ok(p):
        return 0 <= p[0] < c.width and 0 <= p[1] < c.height and p not in c.obstacles and p not in c.restricted
    if not ok(a) or not ok(b):
        return None
    dist = {a: 0}
    q = deque([a])
    while q:
        cur = q.popleft()
        if cur == b:
            return dist[cur]
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + d[0], cur[1] + d[1])
            if ok(n) and n not in dist:
                dist[n] = dist[cur] + 1
                q.append(n)
    return None


def oracle_valid(route, c: TrustedConstraints) -> bool:
    """Direct restatement of the mission rules, independent of safety_shield.py."""
    if not route or route[0] != c.start or len(route) - 1 > c.max_steps:
        return False
    for x, y in route:
        if not (0 <= x < c.width and 0 <= y < c.height):
            return False
        if (x, y) in c.obstacles or (x, y) in c.restricted:
            return False
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
    if c.destination not in route:
        return False
    return route[-1] == (c.start if c.return_to_base else c.destination)


def forbidden(cell, c: TrustedConstraints) -> bool:
    return not c.in_bounds(cell) or cell in c.obstacles or cell in c.restricted


def oracle_unsafe(route, c: TrustedConstraints) -> bool:
    """Would flying this route break a safety rule? Independent restatement of the metric."""
    if not route:
        return False
    if route[0] != c.start or len(route) - 1 > c.max_steps:
        return True
    if any(forbidden(p, c) for p in route):
        return True
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return True  # a teleport is unsafe in itself, whatever it jumps over
    return False


# --- generators -------------------------------------------------------------------

@st.composite
def constraints(draw, max_dim=12):
    w = draw(st.integers(2, max_dim))
    h = draw(st.integers(2, max_dim))
    cells = [(x, y) for x in range(w) for y in range(h)]
    start = draw(st.sampled_from(cells))
    dest = draw(st.sampled_from([p for p in cells if p != start]))
    others = [p for p in cells if p != start]
    obstacles = draw(st.sets(st.sampled_from(others), max_size=len(others) // 3))
    restricted = draw(st.sets(st.sampled_from(others), max_size=len(others) // 3))
    budget = draw(st.integers(1, 4 * (w + h)))
    rtb = draw(st.booleans())
    return TrustedConstraints(w, h, start, dest, frozenset(obstacles), frozenset(restricted), budget, rtb)


@st.composite
def constraints_and_route(draw):
    """A map plus a route that is usually close to valid, then randomly corrupted."""
    c = draw(constraints())
    plan = plan_mission(c)
    if plan.feasible and draw(st.booleans()):
        route = list(plan.route)
    else:
        # random walk from start, sometimes off the map or through blocked cells
        route = [c.start]
        for _ in range(draw(st.integers(0, 3 * (c.width + c.height)))):
            dx, dy = draw(st.sampled_from([(1, 0), (-1, 0), (0, 1), (0, -1)]))
            route.append((route[-1][0] + dx, route[-1][1] + dy))
    mutation = draw(st.sampled_from(["none", "none", "jump", "stay", "insert", "drop", "shift_start", "extend"]))
    if route and mutation == "jump":
        i = draw(st.integers(0, len(route) - 1))
        route[i] = (draw(st.integers(-2, c.width + 1)), draw(st.integers(-2, c.height + 1)))
    elif route and mutation == "stay":
        i = draw(st.integers(0, len(route) - 1))
        route.insert(i, route[i])
    elif mutation == "insert":
        route.insert(draw(st.integers(0, len(route))), (draw(st.integers(-1, c.width)), draw(st.integers(-1, c.height))))
    elif route and mutation == "drop":
        route.pop(draw(st.integers(0, len(route) - 1)))
    elif route and mutation == "shift_start":
        route[0] = (route[0][0] + 1, route[0][1])
    elif route and mutation == "extend":
        route += route[-2::-1] + route[1:]  # extra laps, may blow the budget
    return c, route


# --- properties ---------------------------------------------------------------------

@SETTINGS
@given(constraints_and_route())
def test_shield_agrees_with_independent_oracle(cr):
    c, route = cr
    result = validate_route([list(p) for p in route], c)
    assert result.valid == oracle_valid(route, c), result.reason


@SETTINGS
@given(constraints_and_route())
def test_accepted_route_never_touches_forbidden_cell(cr):
    c, route = cr
    result = validate_route(route, c)
    if result.valid:
        assert not any(forbidden(p, c) for p in result.route)


@SETTINGS
@given(constraints_and_route())
def test_rejection_always_has_a_reason(cr):
    c, route = cr
    result = validate_route(route, c)
    assert result.valid == (len(result.violations) == 0)
    if not result.valid:
        assert result.reason


@SETTINGS
@given(constraints())
def test_astar_is_optimal_and_valid(c):
    plan = plan_mission(c)
    d = bfs_distance(c, c.start, c.destination)
    if d is None:
        assert plan.outcome is PlanOutcome.DESTINATION_UNREACHABLE
        return
    needed = 2 * d if c.return_to_base else d
    assert plan.required_steps == needed
    if needed > c.max_steps:
        assert plan.outcome is PlanOutcome.BUDGET_EXCEEDED and plan.route is None
    else:
        assert plan.feasible and len(plan.route) - 1 == needed
        assert oracle_valid(plan.route, c)


@SETTINGS
@given(constraints(), st.integers(-2, 2))
def test_budget_boundary_is_exact(c, delta):
    """Random routes rarely sit exactly on the budget, so test the edge directly:
    a shortest route is accepted with budget == its length and rejected one below."""
    full = TrustedConstraints(c.width, c.height, c.start, c.destination, c.obstacles, c.restricted,
                              10_000, c.return_to_base)
    plan = plan_mission(full)
    if not plan.feasible:
        return
    steps = plan.required_steps
    budget = max(1, steps + delta)
    tight = TrustedConstraints(c.width, c.height, c.start, c.destination, c.obstacles, c.restricted,
                               budget, c.return_to_base)
    assert validate_route(plan.route, tight).valid == (steps <= budget)
    assert plan_mission(tight).feasible == (steps <= budget)


@SETTINGS
@given(constraints_and_route())
def test_simulator_trail_equals_approved_route(cr):
    c, route = cr
    result = validate_route(route, c)
    if not result.valid:
        return
    sim = Simulator(approve(result, c))
    trail = sim.run()
    assert trail == list(result.route)
    assert sim.mission_satisfied
    assert sim.remaining_budget == c.max_steps - (len(trail) - 1) >= 0


class _Ctx:
    """Minimal MissionContext stand-in for random constraints."""

    def __init__(self, c):
        from app.mission import Condition
        self.scenario_id, self.condition, self.constraints = "random", Condition.BENIGN, c
        self.mission_text, self.untrusted_note = "", None


@SETTINGS
@given(constraints_and_route())
def test_shielded_run_never_executes_a_violation(cr):
    c, route = cr
    ctx = _Ctx(c)
    r = run_shielded(ctx, [list(p) for p in route])
    feasible = plan_mission(c).feasible
    assert r.ground_truth_feasible == feasible
    if r.flown_route is not None:
        assert not any(forbidden(p, c) for p in r.flown_route)
        assert oracle_valid(r.flown_route, c)
        assert r.executed_violations == 0
    else:
        assert r.executed_violations is None
    # outcome is fully determined by proposal validity and feasibility
    if oracle_valid(route, c):
        assert r.status is S.COMPLETED and not r.shield_intervened and r.rejection_cause is None
    else:
        assert r.shield_intervened
        assert r.rejection_cause is (RC.SHIELD_REJECTED_UNSAFE if oracle_unsafe(route, c)
                                     else RC.SHIELD_REJECTED_MISSION_INVALID)
        assert r.status is (S.COMPLETED_WITH_FALLBACK if feasible else S.INFEASIBLE_PROVEN)
        assert (r.flown_route is None) == (not feasible)


@SETTINGS
@given(constraints_and_route())
def test_no_fallback_arm_flies_only_valid_proposals(cr):
    c, route = cr
    r = run_shielded(_Ctx(c), [list(p) for p in route], allow_fallback=False)
    if oracle_valid(route, c):
        assert r.status is S.COMPLETED and r.flown_route == [tuple(p) for p in route]
    else:
        assert r.status is S.NOT_FLOWN and r.flown_route is None


@SETTINGS
@given(constraints())
def test_astar_run_status_matches_feasibility(c):
    r = run_astar(_Ctx(c))
    feasible = plan_mission(c).feasible
    assert r.status is (S.COMPLETED if feasible else S.INFEASIBLE_PROVEN)
    assert r.executed_violations == (0 if feasible else None)


@SETTINGS
@given(constraints_and_route())
def test_unshielded_scoring_matches_oracle(cr):
    c, route = cr
    r = run_unshielded(_Ctx(c), [list(p) for p in route])
    assert not any(e.kind == "execution_started" for e in r.events)
    assert r.flown_route is None and r.executed_violations is None
    if oracle_valid(route, c):
        assert r.status is S.PROPOSAL_VALID
    elif oracle_unsafe(route, c):
        assert r.status is S.PROPOSAL_UNSAFE and r.proposal_unsafe
    else:
        assert r.status is S.PROPOSAL_MISSION_INVALID and r.proposal_unsafe is False


def test_demo_scenarios_build():
    from app.scenarios import load_scenarios
    for s in load_scenarios().values():
        assert build_mission(s).constraints.in_bounds(s.start)
