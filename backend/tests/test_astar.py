from app.astar import PlanOutcome, plan_mission, shortest_path
from app.safety_shield import validate_route

from conftest import make_constraints


def test_shortest_path_open_grid(open_grid):
    path = shortest_path(open_grid, (0, 0), (3, 0))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_detours_around_restricted_zone(walled_grid):
    path = shortest_path(walled_grid, (0, 0), (3, 0))
    assert path is not None
    assert not any(c in walled_grid.restricted for c in path)
    # up 4, right 3, down 4
    assert len(path) - 1 == 11


def test_round_trip_route_is_valid_and_returns(walled_grid):
    plan = plan_mission(walled_grid)
    assert plan.feasible
    assert plan.route[0] == walled_grid.start and plan.route[-1] == walled_grid.start
    assert plan.required_steps == 22
    assert validate_route(plan.route, walled_grid).valid


def test_one_way_mission_ends_at_destination():
    c = make_constraints(return_to_base=False)
    plan = plan_mission(c)
    assert plan.route[-1] == c.destination
    assert plan.required_steps == 3


def test_unreachable_destination_reported():
    # destination fully enclosed by obstacles
    c = make_constraints(destination=(4, 4), obstacles=[(3, 4), (5, 4), (4, 3), (4, 5)])
    plan = plan_mission(c)
    assert plan.outcome is PlanOutcome.DESTINATION_UNREACHABLE
    assert plan.route is None


def test_destination_inside_restricted_zone_is_infeasible():
    c = make_constraints(destination=(3, 0), restricted=[(3, 0)])
    assert plan_mission(c).outcome is PlanOutcome.DESTINATION_UNREACHABLE


def test_budget_covers_return_leg(walled_grid):
    # outbound alone (11) fits in 15, but the 22-step round trip does not
    c = make_constraints(restricted=walled_grid.restricted, max_steps=15)
    plan = plan_mission(c)
    assert plan.outcome is PlanOutcome.BUDGET_EXCEEDED
    assert plan.required_steps == 22


def test_astar_is_deterministic(walled_grid):
    assert plan_mission(walled_grid).route == plan_mission(walled_grid).route
