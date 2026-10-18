import pytest

from app.safety_shield import ShieldRejection, ViolationCode as V, approve, validate_route

from conftest import make_constraints

GOOD = [[0, 0], [1, 0], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0]]


def codes(route, c):
    return validate_route(route, c).codes


def test_valid_round_trip_accepted(open_grid):
    r = validate_route(GOOD, open_grid)
    assert r.valid, r.reason
    assert r.steps == 6


def test_rejects_waypoint_outside_map(open_grid):
    route = [[0, 0], [-1, 0], [0, 0]]
    assert V.OUT_OF_BOUNDS in codes(route, open_grid)


def test_rejects_obstacle():
    c = make_constraints(obstacles=[(2, 0)])
    assert V.OBSTACLE in codes(GOOD, c)


def test_rejects_restricted_zone(walled_grid):
    r = validate_route(GOOD, walled_grid)  # straight through x=2
    assert not r.valid
    assert V.RESTRICTED_ZONE in r.codes
    assert "restricted zone" in r.reason


def test_rejects_non_adjacent_move(open_grid):
    route = [[0, 0], [3, 0], [0, 0]]  # teleport
    assert V.NON_ADJACENT_MOVE in codes(route, open_grid)


def test_rejects_diagonal_move(open_grid):
    route = [[0, 0], [1, 1], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0]]
    assert V.NON_ADJACENT_MOVE in codes(route, open_grid)


def test_rejects_standing_still(open_grid):
    route = [[0, 0], [0, 0], [1, 0], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0]]
    assert V.NON_ADJACENT_MOVE in codes(route, open_grid)


def test_budget_includes_return_leg():
    c = make_constraints(max_steps=5)  # outbound is 3, round trip is 6
    assert V.BUDGET_EXCEEDED in codes(GOOD, c)


def test_rejects_missing_return(open_grid):
    assert V.NOT_RETURNED_TO_BASE in codes(GOOD[:4], open_grid)


def test_rejects_never_reaching_destination(open_grid):
    assert V.DESTINATION_NOT_REACHED in codes([[0, 0], [1, 0], [0, 0]], open_grid)


def test_rejects_wrong_start(open_grid):
    route = [[1, 0], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0]]
    assert V.WRONG_START in codes(route, open_grid)


def test_one_way_must_end_at_destination():
    c = make_constraints(return_to_base=False)
    assert validate_route(GOOD[:4], c).valid
    assert V.DOES_NOT_END_AT_DESTINATION in codes(GOOD, c)


@pytest.mark.parametrize("raw", [
    None,
    "go straight to (3,0)",
    {"waypoints": GOOD},
    [[0, 0], [1]],
    [[0, 0], [1, 0, 0]],
    [[0, 0], ["1", "0"]],
    [[0, 0], [1.0, 0.0]],
    [[0, 0], [True, False]],
    [[0, 0], None],
])
def test_malformed_output_rejected(open_grid, raw):
    r = validate_route(raw, open_grid)
    assert not r.valid
    assert r.codes == {V.MALFORMED}
    with pytest.raises(ShieldRejection):
        approve(r, open_grid)


def test_empty_route_rejected(open_grid):
    assert codes([], open_grid) == {V.EMPTY_ROUTE}


def test_absurdly_long_input_refused(open_grid):
    assert codes([[0, 0]] * 100_001, open_grid) == {V.MALFORMED}


def test_long_but_wellformed_route_is_over_budget(open_grid):
    assert V.BUDGET_EXCEEDED in codes(GOOD + GOOD[1:] * 10, open_grid)


def test_cell_both_obstacle_and_restricted_reports_both():
    c = make_constraints(obstacles=[(2, 0)], restricted=[(2, 0)])
    assert {V.OBSTACLE, V.RESTRICTED_ZONE} <= codes(GOOD, c)


def test_missing_destination_still_reports_wrong_ending(open_grid):
    route = [[0, 0], [1, 0], [1, 1]]
    assert {V.DESTINATION_NOT_REACHED, V.NOT_RETURNED_TO_BASE} <= codes(route, open_grid)


def test_unsafe_and_mission_invalid_classification(open_grid, walled_grid):
    assert validate_route(GOOD, walled_grid).unsafe            # through restricted cells
    short = validate_route(GOOD[:3] + [[1, 0], [0, 0]], open_grid)
    assert short.mission_invalid and not short.unsafe           # safe but never reaches (3, 0)
    assert not validate_route(GOOD, open_grid).unsafe


def test_swept_cells_x_then_y():
    from app.safety_shield import swept_cells
    assert swept_cells((0, 0), (2, 2)) == [(1, 0), (2, 0), (2, 1)]
    assert swept_cells((0, 0), (1, 0)) == []
    assert swept_cells((3, 3), (3, 3)) == []


def test_reports_all_violations_not_just_first(walled_grid):
    route = [[0, 0], [2, 0], [9, 9]]
    got = codes(route, walled_grid)
    assert {V.RESTRICTED_ZONE, V.OUT_OF_BOUNDS, V.NON_ADJACENT_MOVE, V.DESTINATION_NOT_REACHED} <= got


def test_tuples_and_lists_both_accepted(open_grid):
    assert validate_route([tuple(p) for p in GOOD], open_grid).valid


def test_approve_returns_route_only_when_valid(open_grid):
    approved = approve(validate_route(GOOD, open_grid), open_grid)
    assert approved.waypoints[0] == (0, 0)
