import pytest

from app.safety_shield import ApprovedRoute, ShieldRejection, approve, validate_route
from app.simulator import Phase, Simulator

from conftest import make_constraints

GOOD = [[0, 0], [1, 0], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0]]


def sim_for(route, c):
    return Simulator(approve(validate_route(route, c), c))


def test_executes_round_trip(open_grid):
    sim = sim_for(GOOD, open_grid)
    trail = sim.run()
    assert trail == [tuple(p) for p in GOOD]
    assert sim.mission_satisfied
    assert sim.remaining_budget == open_grid.max_steps - 6
    assert sim.phase is Phase.DONE


def test_phase_switches_to_returning_at_destination(open_grid):
    sim = sim_for(GOOD, open_grid)
    for _ in range(3):
        sim.step()
    assert sim.position == (3, 0)
    assert sim.phase is Phase.RETURNING
    assert not sim.mission_satisfied


def test_refuses_unapproved_route(open_grid):
    with pytest.raises(TypeError):
        Simulator(GOOD)  # raw list, never went through the shield


def test_revalidates_forged_approval(walled_grid):
    # An ApprovedRoute built by hand around the shield must still be refused.
    forged = ApprovedRoute(tuple(tuple(p) for p in GOOD), walled_grid)
    with pytest.raises(ShieldRejection):
        Simulator(forged)


def test_reset_returns_to_base(open_grid):
    sim = sim_for(GOOD, open_grid)
    sim.run()
    sim.reset()
    assert sim.position == open_grid.start
    assert sim.trail == [open_grid.start]
    assert sim.remaining_budget == open_grid.max_steps
    assert sim.phase is Phase.OUTBOUND


def test_cancel_stops_and_is_not_success(open_grid):
    sim = sim_for(GOOD, open_grid)
    sim.step()
    sim.cancel()
    assert sim.finished
    assert not sim.mission_satisfied


def test_one_way_mission_satisfied_at_destination():
    c = make_constraints(return_to_base=False)
    sim = sim_for(GOOD[:4], c)
    sim.run()
    assert sim.mission_satisfied
