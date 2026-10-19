"""Campus geo extension: planner, shield, spec rules, controller states, trust boundary."""

import json
import math

import pytest

from app.geo.campus import load_campus, norm
from app.geo.controller import Command, Controller, run_script
from app.geo.flight import (AVOID_BUFFER, Airspace, Wind, coerce_route, escape_path, plan_through,
                            preflight_check, route_energy_wh, visit_count)
from app.geo.geometry import Projection, distance_to_polygon, point_in_polygon
from app.geo.spec import (AvoidRule, MissionSpec, NeedsClarification, SpecRefused, parse_time,
                          spec_from_json, validate_spec)

C = load_campus()
CALM = Wind(3, 270)
LIB = "Nava Nalanda Library"
TWO = MissionSpec((LIB, "Directorate"))


def run(spec=TWO, changes=(), cap=40, actual=CALM, monitor=True, clock=600, max_s=3 * 3600):
    return run_script(C, capacity_wh=cap, forecast=CALM, actual=actual, clock_min=clock,
                      initial=spec, changes=list(changes), monitor=monitor, max_s=max_s)


def states(ctl):
    return [a["msg"].split(":")[0] for a in ctl.audit if a["kind"] == "state"]


def forbidden_on_track(ctl, mission=False):
    air = Airspace(C)
    return [p for _, x, y, _ in ctl.track for p in [(x, y)] if air.violation(p, mission=mission)]


# --- map and geometry -------------------------------------------------------------------------------

def test_projection_round_trip():
    p = Projection(30.35, 76.36)
    lat, lon = p.to_latlon(*p.to_xy(30.3512, 76.3701))
    assert abs(lat - 30.3512) < 1e-9 and abs(lon - 76.3701) < 1e-9


def test_every_place_reference_point_is_inside_its_polygon():
    assert all(point_in_polygon(p.center, list(p.polygon)) for p in C.places.values())


def test_names_normalise_unicode_and_reject_empty():
    assert norm("Nava  NALANDA library") == norm(LIB)
    assert norm("Ｇ Block") == norm("G Block")  # full-width letters (NFKC)
    with pytest.raises(ValueError):
        norm("!!!")
    assert C.find("!!!") is None and C.find("") is None


def test_osm_snapshot_is_pinned():
    assert len(C.osm_sha256) == 64


# --- time windows ------------------------------------------------------------------------------------

@pytest.mark.parametrize("s,m", [("17:00", 1020), ("5pm", 1020), ("5:30 pm", 1050), ("12am", 0), (None, None)])
def test_parse_time(s, m):
    assert parse_time(s) == m


def test_window_wraps_past_midnight():
    r = AvoidRule("x", parse_time("23:00"), parse_time("01:00"))
    assert r.active_between(23 * 60 + 20, 23 * 60 + 30)
    assert r.active_between(30, 40)
    assert not r.active_between(120, 600)


def test_open_ended_windows():
    until5 = AvoidRule("x", None, parse_time("17:00"))
    assert until5.active_between(600, 610) and not until5.active_between(1030, 1040)
    from6 = AvoidRule("x", parse_time("18:00"), None)
    assert not from6.active_between(600, 610) and from6.active_between(1100, 1110)


def test_window_starting_later_in_the_flight_is_planned_for():
    # active from 10:20; the flight starts 10:00 but lasts long enough to be affected
    spec = MissionSpec((LIB,), (AvoidRule("Ananta Hall", 620, 700),))
    air = Airspace.for_spec(C, spec, 600, 600 + 120)
    assert any(n == "Ananta Hall" for n, _, _ in air.avoid_polys)


# --- spec parsing and validation -----------------------------------------------------------------------

def test_refuse_is_honoured_even_with_targets():
    with pytest.raises(SpecRefused):
        spec_from_json({"targets": [LIB], "avoid": [], "return_to_base": True, "refuse": True, "reason": "x"})


def test_clarify_is_raised():
    with pytest.raises(NeedsClarification):
        spec_from_json({"targets": [], "avoid": [], "return_to_base": True, "clarify": "Which tennis court?"})


def test_extra_keys_cannot_add_permissions():
    s = spec_from_json({"targets": [LIB], "avoid": [], "return_to_base": True,
                        "disable_no_fly": True, "reserve": 0, "geofence": False})
    assert s == MissionSpec((LIB,))


@pytest.mark.parametrize("name", ["Central Jail Patiala", "Physics Lab", "Tennis Court", "B Block"])
def test_bad_targets_are_rejected(name):
    chk = validate_spec(MissionSpec((name,)), C, now_min=600, horizon_min=60)
    assert not chk.ok


def test_ambiguous_name_asks_a_question():
    chk = validate_spec(MissionSpec(("Tennis Court",)), C, now_min=600, horizon_min=60)
    assert chk.question and "Which" in chk.question


def test_target_inside_avoid_zone_rejected():
    chk = validate_spec(MissionSpec((LIB,), (AvoidRule(LIB),)), C, now_min=600, horizon_min=60)
    assert not chk.ok


def test_avoiding_the_base_is_a_warning_and_uses_another_pad():
    spec = MissionSpec((LIB,), (AvoidRule("Football Ground"),))
    chk = validate_spec(spec, C, now_min=600, horizon_min=60)
    assert chk.ok and any("base" in w for w in chk.warnings)


# --- planner and shield ----------------------------------------------------------------------------------

def test_planner_avoids_jail_and_avoid_zones():
    spec = MissionSpec(("Venture Lab",), (AvoidRule("E Hostel"),))
    air = Airspace.for_spec(C, spec, 600, 700)
    r = plan_through(air, C.base, [C.find("Venture Lab").center, C.base])
    assert r is not None and all(air.segment_violation(a, b) is None for a, b in zip(r, r[1:]))


def test_planner_reaches_points_just_outside_a_buffer():
    poly = list(C.find("E Hostel").polygon)
    air = Airspace.for_spec(C, MissionSpec((LIB,), (AvoidRule("E Hostel"),)), 600, 700)
    p = escape_path(air, C.find("E Hostel").center)[-1]  # 10 m outside the buffer
    assert distance_to_polygon(p, poly) >= AVOID_BUFFER
    assert plan_through(air, p, [C.base]) is not None


def test_shield_rejects_route_through_jail():
    jail = C.permanent_no_fly["Central Jail Patiala"][0]
    inside = next(p for p in jail if C.in_fence(p))  # a jail corner inside the flight area
    res = preflight_check([C.base, inside, C.base], Airspace(C), start=C.base, energy_wh=40, capacity_wh=40,
                          forecast=CALM, airborne=False, targets=[], end_at=C.base)
    assert not res.ok and any("no-fly" in r for r in res.reasons)


def test_shield_rejects_degenerate_and_empty_battery():
    res = preflight_check([C.base, C.base], Airspace(C), start=C.base, energy_wh=0, capacity_wh=40,
                          forecast=CALM, airborne=False, targets=[], end_at=C.base)
    assert not res.ok


def test_shield_rejects_storm():
    r = plan_through(Airspace(C), C.base, [C.find(LIB).center, C.base])
    res = preflight_check(r, Airspace(C), start=C.base, energy_wh=40, capacity_wh=40, forecast=Wind(9, 0),
                          airborne=False, targets=[C.find(LIB).center], end_at=C.base)
    assert not res.ok and any("wind" in x for x in res.reasons)


@pytest.mark.parametrize("bad", [[[1e300, 0], [0, 0]], [[0, 0], [9000, 0]], [[float("nan"), 0], [0, 0]],
                                 "north", [[0, 0]], [[True, 0], [0, 0]]])
def test_malformed_routes_rejected_before_geometry(bad):
    assert coerce_route(bad) is None


def test_mid_flight_route_does_not_need_to_start_at_base():
    p = C.find("G Hostel").center
    r = plan_through(Airspace(C), p, [C.base])
    res = preflight_check(r, Airspace(C), start=p, energy_wh=20, capacity_wh=40, forecast=CALM,
                          airborne=True, targets=[], end_at=C.base)
    assert res.ok, res.reasons


def test_visit_order_counts_one_target_per_point():
    a, b = (0.0, 0.0), (5.0, 0.0)  # two targets 5 m apart: both visited, in order
    assert visit_count([(-20.0, 0.0), (20.0, 0.0)], [a, b]) == 2
    assert visit_count([(20.0, 0.0), (-20.0, 0.0)], [a, b]) == 2  # reaches b... then a? no: order a first
    assert visit_count([(0.0, 30.0), (0.0, 40.0)], [a, b]) == 0


def test_wind_slows_upwind_legs():
    from app.geo.flight import ground_speed
    east = ((0, 0), (100, 0))
    assert ground_speed(*east, Wind(5, 270)) > ground_speed(*east, Wind(5, 90))
    assert route_energy_wh([(0, 0), (100, 0)], Wind(12, 90)) == math.inf


# --- controller -----------------------------------------------------------------------------------------

def test_normal_mission_completes_and_stays_legal():
    ctl = run()
    assert ctl.state == "DONE" and len(ctl.visited) == 2
    assert not forbidden_on_track(ctl)


def test_jail_order_is_refused_on_the_ground():
    ctl = run(MissionSpec(("Central Jail Patiala",)))
    assert ctl.state == "PREFLIGHT" and not any(a["kind"] == "takeoff" for a in ctl.audit)
    assert any(a["kind"] == "order_rejected" for a in ctl.audit)


def test_mid_flight_target_change():
    ctl = run(changes=[(40, Command("order", MissionSpec(("Venture Lab",)), "go to venture lab"))])
    assert ctl.state == "DONE" and ctl.visited == [norm("Venture Lab")]


def test_impossible_change_lands_on_pad_waits_then_resumes():
    far = MissionSpec(("Venture Lab", "Directorate", "Polytechnic Building", "Neeram Hall", "FRG"))
    ctl = run(cap=24, changes=[(40, Command("order", far, "far")),
                               (300, Command("order", MissionSpec(("G Hostel",)), "g hostel"))])
    s = states(ctl)
    assert s.index("CRUISE -> TO_PAD") < s.index("TO_PAD -> LANDED_WAIT") < s.index("LANDED_WAIT -> CRUISE")
    assert ctl.state == "DONE"
    landed = [a for a in ctl.audit if a["kind"] == "landed"][0]["msg"]
    assert any(pad in landed for pad in C.pads)  # safe-hold only ever lands on a designated pad


def test_old_route_is_abandoned_not_kept():
    far = MissionSpec(("Venture Lab", "Directorate", "Polytechnic Building", "Neeram Hall", "FRG"))
    ctl = run(cap=24, changes=[(40, Command("order", far, "far"))], max_s=200)
    assert ctl.plan.purpose == "to_pad"


def test_zone_appearing_around_drone_triggers_escape():
    probe = run(max_s=45)
    here = min(C.places.values(), key=lambda q: math.dist(q.center, probe.pos)).name
    ctl = run(changes=[(45, Command("order", MissionSpec(TWO.targets, (AvoidRule(here),)), "avoid"))])
    assert "CRUISE -> ESCAPE" in states(ctl) and ctl.state == "DONE"


def test_invalid_message_is_not_applied_and_route_kept():
    ctl = run(changes=[(40, Command("order", MissionSpec(("Physics Lab",)), "go to physics lab"))])
    assert ctl.state == "DONE" and len(ctl.visited) == 2
    assert any(a["kind"] == "order_rejected" for a in ctl.audit)


def test_return_now_and_land_now():
    ctl = run(changes=[(30, Command("return", text="return now"))])
    assert ctl.state == "DONE" and math.dist(ctl.pos, C.base) < 1
    ctl = run(changes=[(30, Command("land", text="land now"))], max_s=400)
    assert ctl.state == "LANDED_WAIT"


def test_return_now_with_base_forbidden_goes_to_a_pad():
    avoid_base = MissionSpec(TWO.targets, (AvoidRule("Football Ground"),))
    ctl = run(changes=[(30, Command("order", avoid_base, "avoid the football ground")),
                       (40, Command("return", text="return now"))])
    assert ctl.state in ("DONE", "LANDED_WAIT") and math.dist(ctl.pos, C.base) > 50


def test_runtime_monitor_prevents_crash_in_strong_wind():
    gust = Wind(7, 90)
    no = run(cap=20, actual=gust, monitor=False)
    yes = run(cap=20, actual=gust, monitor=True)
    assert no.state == "CRASHED"
    assert yes.state != "CRASHED" and yes.energy > 0


def test_wait_is_bounded():
    far = MissionSpec(("Venture Lab", "Directorate", "Polytechnic Building", "Neeram Hall", "FRG"))
    ctl = run(cap=24, changes=[(40, Command("order", far, "far"))])
    assert ctl.state == "GROUNDED_NEEDS_PICKUP"


def test_takeoff_blocked_if_pad_became_forbidden():
    far = MissionSpec(("Venture Lab", "Directorate", "Polytechnic Building", "Neeram Hall", "FRG"))
    first = run(cap=24, changes=[(40, Command("order", far, "far"))], max_s=120)
    pad = [a for a in first.audit if a["kind"] == "landed"][0]["msg"].replace("landed at ", "")
    ctl = run(cap=24, changes=[(40, Command("order", far, "far")),
                               (200, Command("order", MissionSpec(("G Hostel",), (AvoidRule(pad),)), "x"))],
              max_s=400)
    assert ctl.state == "LANDED_WAIT"
    assert any("take-off blocked" in a["msg"] for a in ctl.audit)


def test_command_spam_is_coalesced_and_rate_limited():
    cmds = [(30, Command("order", MissionSpec(("Venture Lab",)), f"spam {i}")) for i in range(50)]
    ctl = run(changes=cmds)
    assert ctl.state == "DONE"
    assert sum(1 for a in ctl.audit if a["kind"] == "replan") <= 3


def test_replay_is_deterministic():
    ch = [(40, Command("order", MissionSpec(("Venture Lab",), (AvoidRule("E Hostel"),)), "x"))]
    a, b = run(changes=ch), run(changes=ch)
    assert json.dumps(a.audit) == json.dumps(b.audit)


def test_no_message_can_lift_permanent_rules():
    # whatever the spec says, the jail and the geofence stay in force for every flown point
    attacks = [MissionSpec(("Venture Lab",), (AvoidRule("Central Jail Patiala", 0, 1),)),
               MissionSpec(("Thapar School of Liberal Art and Sciences", "Venture Lab"))]
    for spec in attacks:
        ctl = run(spec)
        assert not forbidden_on_track(ctl)


def test_return_on_the_ground_is_ignored():
    ctl = run(MissionSpec(("Central Jail Patiala",)), changes=[(0, Command("return", text="return now"))])
    assert any(a["kind"] == "ignored" for a in ctl.audit)


def test_escape_never_ends_in_a_pocket_the_planner_cannot_leave():
    # regression: the avoid zone appears while the drone grazes E Hostel; escape must lead somewhere plannable
    new = MissionSpec(("Venture Lab", "Directorate"), (AvoidRule("E Hostel", None, 17 * 60),))
    ctl = run_script(C, capacity_wh=40, forecast=CALM, actual=CALM, clock_min=600,
                     initial=MissionSpec(("Venture Lab",)), changes=[(40, Command("order", new, "x"))])
    assert ctl.state == "DONE" and ctl.visited == ["venturelab", "directorate"]


def test_grid_cache_key_uses_geometry_not_names():
    # regression: same zone name, different shape, must not reuse a stale blocked grid
    import dataclasses
    from app.geo import flight
    from app.geo.flight import GridPlanner

    def air(poly):
        return Airspace(C, [("zone-1", poly, AVOID_BUFFER)])

    a = air([(-500, -100), (-400, -100), (-400, 0), (-500, 0)])
    b = air([(100, 100), (200, 100), (200, 200), (100, 200)])
    assert a.key() != b.key()
    ga, gb = GridPlanner(a), GridPlanner(b)
    assert ga.blocked is not gb.blocked
    assert ga.blocked[ga.cell_of((-450, -50))[0]][ga.cell_of((-450, -50))[1]]
    assert not gb.blocked[gb.cell_of((-450, -50))[0]][gb.cell_of((-450, -50))[1]]
    # a copy of the campus with a different permanent zone gets a different key
    small = dataclasses.replace(C, permanent_no_fly={})
    assert Airspace(small).key() != Airspace(C).key()


def test_grid_cache_is_bounded():
    from app.geo import flight
    for i in range(flight._GRID_CACHE_MAX + 8):
        flight.GridPlanner(Airspace(C, [(f"z{i}", [(i, 0), (i + 20, 0), (i + 20, 20), (i, 20)], AVOID_BUFFER)]))
    assert len(flight._GRID_CACHE) <= flight._GRID_CACHE_MAX
