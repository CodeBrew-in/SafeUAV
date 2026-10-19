"""Conditional missions: the drone decides during the flight, but only by skipping stops or ending early."""

import pytest

from app.geo.campus import load_campus, norm
from app.geo.controller import Command, Controller
from app.geo.flight import Wind
from app.geo.session import merge_reply
from app.geo.spec import Condition, MissionSpec, spec_from_json, validate_spec

C = load_campus()
CALM = Wind(0.0, 0.0)
LIB, HOSTEL, CANTEEN = "Nava Nalanda Library", "Directorate", "SBOP Lawns"


def fly(spec, *, battery_pct=100.0, cap=40.0):
    ctl = Controller(C, capacity_wh=cap, forecast=CALM, actual=CALM, clock_min=600)
    ctl.energy = cap * battery_pct / 100
    ctl.submit(Command("order", spec, "test"))
    ctl.run()
    return ctl


def test_json_round_trip_and_strict_parsing():
    d = {"targets": [LIB, HOSTEL], "avoid": [], "stay_at_end": False, "refuse": False,
         "conditions": [{"kind": "skip_if_battery_below", "places": [HOSTEL], "battery_pct": 40}]}
    spec = spec_from_json(d)
    assert spec.conditions == (Condition("skip_if_battery_below", (HOSTEL,), 40.0),)
    assert spec_from_json(spec.to_json()) == spec
    assert "conditions" not in MissionSpec((LIB,)).to_json()
    for bad in ([{"kind": "teleport", "places": []}],
                [{"kind": "skip_if_battery_below", "places": [LIB, HOSTEL], "battery_pct": 40}],
                [{"kind": "skip_if_battery_below", "places": [LIB], "battery_pct": 0}],
                [{"kind": "skip_if_battery_below", "places": [LIB], "battery_pct": float("nan")}],
                [{"kind": "skip_if_battery_below", "places": [LIB], "battery_pct": True}],
                [{"kind": "closest_of", "places": [LIB]}], "skip everything", [{"kind": "stay_if_battery_below"}] * 5):
        with pytest.raises(ValueError):
            spec_from_json({**d, "conditions": bad})


def test_a_condition_may_only_name_targets_and_real_places():
    spec = MissionSpec((LIB,), conditions=(Condition("skip_if_battery_below", (HOSTEL,), 40.0),))
    chk = validate_spec(spec, C, now_min=600, horizon_min=60)
    assert not chk.ok and "must also be one of the targets" in chk.errors[0]
    assert spec.pruned().conditions == ()                                 # removing the target drops the rule


def test_skip_if_battery_below_skips_only_when_low():
    spec = MissionSpec((LIB, HOSTEL), conditions=(Condition("skip_if_battery_below", (HOSTEL,), 60.0),))
    full = fly(spec, battery_pct=100)
    assert full.state == "DONE" and norm(HOSTEL) in full.visited and not full.skipped
    low = fly(spec, battery_pct=55)
    assert low.state == "DONE" and norm(HOSTEL) not in low.visited and norm(HOSTEL) in low.skipped
    assert any(e["kind"] == "condition" and "skipping" in e["msg"] for e in low.audit)
    assert low.energy > low.floor_wh()                                    # the reserve is untouched


def test_skip_is_decided_when_the_stop_is_next_not_at_the_start():
    """Battery is high at take-off, so the first stop is flown; a later stop is judged when it comes up."""
    spec = MissionSpec((LIB, HOSTEL), conditions=(Condition("skip_if_battery_below", (HOSTEL,), 99.0),))
    ctl = fly(spec, battery_pct=100)
    assert norm(LIB) in ctl.visited and norm(HOSTEL) in ctl.skipped       # battery fell under 99% after the first leg


def test_closest_of_visits_only_the_nearer_place():
    spec = MissionSpec((CANTEEN, HOSTEL), conditions=(Condition("closest_of", (CANTEEN, HOSTEL), None),))
    ctl = fly(spec)
    near = min((CANTEEN, HOSTEL), key=lambda p: ((C.find(p).center[0] - C.base[0]) ** 2 + (C.find(p).center[1] - C.base[1]) ** 2))
    far = HOSTEL if near == CANTEEN else CANTEEN
    assert ctl.state == "DONE" and ctl.visited == [norm(near)] and norm(far) in ctl.skipped


def test_stay_if_battery_below_lands_at_the_last_stop_instead_of_flying_home():
    spec = MissionSpec((LIB,), conditions=(Condition("stay_if_battery_below", (), 99.5),))
    ctl = fly(spec, battery_pct=100)
    assert ctl.state == "DONE" and ctl.force_stay
    assert any("instead of flying home" in e["msg"] for e in ctl.audit)
    home = fly(MissionSpec((LIB,), conditions=(Condition("stay_if_battery_below", (), 5.0),)))
    assert home.state == "DONE" and not home.force_stay


def test_conditions_never_bypass_the_shield():
    """A rule cannot make the drone fly where the shield forbids: every re-plan goes through preflight."""
    spec = MissionSpec((LIB, HOSTEL), conditions=(Condition("closest_of", (LIB, HOSTEL), None),))
    ctl = fly(spec, battery_pct=100)
    routes = [e for e in ctl.audit if e["kind"] == "route"]
    assert routes and all(r["data"]["points"] for r in routes)
    assert not [e for e in ctl.audit if e["kind"] in ("crash", "violation")] and ctl.state == "DONE"


def test_merge_reply_keeps_adds_and_removes_conditions():
    cond = Condition("skip_if_battery_below", (HOSTEL,), 40.0)
    base = MissionSpec((LIB, HOSTEL))
    added = MissionSpec((LIB, HOSTEL), conditions=(cond,))
    assert merge_reply(base, added, base).conditions == (cond,)
    assert merge_reply(added, base, added).conditions == ()               # "forget that rule"
    assert merge_reply(added, added, added).conditions == (cond,)         # untouched stays
    gone = MissionSpec((LIB,))
    assert merge_reply(added, gone, added).conditions == ()               # target removed: rule goes with it


class CondLLM:
    def ask(self, system, user, schema, seed=0, max_tokens=0):
        from app.geo.llm import LLMReply
        d = {"targets": [LIB, HOSTEL], "avoid": [], "stay_at_end": False, "refuse": False, "reason": "",
             "conditions": [{"kind": "skip_if_battery_below", "places": [HOSTEL], "battery_pct": 60}]}
        return LLMReply("fake", "fake", str(d), d, 1.0, None)


def test_compare_shows_ai_reading_the_condition_and_the_basic_parser_missing_it():
    from app.geo.session import Session
    s = Session(llm_factory=lambda m: CondLLM())
    order = f"inspect the {LIB}, then {HOSTEL}, but skip {HOSTEL} if the battery is below 60%"
    r = s.compare(order, 55)
    assert r["ai"]["shield"] == "accepted" and r["ai"]["skipped"] == [HOSTEL] and r["ai"]["visited"] == [LIB]
    assert r["ai"]["rules"] and not r["classic"].get("rules")
    full = s.compare(order, 100)
    assert full["ai"]["visited"] == [LIB, HOSTEL] and full["ai"]["skipped"] == []
    with pytest.raises(Exception):
        s.compare("   ", 100)
    assert s.compare(order, "abc" if False else 5)["battery_pct"] == 15.0          # clamped, never below 15%
    assert s.ctl.state == "PREFLIGHT" and not s.started                              # the live drone was never touched
