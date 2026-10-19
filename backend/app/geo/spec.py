"""Mission specification: what the operator asked for, in checkable form.

A spec comes from the operator's English order (via an LLM or the regex parser) or
is written by hand as ground truth. It can only choose targets and ADD zones to
avoid: there is no field for the permanent no-fly zone, the geofence or the
battery reserve, so no order or chat message can change those.

Times are minutes on a simulated local clock (Asia/Kolkata) for the mission day.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

TIME_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.I)
DAY = 24 * 60


class SpecRefused(ValueError):
    """The model (or parser) declined the order; `reason` says why."""


class NeedsClarification(ValueError):
    """The order is ambiguous; `question` is what to ask the operator."""


@dataclass(frozen=True)
class AvoidRule:
    zone: str
    start_min: int | None = None  # None = already in force
    end_min: int | None = None    # None = until further notice

    def intervals(self) -> list[tuple[float, float]]:
        """The rule's active intervals on the mission-day clock (a window past midnight wraps)."""
        s = -math.inf if self.start_min is None else self.start_min
        e = math.inf if self.end_min is None else self.end_min
        if s <= e:
            return [(s, e)]
        return [(s, e + DAY), (s - DAY, e)]  # e.g. 23:00-01:00

    def active_between(self, t0: float, t1: float) -> bool:
        """In force at any moment of [t0, t1]?"""
        return any(a <= t1 and t0 <= b for a, b in self.intervals())


COND_KINDS = ("skip_if_battery_below", "closest_of", "stay_if_battery_below")
MAX_CONDITIONS = 4


@dataclass(frozen=True)
class Condition:
    """A rule the drone decides DURING the flight. A closed list of three kinds, so an order can only
    skip stops or end the flight early: it can never loosen a safety rule.

    skip_if_battery_below  places=(X,)  : skip stop X if the battery is under `pct` % when X is next.
    closest_of             places=(A,B) : of these stops, visit only the one nearest the drone when the first is next.
    stay_if_battery_below  places=()    : after the last stop, land there instead of flying home if under `pct` %.
    Every place named here must also be one of the mission's targets.
    """
    kind: str
    places: tuple[str, ...] = ()
    pct: float | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind, "places": list(self.places), "battery_pct": self.pct}

    def describe(self) -> str:
        if self.kind == "skip_if_battery_below":
            return f"skip {self.places[0]} if the battery is below {self.pct:g}%"
        if self.kind == "closest_of":
            return "visit only the nearest of " + ", ".join(self.places)
        return f"stay at the last place if the battery is below {self.pct:g}% (otherwise return to base)"


@dataclass(frozen=True)
class MissionSpec:
    targets: tuple[str, ...]           # inspection targets, in order
    avoid: tuple[AvoidRule, ...] = ()
    return_to_base: bool = True
    conditions: tuple[Condition, ...] = ()

    def to_json(self) -> dict:
        def hhmm(m):
            return None if m is None else f"{m // 60:02d}:{m % 60:02d}"
        d = {"targets": list(self.targets),
             "avoid": [{"zone": r.zone, "from": hhmm(r.start_min), "to": hhmm(r.end_min)} for r in self.avoid],
             "stay_at_end": not self.return_to_base}
        if self.conditions:
            d["conditions"] = [c.to_json() for c in self.conditions]
        return d

    def pruned(self) -> "MissionSpec":
        """Drop condition places that are no longer targets, and conditions left with nothing to decide."""
        from .campus import norm

        def key(n):
            try:
                return norm(n)
            except ValueError:
                return None
        have = {key(t) for t in self.targets}
        out = []
        for c in self.conditions:
            if c.kind == "closest_of":
                places = tuple(p for p in c.places if key(p) in have)
                if len(places) >= 2:
                    out.append(Condition(c.kind, places, c.pct))
            elif c.kind == "skip_if_battery_below":
                if c.places and key(c.places[0]) in have:
                    out.append(c)
            else:
                out.append(c)
        return self if tuple(out) == self.conditions else MissionSpec(self.targets, self.avoid, self.return_to_base, tuple(out))


@dataclass
class SpecCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spec: MissionSpec | None = None
    question: str | None = None  # set when the fix is to ask the operator


def parse_time(s) -> int | None:
    """'17:00', '5pm', '5:30 pm' -> minutes after midnight; None/'' -> None."""
    if s is None or (isinstance(s, str) and s.strip().lower() in ("", "null", "none")):
        return None
    m = TIME_RE.match(str(s))
    if not m:
        raise ValueError(f"bad time {s!r}, expected HH:MM")
    h, mm, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ampm:
        if not 1 <= h <= 12:
            raise ValueError(f"bad time {s!r}")
        h = h % 12 + (12 if ampm == "pm" else 0)
    if not (0 <= h < 24 and 0 <= mm < 60):
        raise ValueError(f"bad time {s!r}")
    return h * 60 + mm


def spec_from_json(obj: object) -> MissionSpec:
    """Build a spec from model JSON. Raises SpecRefused / NeedsClarification / ValueError."""
    if not isinstance(obj, dict):
        raise ValueError(f"expected an object, got {type(obj).__name__}")
    if obj.get("refuse") is True:
        raise SpecRefused(str(obj.get("reason") or "refused")[:300])
    q = obj.get("clarify")
    if isinstance(q, str) and q.strip():
        raise NeedsClarification(q.strip()[:300])
    targets = obj.get("targets")
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError("'targets' must be a list of place names")
    avoid_raw = obj.get("avoid", [])
    if not isinstance(avoid_raw, list) or len(avoid_raw) > 20:
        raise ValueError("'avoid' must be a list of at most 20 rules")
    rules = []
    for a in avoid_raw:
        if not isinstance(a, dict) or not isinstance(a.get("zone"), str):
            raise ValueError(f"bad avoid rule {a!r}"[:200])
        rules.append(AvoidRule(a["zone"], parse_time(a.get("from")), parse_time(a.get("to"))))
    if "stay_at_end" in obj:  # the LLM-facing field; defaulting to false is what small models do anyway
        stay = obj["stay_at_end"]
        if not isinstance(stay, bool):
            raise ValueError("'stay_at_end' must be true or false")
        rtb = not stay
    else:
        rtb = obj.get("return_to_base", True)
        if not isinstance(rtb, bool):
            raise ValueError("'return_to_base' must be true or false")
    if len(targets) > 10:
        raise ValueError("at most 10 targets")
    return MissionSpec(tuple(targets), tuple(rules), rtb, _conditions_from_json(obj.get("conditions")))


def _conditions_from_json(raw) -> tuple[Condition, ...]:
    """Strict: an unknown kind, a wrong shape or a silly percentage rejects the whole answer."""
    if raw in (None, []):
        return ()
    if not isinstance(raw, list) or len(raw) > MAX_CONDITIONS:
        raise ValueError(f"'conditions' must be a list of at most {MAX_CONDITIONS} rules")
    out = []
    for c in raw:
        if not isinstance(c, dict) or c.get("kind") not in COND_KINDS:
            raise ValueError(f"unknown condition {str(c)[:80]!r}")
        places = c.get("places") or []
        if not isinstance(places, list) or not all(isinstance(p, str) for p in places) or len(places) > 4:
            raise ValueError("condition 'places' must be a list of at most 4 place names")
        kind, pct = c["kind"], c.get("battery_pct")
        if kind == "closest_of":
            if not 2 <= len(places) <= 4:
                raise ValueError("closest_of needs 2 to 4 places")
            pct = None
        else:
            if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not math.isfinite(pct) or not 5 <= pct <= 95:
                raise ValueError("battery_pct must be a number from 5 to 95")
            pct = float(pct)
            if kind == "skip_if_battery_below" and len(places) != 1:
                raise ValueError("skip_if_battery_below needs exactly one place")
            if kind == "stay_if_battery_below":
                places = []
        cond = Condition(kind, tuple(places), pct)
        if cond not in out:
            out.append(cond)
    return tuple(out)


def validate_spec(spec: MissionSpec, campus, *, now_min: float, horizon_min: float,
                  position=None, visited: frozenset[str] = frozenset()) -> SpecCheck:
    """Names must be real and unambiguous; targets must be legal places to fly to.

    Geometry checks (not just names): a target inside the jail buffer or inside an
    active avoid-zone is an error; an avoid-zone covering the base is a warning (the
    drone will recover to another landing pad); one covering the drone's current
    position is a warning (the controller will leave it by the shortest exit).
    """
    from .campus import norm
    from .flight import AVOID_BUFFER, Airspace

    errors, warnings, question = [], [], None
    remaining = [t for t in spec.targets if _key(t, norm) not in visited]
    if not spec.targets:
        errors.append("no inspection targets")
    tkeys = {_key(t, norm) for t in spec.targets}
    for cond in spec.conditions:
        for p in cond.places:
            if _key(p, norm) not in tkeys:
                errors.append(f"condition place {p!r} must also be one of the targets")
    for name in [*spec.targets, *(r.zone for r in spec.avoid)]:
        k = _key(name, norm)
        if k is None:
            errors.append(f"{name!r} is not a valid place name")
        elif k in campus.ambiguous:
            errors.append(f"{name!r} is ambiguous: several places on the map share that name")
            question = f"Which {name} do you mean? Name a nearby building to pick one."
        elif campus.find(name) is None:
            errors.append(f"{name!r} is not a place on the map")
    if len({_key(t, norm) for t in spec.targets}) != len(spec.targets):
        errors.append("a target is listed twice")
    if errors:
        return SpecCheck(False, errors, warnings, None, question)

    air = Airspace.for_spec(campus, spec, now_min, now_min + horizon_min)
    for t in remaining:
        why = air.violation(campus.find(t).center)
        if why:
            errors.append(f"target {t!r} cannot be inspected: it is inside {why}")
    for r in spec.avoid:
        if not r.active_between(now_min, now_min + horizon_min):
            continue
        poly = list(campus.find(r.zone).polygon)
        from .geometry import distance_to_polygon
        if distance_to_polygon(campus.base, poly) < AVOID_BUFFER:
            warnings.append(f"avoid-zone {r.zone!r} covers the base: the drone will land at another pad")
        if position is not None and distance_to_polygon(position, poly) < AVOID_BUFFER:
            warnings.append(f"the drone is inside avoid-zone {r.zone!r}: it will leave by the shortest exit")
    return SpecCheck(not errors, errors, warnings, spec if not errors else None, question)


def _key(name, norm):
    try:
        return norm(name)
    except ValueError:
        return None
