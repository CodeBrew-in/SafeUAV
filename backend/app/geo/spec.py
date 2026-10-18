"""Mission specification: what the operator asked for, in checkable form.

A spec is produced either by hand (ground truth / classic form) or by an LLM
from a plain-English order. It can only ADD restrictions (zones to avoid) and
choose targets; it has no way to remove a permanent no-fly zone or the geofence.
`validate_spec` checks every name against the real map before anything is planned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .campus import Campus, norm

TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


@dataclass(frozen=True)
class AvoidRule:
    zone: str
    start_min: int | None = None  # minutes after midnight; None = always
    end_min: int | None = None

    def active_between(self, t0: float, t1: float) -> bool:
        """Is the rule in force at any time in [t0, t1] (minutes after midnight)?"""
        if self.start_min is None or self.end_min is None:
            return True
        return self.start_min <= t1 and t0 <= self.end_min


@dataclass(frozen=True)
class MissionSpec:
    targets: tuple[str, ...]           # inspection targets, in order
    avoid: tuple[AvoidRule, ...] = ()
    return_to_base: bool = True


@dataclass
class SpecCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    spec: MissionSpec | None = None


def parse_time(s) -> int | None:
    if s in (None, "", "null"):
        return None
    m = TIME_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"bad time {s!r}, expected HH:MM")
    return int(m.group(1)) * 60 + int(m.group(2))


def spec_from_json(obj: object) -> MissionSpec:
    """Build a spec from model JSON. Raises ValueError with a readable reason."""
    if not isinstance(obj, dict):
        raise ValueError(f"expected an object, got {type(obj).__name__}")
    targets = obj.get("targets")
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError("'targets' must be a list of place names")
    avoid_raw = obj.get("avoid", [])
    if not isinstance(avoid_raw, list):
        raise ValueError("'avoid' must be a list")
    rules = []
    for a in avoid_raw:
        if not isinstance(a, dict) or not isinstance(a.get("zone"), str):
            raise ValueError(f"bad avoid rule {a!r}"[:200])
        start, end = parse_time(a.get("from")), parse_time(a.get("to"))
        if (start is None) != (end is None):
            raise ValueError(f"avoid rule for {a['zone']!r} needs both 'from' and 'to', or neither")
        rules.append(AvoidRule(a["zone"], start, end))
    rtb = obj.get("return_to_base", True)
    if not isinstance(rtb, bool):
        raise ValueError("'return_to_base' must be true or false")
    return MissionSpec(tuple(targets), tuple(rules), rtb)


def validate_spec(spec: MissionSpec, campus: Campus) -> SpecCheck:
    """Every name must be a real, unambiguous place; targets must not be forbidden."""
    errors = []
    if not spec.targets:
        errors.append("no inspection targets")
    for name in [*spec.targets, *(r.zone for r in spec.avoid)]:
        if norm(name) in campus.ambiguous:
            errors.append(f"{name!r} is ambiguous on this map (several places share the name)")
        elif campus.find(name) is None:
            errors.append(f"{name!r} is not a place on the map")
    for t in spec.targets:
        if any(norm(t) == norm(z) for z in campus.permanent_no_fly):
            errors.append(f"target {t!r} is inside a permanent no-fly zone")
        if any(norm(t) == norm(r.zone) for r in spec.avoid):
            errors.append(f"target {t!r} is also a zone to avoid")
    if len({norm(t) for t in spec.targets}) != len(spec.targets):
        errors.append("a target is listed twice")
    return SpecCheck(not errors, errors, spec if not errors else None)
