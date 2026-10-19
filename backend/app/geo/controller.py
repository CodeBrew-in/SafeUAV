"""Deterministic mission controller: the state machine that actually flies the drone.

No LLM runs inside this loop. Orders and mid-flight changes arrive as already-parsed
MissionSpecs (or the keyword commands RETURN / LAND) through `submit`, are queued,
coalesced and applied at the start of a tick. A new route is swapped in atomically and
only after the preflight shield passes it.

States
  PREFLIGHT              on the ground at base, no accepted order yet
  CRUISE                 flying the accepted mission route
  ESCAPE                 leaving a zone that became forbidden around the drone
  TO_PAD                 the mission cannot continue safely: flying to the nearest usable pad
  LANDED_WAIT            on a pad, idle, waiting for a new order (bounded by MAX_WAIT_MIN)
  EMERGENCY              no pad reachable: landing at the nearest legal spot
  DONE                   mission finished and landed
  GROUNDED_NEEDS_PICKUP  landed somewhere it cannot or may not take off from
  CRASHED                battery ran out in the air (only possible with the monitor off)

Key rule: when a change makes the current mission impossible, the old route is NOT kept
(the new rules may forbid it). The drone obeys the new rules, escapes if it is inside a
newly forbidden zone, lands on the nearest usable pad and waits for a feasible order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .campus import Campus, norm
from .flight import (CRUISE_W, FLOOR_FRACTION, HOVER_W, IDLE_W, LANDING_WH, MAX_WAIT_MIN,
                     MEASURED_MARGIN, TAKEOFF_WH, VISIT_RADIUS, Airspace, Wind, escape_path,
                     flyable_weather, ground_speed, horizon_min, plan_through, preflight_check,
                     route_energy_wh, route_to_nearest_pad)
from .geometry import Point, dist
from .spec import MissionSpec, validate_spec

FLYING = {"CRUISE", "ESCAPE", "TO_PAD", "EMERGENCY"}
GROUNDED = {"PREFLIGHT", "LANDED_WAIT", "DONE", "GROUNDED_NEEDS_PICKUP", "CRASHED"}
MONITOR_EVERY_S = 5
MAX_REPLANS_PER_MIN = 6


@dataclass(frozen=True)
class Command:
    kind: str                      # "order" (a full spec), "return", "land"
    spec: MissionSpec | None = None
    text: str = ""                 # the operator's words, for the audit log


@dataclass
class Plan:
    route: list[Point]
    purpose: str                   # mission | return | to_pad | emergency
    end_name: str                  # where it lands
    escape_points: int = 0         # first n route points are the escape leg
    version: int = 0


class Controller:
    def __init__(self, campus: Campus, *, capacity_wh: float, forecast: Wind, actual: Wind,
                 clock_min: float, monitor: bool = True, dt: float = 1.0):
        self.c = campus
        self.capacity = capacity_wh
        self.energy = capacity_wh
        self.forecast, self.actual = forecast, actual
        self.clock0 = clock_min
        self.monitor = monitor
        self.dt = dt
        self.t = 0.0
        self.pos: Point = campus.base
        self.state = "PREFLIGHT"
        self.spec: MissionSpec | None = None
        self.visited: list[str] = []          # normalised names, in the order inspected
        self.plan: Plan | None = None
        self.ri = 1                           # index of the next route point
        self.version = 0
        self.inbox: list[Command] = []
        self.replan_times: list[float] = []
        self.landed_at: float | None = None
        self.no_headway_s = 0.0
        self.audit: list[dict] = []
        self.track: list[tuple[float, float, float, str]] = [(0.0, *self.pos, self.state)]
        self._log("start", "drone on the ground at base", capacity_wh=capacity_wh,
                  forecast=[forecast.speed, forecast.from_deg], actual=[actual.speed, actual.from_deg],
                  clock=self.clock(), monitor=monitor)

    # --- helpers -------------------------------------------------------------------------------------
    def clock(self) -> float:
        return self.clock0 + self.t / 60.0

    def hhmm(self) -> str:
        m = int(self.clock()) % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    def floor_wh(self) -> float:
        return FLOOR_FRACTION * self.capacity

    def airspace(self, spec: MissionSpec | None) -> Airspace:
        return Airspace.for_spec(self.c, spec, self.clock(), self.clock() + horizon_min(self.capacity))

    def _log(self, kind: str, msg: str, **data) -> None:
        self.audit.append({"t": round(self.t, 1), "clock": self.hhmm() if hasattr(self, "clock0") else None,
                           "state": self.state, "kind": kind, "msg": msg, "data": data})

    def _set_state(self, s: str, why: str) -> None:
        if s != self.state:
            old = self.state
            self.state = s
            self._log("state", f"{old} -> {s}: {why}")

    def remaining_targets(self, spec: MissionSpec) -> list[str]:
        return [t for t in spec.targets if norm(t) not in self.visited]

    # --- public API ----------------------------------------------------------------------------------
    def submit(self, cmd: Command) -> None:
        """Queue an operator command; it is applied at the start of the next tick."""
        self.inbox.append(cmd)
        self._log("received", cmd.text or cmd.kind, command=cmd.kind,
                  spec=cmd.spec.to_json() if cmd.spec else None)

    def run(self, max_s: float = 3 * 3600) -> "Controller":
        while self.state not in ("DONE", "GROUNDED_NEEDS_PICKUP", "CRASHED") and self.t < max_s:
            if self.state == "PREFLIGHT" and not self.inbox:
                break
            self.step()
        return self

    def step(self) -> None:
        self._process_inbox()
        if self.state in FLYING:
            self._fly()
        elif self.state == "LANDED_WAIT":
            self._drain(IDLE_W, self.dt)
            if self.landed_at is not None and self.t - self.landed_at > MAX_WAIT_MIN * 60:
                self._set_state("GROUNDED_NEEDS_PICKUP", f"no feasible order within {MAX_WAIT_MIN} min")
        self.t += self.dt
        self.track.append((self.t, *self.pos, self.state))

    # --- command handling --------------------------------------------------------------------------------
    def _process_inbox(self) -> None:
        if not self.inbox or self.state in ("DONE", "GROUNDED_NEEDS_PICKUP", "CRASHED"):
            self.inbox.clear()
            return
        # rate limit: at most MAX_REPLANS_PER_MIN applied per minute; the rest wait (coalesced)
        self.replan_times = [x for x in self.replan_times if self.t - x < 60]
        if len(self.replan_times) >= MAX_REPLANS_PER_MIN:
            return
        cmds, self.inbox = self.inbox, []
        # coalesce: RETURN/LAND win over orders; otherwise only the latest order matters
        cmd = next((c for c in reversed(cmds) if c.kind in ("land", "return")), cmds[-1])
        if len(cmds) > 1:
            self._log("coalesced", f"{len(cmds)} queued commands -> applying the latest '{cmd.kind}'")
        self.replan_times.append(self.t)
        if cmd.kind in ("return", "land") and self.state not in FLYING:
            self._log("ignored", f"'{cmd.kind}' ignored: the drone is already on the ground")
            return
        if cmd.kind == "order":
            self._apply_order(cmd.spec)
        elif cmd.kind == "return":
            self._go_home("operator: return now")
        elif cmd.kind == "land":
            self._go_to_pad("operator: land now")

    def _apply_order(self, spec: MissionSpec) -> None:
        airborne = self.state in FLYING
        check = validate_spec(spec, self.c, now_min=self.clock(), horizon_min=horizon_min(self.capacity),
                              position=self.pos, visited=frozenset(self.visited))
        if not check.ok:
            # an invalid message never takes effect: the current (valid) rules and route stay
            self._log("order_rejected", "message not applied: " + "; ".join(check.errors),
                      question=check.question)
            return
        for w in check.warnings:
            self._log("warning", w)
        self.spec = spec  # valid new rules apply from now on, even if the mission becomes impossible
        self._log("replan", "replanning from the current position" if airborne else "planning")
        plan, reasons = self._mission_plan(spec, airborne)
        if plan is not None:
            self._swap(plan, "mission route accepted by the shield")
            if not airborne:
                self._takeoff()
            self._set_state("ESCAPE" if plan.escape_points else "CRUISE", "new route")
            return
        self._log("replan_failed", "mission impossible under the new rules: " + "; ".join(reasons))
        if airborne:
            self._go_to_pad("old route abandoned (it may break the new rules)")
        else:
            self._set_state("LANDED_WAIT" if self.state != "PREFLIGHT" else "PREFLIGHT",
                            "staying on the ground until a feasible order arrives")

    def _mission_plan(self, spec: MissionSpec, airborne: bool) -> tuple[Plan | None, list[str]]:
        air = self.airspace(spec)
        if not flyable_weather(self.forecast):
            return None, [f"wind {self.forecast.speed:.0f} m/s is too strong to fly"]
        if air.violation(self.pos):
            if not airborne:
                return None, [f"take-off blocked: the drone's pad is inside {air.violation(self.pos)}"]
            esc = escape_path(air, self.pos)
            if esc is None:
                return None, ["no safe exit from the forbidden area"]
        else:
            esc = [self.pos]
        remaining = self.remaining_targets(spec)
        stops = [self.c.find(t).center for t in remaining]
        if spec.return_to_base:
            end_name, end = self._home(air)
            if end is None:
                return None, ["no usable landing pad to finish at"]
        else:
            end_name, end = (remaining[-1], stops[-1]) if remaining else ("here", esc[-1])
        tail = plan_through(air, esc[-1], stops + ([end] if spec.return_to_base else []))
        if tail is None:
            return None, ["no safe path to the remaining targets"]
        route = esc[:-1] + tail
        shield = preflight_check(route, air, start=self.pos, energy_wh=self.energy, capacity_wh=self.capacity,
                                 forecast=self.forecast, airborne=airborne, targets=stops, end_at=end,
                                 escape_first=len(esc) > 1)
        if not shield.ok:
            return None, shield.reasons
        return Plan(route, "mission", end_name, len(esc) - 1), []

    def _home(self, air: Airspace) -> tuple[str, Point | None]:
        if air.violation(self.c.base) is None:
            return "base", self.c.base
        pads = [(dist(self.pos, p), n, p) for n, p in self.c.pads.items() if air.violation(p) is None]
        if not pads:
            return "none", None
        _, n, p = min(pads)
        self._log("warning", f"base is inside an avoid-zone: finishing at pad {n} instead")
        return n, p

    def _go_home(self, why: str) -> None:
        air = self.airspace(self.spec)
        name, end = self._home(air)
        if end is not None:
            esc = escape_path(air, self.pos) if self.state in FLYING else [self.pos]
            tail = plan_through(air, esc[-1], [end]) if esc else None
            if tail is not None:
                route = esc[:-1] + tail
                e = route_energy_wh(route, self.actual.scaled(MEASURED_MARGIN)) + LANDING_WH
                if e <= self.energy - self.floor_wh() * 0.5:
                    if self.state not in FLYING:
                        self._takeoff()
                    self._swap(Plan(route, "return", name, len(esc) - 1), why)
                    self._set_state("CRUISE", why)
                    return
        self._go_to_pad(why + "; base not reachable")

    def _go_to_pad(self, why: str) -> None:
        if self.state not in FLYING:
            self._set_state("LANDED_WAIT", why + " (already on the ground)")
            self.landed_at = self.landed_at if self.landed_at is not None else self.t
            return
        air = self.airspace(self.spec)
        found = route_to_nearest_pad(air, self.pos, self.actual.scaled(MEASURED_MARGIN), self.energy,
                                     self.floor_wh() * 0.25)
        if found is not None:
            name, route = found
            self._swap(Plan(route, "to_pad", name), why)
            self._set_state("TO_PAD", f"{why}: landing at pad {name} to wait")
            return
        self._emergency(why + "; no pad reachable")

    def _emergency(self, why: str) -> None:
        air = self.airspace(self.spec)
        esc = escape_path(air, self.pos) or escape_path(Airspace(self.c), self.pos) or [self.pos]
        route = esc if len(esc) > 1 else [self.pos, self.pos]
        self._swap(Plan(route, "emergency", "nearest legal spot"), why)
        self._set_state("EMERGENCY", why)

    def _swap(self, plan: Plan, why: str) -> None:
        self.version += 1
        plan.version = self.version
        self.plan, self.ri = plan, 1
        self._log("route", why, version=self.version, purpose=plan.purpose, to=plan.end_name,
                  points=[[round(x, 1), round(y, 1)] for x, y in plan.route])

    def _takeoff(self) -> None:
        self.energy -= TAKEOFF_WH
        self.landed_at = None
        self._log("takeoff", f"take-off ({TAKEOFF_WH} Wh)")

    # --- flight -----------------------------------------------------------------------------------------
    def _drain(self, watts: float, seconds: float) -> None:
        self.energy -= watts * seconds / 3600.0

    def _fly(self) -> None:
        plan = self.plan
        if self.ri >= len(plan.route):
            return self._arrive()
        target = plan.route[self.ri]
        g = ground_speed(self.pos, target, self.actual)
        if g <= 0.5:
            self._drain(HOVER_W, self.dt)
            self.no_headway_s += self.dt
            if self.no_headway_s >= 10 and plan.purpose != "emergency":
                self._emergency("no headway against the wind")
            self._check_battery()
            return
        self.no_headway_s = 0.0
        step = g * self.dt
        d = dist(self.pos, target)
        old = self.pos
        if d <= step:
            self.pos, self.ri = target, self.ri + 1
        else:
            self.pos = (self.pos[0] + (target[0] - self.pos[0]) * step / d,
                        self.pos[1] + (target[1] - self.pos[1]) * step / d)
        self._drain(CRUISE_W, self.dt)
        self._mark_visits(old, self.pos)
        if self.state == "ESCAPE" and self.ri > plan.escape_points:
            self._set_state("CRUISE", "left the forbidden zone")
        if self._check_battery():
            return
        if self.monitor and int(self.t) % MONITOR_EVERY_S == 0:
            self._monitor()
        if self.ri >= len(plan.route):
            self._arrive()

    def _mark_visits(self, a: Point, b: Point) -> None:
        if self.spec is None or self.plan.purpose != "mission":
            return
        rem = self.remaining_targets(self.spec)
        if not rem:
            return
        nxt = rem[0]
        p = self.c.find(nxt).center
        from .geometry import point_segment_distance
        if point_segment_distance(p, a, b) <= VISIT_RADIUS:
            self.visited.append(norm(nxt))
            self._log("inspected", f"inspected {nxt}")

    def _check_battery(self) -> bool:
        if self.energy <= 0:
            self.energy = 0.0
            self._set_state("CRASHED", "battery empty in the air")
            return True
        return False

    def _monitor(self) -> None:
        """Runtime shield: can we still finish this route, using the wind we actually measure?"""
        if self.plan.purpose == "emergency":
            return
        measured = self.actual.scaled(MEASURED_MARGIN)
        rest = [self.pos] + self.plan.route[self.ri:]
        need = route_energy_wh(rest, measured) + LANDING_WH
        if self.energy - need >= self.floor_wh() * 0.5:
            return
        self._log("monitor", f"runtime shield: {self.energy:.1f} Wh left, {need:.1f} Wh needed to finish "
                             f"this route in the measured wind")
        if self.plan.purpose in ("mission", "return"):
            self._go_to_pad("battery too low to finish")
        else:
            self._emergency("battery too low to reach the pad")

    def _arrive(self) -> None:
        purpose = self.plan.purpose
        self.energy -= LANDING_WH
        if self._check_battery():
            return
        self._log("landed", f"landed at {self.plan.end_name}")
        if purpose == "mission":
            if self.spec is not None and self.remaining_targets(self.spec):
                self._set_state("LANDED_WAIT", "landed before all targets were inspected")
                self.landed_at = self.t
            else:
                self._set_state("DONE", "mission complete")
        elif purpose == "return":
            self._set_state("DONE", "returned on operator command")
        elif purpose == "to_pad":
            self.landed_at = self.t
            self._set_state("LANDED_WAIT", f"waiting on pad {self.plan.end_name} for a feasible order")
        else:
            self._set_state("GROUNDED_NEEDS_PICKUP", "emergency landing")

    # --- reporting ---------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "t": round(self.t, 1), "clock": self.hhmm(), "state": self.state,
            "pos": [round(self.pos[0], 1), round(self.pos[1], 1)],
            "battery_wh": round(self.energy, 2), "battery_pct": round(100 * self.energy / self.capacity, 1),
            "spec": self.spec.to_json() if self.spec else None,
            "visited": list(self.visited),
            "route": [[round(x, 1), round(y, 1)] for x, y in self.plan.route] if self.plan else [],
            "route_index": self.ri, "version": self.version,
        }


def run_script(campus: Campus, *, capacity_wh: float, forecast: Wind, actual: Wind, clock_min: float,
               initial: MissionSpec, changes: list[tuple[float, Command]] = (), monitor: bool = True,
               max_s: float = 3 * 3600) -> Controller:
    """Deterministic replay: same inputs -> same audit log."""
    ctl = Controller(campus, capacity_wh=capacity_wh, forecast=forecast, actual=actual,
                     clock_min=clock_min, monitor=monitor)
    ctl.submit(Command("order", initial, "initial order"))
    pending = sorted(changes, key=lambda x: x[0])
    while ctl.t < max_s and ctl.state not in ("DONE", "GROUNDED_NEEDS_PICKUP", "CRASHED"):
        while pending and pending[0][0] <= ctl.t:
            ctl.submit(pending.pop(0)[1])
        if ctl.state in ("PREFLIGHT",) and not ctl.inbox and not pending:
            break
        if ctl.state == "LANDED_WAIT" and not pending and not ctl.inbox:
            # nothing more will arrive: skip ahead through the wait (idle drain applies)
            ctl.step()
            continue
        ctl.step()
    return ctl
