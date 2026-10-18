"""Flight physics, the route planner, the safety shield and the simulator for the campus map.

Physics (stated simplifications): constant airspeed, constant electrical power in
cruise, wind as a constant vector per flight, fixed altitude above all buildings.
Ground speed along a leg follows from the wind triangle.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .campus import Campus, norm
from .geometry import Point, dist, distance_to_polygon, path_length, sample_segment
from .spec import MissionSpec

AIRSPEED = 10.0          # m/s
POWER_W = 180.0          # cruise power draw
RESERVE_FRACTION = 0.20  # battery that must remain on landing (preflight rule)
CRASH_RESERVE_WH = 0.0   # below this the drone falls out of the sky
VISIT_RADIUS = 15.0      # passing within this distance of a target's centre counts as inspecting it
CELL = 10.0              # planner grid resolution, metres
PLAN_MARGIN = 10.0       # planner keeps this much extra clearance beyond each rule's buffer
AVOID_BUFFER = 10.0      # clearance the shield requires around mission avoid-zones
SAMPLE = 2.0             # shield checks route segments every SAMPLE metres


@dataclass(frozen=True)
class Wind:
    speed: float       # m/s
    from_deg: float    # meteorological: direction the wind blows FROM, clockwise from north

    @property
    def vector(self) -> tuple[float, float]:
        to = math.radians(self.from_deg + 180.0)
        return (self.speed * math.sin(to), self.speed * math.cos(to))


def ground_speed(a: Point, b: Point, wind: Wind) -> float:
    """Ground speed flying a->b with constant airspeed; 0 if the wind makes it impossible."""
    L = dist(a, b)
    if L == 0:
        return AIRSPEED
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    wx, wy = wind.vector
    w_par = wx * ux + wy * uy
    w_perp2 = (wx * wx + wy * wy) - w_par * w_par
    if w_perp2 >= AIRSPEED * AIRSPEED:
        return 0.0
    g = w_par + math.sqrt(AIRSPEED * AIRSPEED - w_perp2)
    return max(g, 0.0)


def route_energy_wh(route: list[Point], wind: Wind) -> float:
    """Energy to fly the route; inf if any leg cannot be flown against the wind."""
    seconds = 0.0
    for a, b in zip(route, route[1:]):
        g = ground_speed(a, b, wind)
        if g <= 0.1:
            return math.inf
        seconds += dist(a, b) / g
    return POWER_W * seconds / 3600.0


# --- forbidden-space queries -----------------------------------------------------------------

@dataclass
class Airspace:
    """Everything the drone must stay out of for one mission, with the buffers to use."""
    campus: Campus
    avoid_polys: list[tuple[str, list[Point], float]] = field(default_factory=list)  # name, poly, buffer

    @classmethod
    def for_spec(cls, campus: Campus, spec: MissionSpec | None, t0: float, t1: float) -> "Airspace":
        polys = []
        if spec is not None:
            for r in spec.avoid:
                p = campus.find(r.zone)
                if p is not None and r.active_between(t0, t1):
                    polys.append((p.name, list(p.polygon), AVOID_BUFFER))
        return cls(campus, polys)

    def violation(self, p: Point, extra: float = 0.0) -> str | None:
        """Why p is forbidden (with `extra` clearance added to every buffer), or None."""
        if not self.campus.in_fence(p) if extra == 0 else not _in_fence(self.campus, p, extra):
            return "geofence"
        for name, (poly, buf) in self.campus.permanent_no_fly.items():
            if distance_to_polygon(p, poly) < buf + extra:
                return f"no-fly: {name}"
        for name, poly, buf in self.avoid_polys:
            if distance_to_polygon(p, poly) < buf + extra:
                return f"avoid: {name}"
        return None


def _in_fence(c: Campus, p: Point, extra: float) -> bool:
    x0, y0, x1, y1 = c.fence
    return x0 + extra <= p[0] <= x1 - extra and y0 + extra <= p[1] <= y1 - extra


# --- planner ---------------------------------------------------------------------------------------

class GridPlanner:
    """8-connected A* on a 10 m grid, then shortcutting with exact line-of-sight checks."""

    def __init__(self, airspace: Airspace):
        self.air = airspace
        x0, y0, x1, y1 = airspace.campus.fence
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = int((x1 - x0) // CELL) + 1, int((y1 - y0) // CELL) + 1
        self.blocked = [[airspace.violation(self.center((i, j)), PLAN_MARGIN) is not None
                         for j in range(self.ny)] for i in range(self.nx)]

    def center(self, c: tuple[int, int]) -> Point:
        return (self.x0 + c[0] * CELL, self.y0 + c[1] * CELL)

    def cell_of(self, p: Point) -> tuple[int, int]:
        return (min(self.nx - 1, max(0, round((p[0] - self.x0) / CELL))),
                min(self.ny - 1, max(0, round((p[1] - self.y0) / CELL))))

    def clear(self, a: Point, b: Point) -> bool:
        return all(self.air.violation(p, PLAN_MARGIN / 2) is None for p in sample_segment(a, b, SAMPLE))

    def leg(self, a: Point, b: Point) -> list[Point] | None:
        if self.air.violation(a) or self.air.violation(b):
            return None
        if self.clear(a, b):
            return [a, b]
        s, g = self.cell_of(a), self.cell_of(b)
        if self.blocked[s[0]][s[1]] or self.blocked[g[0]][g[1]]:
            s, g = self._nearest_free(s), self._nearest_free(g)
            if s is None or g is None:
                return None
        cells = self._astar(s, g)
        if cells is None:
            return None
        pts = [a] + [self.center(c) for c in cells] + [b]
        return self._shortcut(pts)

    def _nearest_free(self, c):
        for r in range(1, 4):
            for i in range(c[0] - r, c[0] + r + 1):
                for j in range(c[1] - r, c[1] + r + 1):
                    if 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[i][j]:
                        return (i, j)
        return None

    def _astar(self, s, g):
        def h(c):
            return math.hypot(c[0] - g[0], c[1] - g[1])
        openh = [(h(s), 0.0, s)]
        came, best = {}, {s: 0.0}
        while openh:
            _, cost, cur = heapq.heappop(openh)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if cost > best.get(cur, math.inf):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    n = (cur[0] + dx, cur[1] + dy)
                    if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or self.blocked[n[0]][n[1]]:
                        continue
                    if dx and dy and (self.blocked[cur[0] + dx][cur[1]] or self.blocked[cur[0]][cur[1] + dy]):
                        continue  # no corner cutting
                    nc = cost + (1.4142135 if dx and dy else 1.0)
                    if nc < best.get(n, math.inf):
                        best[n] = nc
                        came[n] = cur
                        heapq.heappush(openh, (nc + h(n), nc, n))
        return None

    def _shortcut(self, pts: list[Point]) -> list[Point] | None:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.clear(pts[i], pts[j]):
                j -= 1
            if not self.clear(pts[i], pts[j]):
                return None
            out.append(pts[j])
            i = j
        return out


def plan_route(campus: Campus, spec: MissionSpec, start_min: float) -> tuple[list[Point] | None, str]:
    """Base -> targets in order (-> base). Avoid rules active during the next hour are blocked."""
    air = Airspace.for_spec(campus, spec, start_min, start_min + 60)
    planner = GridPlanner(air)
    stops = [campus.base] + [campus.find(t).center for t in spec.targets]
    if spec.return_to_base:
        stops.append(campus.base)
    route: list[Point] = [stops[0]]
    for a, b in zip(stops, stops[1:]):
        leg = planner.leg(a, b)
        if leg is None:
            name = next((t for t in spec.targets if campus.find(t).center == b), "base")
            return None, f"no safe path to {name}"
        route += leg[1:]
    return route, f"route {path_length(route):.0f} m with {len(route)} waypoints"


# --- shield ------------------------------------------------------------------------------------------

@dataclass
class ShieldResult:
    ok: bool
    reasons: list[str]
    energy_wh: float


def preflight_check(route: object, campus: Campus, spec: MissionSpec | None, start_min: float,
                    forecast: Wind, battery_wh: float) -> ShieldResult:
    """Check a whole route before take-off. Permanent rules and the geofence always apply;
    the spec's avoid-zones and targets apply when a spec is given."""
    reasons: list[str] = []
    pts = _coerce(route)
    if pts is None:
        return ShieldResult(False, ["route is not a list of [x, y] points"], math.inf)
    if len(pts) < 2:
        return ShieldResult(False, ["route has fewer than two points"], math.inf)
    if dist(pts[0], campus.base) > VISIT_RADIUS:
        reasons.append("route does not start at base")
    air = Airspace.for_spec(campus, spec, start_min, start_min + 60)
    for a, b in zip(pts, pts[1:]):
        for p in sample_segment(a, b, SAMPLE):
            why = air.violation(p)
            if why:
                reasons.append(f"enters {why} near ({p[0]:.0f}, {p[1]:.0f})")
                break
        if len(reasons) > 5:
            break
    if spec is not None:
        if not visits_in_order(pts, [campus.find(t).center for t in spec.targets]):
            reasons.append("does not visit every target in the requested order")
        if spec.return_to_base and dist(pts[-1], campus.base) > VISIT_RADIUS:
            reasons.append("does not return to base")
    e = route_energy_wh(pts, forecast)
    if e > battery_wh * (1 - RESERVE_FRACTION):
        reasons.append(f"needs {e:.1f} Wh but only {battery_wh * (1 - RESERVE_FRACTION):.1f} Wh may be used "
                       f"({RESERVE_FRACTION:.0%} reserve)")
    return ShieldResult(not reasons, list(dict.fromkeys(reasons)), e)


def _coerce(route: object) -> list[Point] | None:
    if not isinstance(route, (list, tuple)) or len(route) > 5000:
        return None
    out = []
    for p in route:
        if not (isinstance(p, (list, tuple)) and len(p) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)):
            return None
        out.append((float(p[0]), float(p[1])))
    return out


def visits_in_order(track: list[Point], targets: list[Point], radius: float = VISIT_RADIUS) -> bool:
    """Does the track pass within `radius` of each target, in the given order?"""
    if not targets:
        return True
    k = 0
    for a, b in zip(track, track[1:] or track):
        for p in sample_segment(a, b, SAMPLE):
            while k < len(targets) and dist(p, targets[k]) <= radius:
                k += 1
            if k == len(targets):
                return True
    return k == len(targets)


# --- simulator with runtime monitor ---------------------------------------------------------------

@dataclass
class FlightLog:
    outcome: str                    # completed | returned_early | crashed_battery | refused
    track: list[Point]
    times: list[float]              # seconds after take-off, one per track point
    seconds: float
    energy_used_wh: float
    battery_left_wh: float
    events: list[str]


def simulate(route: list[Point], campus: Campus, actual: Wind, battery_wh: float, *,
             monitor: bool, start_min: float, dt: float = 1.0) -> FlightLog:
    """Fly the route in the real (actual) wind. With `monitor` on, a runtime shield checks every
    5 s whether the battery still covers the safest way home, and turns back if it does not."""
    home_planner = GridPlanner(Airspace(campus)) if monitor else None
    track = [route[0]]
    times = [0.0]
    energy = battery_wh
    t = 0.0
    events: list[str] = []
    legs = list(zip(route, route[1:]))
    li, pos = 0, route[0]
    returning = False
    while li < len(legs):
        a, b = legs[li]
        g = ground_speed(pos, b, actual)
        if g <= 0.1:
            events.append(f"t={t:.0f}s cannot make headway against the wind")
            return FlightLog("crashed_battery", track, times, t, battery_wh - energy, energy, events)
        step = g * dt
        d = dist(pos, b)
        if d <= step:
            pos, li = b, li + 1
            used_t = d / g
        else:
            pos = (pos[0] + (b[0] - pos[0]) * step / d, pos[1] + (b[1] - pos[1]) * step / d)
            used_t = dt
        t += used_t
        energy -= POWER_W * used_t / 3600.0
        track.append(pos)
        times.append(t)
        if energy <= CRASH_RESERVE_WH:
            events.append(f"t={t:.0f}s battery empty at ({pos[0]:.0f}, {pos[1]:.0f})")
            return FlightLog("crashed_battery", track, times, t, battery_wh, 0.0, events)
        if monitor and not returning and int(t) % 5 == 0 and used_t == dt:
            home = home_planner.leg(pos, campus.base)
            need = route_energy_wh(home, actual) if home else math.inf
            if energy - need * 1.15 < battery_wh * 0.05:
                events.append(f"t={t:.0f}s runtime shield: {energy:.1f} Wh left, {need:.1f} Wh needed "
                              f"to get home -> returning to base")
                legs = list(zip(home, home[1:]))
                li, returning = 0, True
    outcome = "returned_early" if returning else "completed"
    return FlightLog(outcome, track, times, t, battery_wh - energy, energy, events)


def mission_start_from_text(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def place_center(campus: Campus, name: str) -> Point:
    p = campus.places[norm(name)]
    return p.center
