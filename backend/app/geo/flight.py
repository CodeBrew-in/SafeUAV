"""Flight physics, airspace, route planner, escape, landing pads and the preflight shield.

Simulation assumptions (stated, not real-drone validated): fixed cruise altitude above
all buildings, constant airspeed, constant power per flight phase, wind as a constant
vector per flight. Ground speed along a leg follows from the wind triangle.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from .campus import Campus
from .geometry import Point, dist, distance_to_polygon, path_length, sample_segment
from .spec import MissionSpec

# --- physics and energy ------------------------------------------------------------------------
AIRSPEED = 10.0        # m/s
CRUISE_W = 180.0       # electrical power in cruise
HOVER_W = 200.0        # hovering in place (waiting for a decision)
IDLE_W = 8.0           # landed, avionics on, waiting for orders
TAKEOFF_WH = 2.0       # climb to cruise altitude
LANDING_WH = 1.2       # descent and touchdown
FLOOR_FRACTION = 0.10  # never plan to land with less than this share of a full battery
FORECAST_MARGIN = 1.25 # planning assumes wind this much stronger than forecast
MEASURED_MARGIN = 1.10 # the in-flight monitor uses measured wind plus this margin
MAX_WAIT_MIN = 30      # longest a drone sits on a pad before it must be collected

# --- geometry rules ----------------------------------------------------------------------------
VISIT_RADIUS = 15.0    # passing within this distance of a target's reference point inspects it
CELL = 10.0            # planner grid resolution
PLAN_MARGIN = 10.0     # extra clearance the planner keeps away from route end points
AVOID_BUFFER = 10.0    # clearance the shield requires around mission avoid-zones
SAMPLE = 2.0           # shield checks route segments every SAMPLE metres
MAX_COORD = 100_000.0  # any coordinate beyond this is rejected before geometry runs
MAX_LEG = 5_000.0      # longest straight leg accepted


@dataclass(frozen=True)
class Wind:
    speed: float       # m/s
    from_deg: float    # meteorological: direction the wind blows FROM, clockwise from north

    @property
    def vector(self) -> tuple[float, float]:
        to = math.radians(self.from_deg + 180.0)
        return (self.speed * math.sin(to), self.speed * math.cos(to))

    def scaled(self, k: float) -> "Wind":
        return Wind(self.speed * k, self.from_deg)


def flyable_weather(wind: Wind) -> bool:
    return wind.speed < 0.8 * AIRSPEED


def ground_speed(a: Point, b: Point, wind: Wind) -> float:
    """Ground speed flying a->b at constant airspeed; 0 if the wind makes it impossible."""
    L = dist(a, b)
    if L == 0:
        return AIRSPEED
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    wx, wy = wind.vector
    w_par = wx * ux + wy * uy
    w_perp2 = max(0.0, (wx * wx + wy * wy) - w_par * w_par)
    if w_perp2 >= AIRSPEED * AIRSPEED:
        return 0.0
    return max(w_par + math.sqrt(AIRSPEED * AIRSPEED - w_perp2), 0.0)


def route_energy_wh(route: list[Point], wind: Wind) -> float:
    """Cruise energy to fly the route; inf if a leg cannot be flown against the wind."""
    seconds = 0.0
    for a, b in zip(route, route[1:]):
        g = ground_speed(a, b, wind)
        if g <= 0.5:
            return math.inf
        seconds += dist(a, b) / g
    return CRUISE_W * seconds / 3600.0


def horizon_min(battery_wh: float) -> float:
    """Longest a full battery can keep the drone airborne, in minutes: the planning horizon."""
    return battery_wh / CRUISE_W * 60.0


# --- airspace ---------------------------------------------------------------------------------------

@dataclass
class Airspace:
    """What the drone must stay out of. Permanent rules and the geofence always apply;
    `avoid_polys` are the mission's avoid-zones that are in force over the planning window."""
    campus: Campus
    avoid_polys: list[tuple[str, list[Point], float]] = field(default_factory=list)

    @classmethod
    def for_spec(cls, campus: Campus, spec: MissionSpec | None, t0: float, t1: float) -> "Airspace":
        polys = []
        if spec is not None:
            for r in spec.avoid:
                p = campus.find(r.zone)
                if p is not None and r.active_between(t0, t1):
                    polys.append((p.name, list(p.polygon), AVOID_BUFFER))
        return cls(campus, polys)

    def key(self) -> str:
        """Cache key from the GEOMETRY that decides what is blocked (fence, permanent zones, avoid zones),
        never from names or object ids: two different shapes with the same name must not share a grid."""
        h = hashlib.sha256()
        h.update(repr([round(v, 3) for v in self.campus.fence]).encode())
        for name, (poly, buf) in sorted(self.campus.permanent_no_fly.items()):
            h.update(repr((name, buf, [(round(x, 3), round(y, 3)) for x, y in poly])).encode())
        for name, poly, buf in sorted(self.avoid_polys, key=lambda a: a[0]):
            h.update(repr((name, buf, [(round(x, 3), round(y, 3)) for x, y in poly])).encode())
        return h.hexdigest()

    def violation(self, p: Point, extra: float = 0.0, *, mission: bool = True) -> str | None:
        """Why p is forbidden (each buffer grown by `extra`), or None.
        mission=False checks only the permanent rules and the geofence."""
        x0, y0, x1, y1 = self.campus.fence
        if not (x0 + extra <= p[0] <= x1 - extra and y0 + extra <= p[1] <= y1 - extra):
            return "geofence"
        for name, (poly, buf) in self.campus.permanent_no_fly.items():
            if distance_to_polygon(p, poly) < buf + extra:
                return f"no-fly zone {name}"
        if mission:
            for name, poly, buf in self.avoid_polys:
                if distance_to_polygon(p, poly) < buf + extra:
                    return f"avoid-zone {name}"
        return None

    def segment_violation(self, a: Point, b: Point, *, mission: bool = True) -> str | None:
        for p in sample_segment(a, b, SAMPLE):
            why = self.violation(p, mission=mission)
            if why:
                return why
        return None


# --- planner -------------------------------------------------------------------------------------------

_GRID_CACHE: "OrderedDict[str, list[list[bool]]]" = OrderedDict()  # LRU, shared by the sim and API threads
_GRID_CACHE_MAX = 32
_GRID_LOCK = threading.Lock()


class GridPlanner:
    """8-connected A* on a 10 m grid, then shortcutting with exact line-of-sight checks.

    Interior points of a leg keep PLAN_MARGIN/2 extra clearance; points within
    2*PLAN_MARGIN of the leg's ends only need the shield's own buffer, so a start or
    goal that sits just outside a zone (where the shield allows it) is still reachable.
    """

    def __init__(self, airspace: Airspace):
        self.air = airspace
        x0, y0, x1, y1 = airspace.campus.fence
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = int((x1 - x0) // CELL) + 1, int((y1 - y0) // CELL) + 1
        key = airspace.key()
        with _GRID_LOCK:
            grid = _GRID_CACHE.get(key)
            if grid is not None:
                _GRID_CACHE.move_to_end(key)
        if grid is None:  # built outside the lock; a duplicate build is harmless
            grid = [[airspace.violation(self.center((i, j)), PLAN_MARGIN) is not None
                     for j in range(self.ny)] for i in range(self.nx)]
            with _GRID_LOCK:
                _GRID_CACHE[key] = grid
                while len(_GRID_CACHE) > _GRID_CACHE_MAX:
                    _GRID_CACHE.popitem(last=False)
        self.blocked = grid

    def center(self, c: tuple[int, int]) -> Point:
        return (self.x0 + c[0] * CELL, self.y0 + c[1] * CELL)

    def cell_of(self, p: Point) -> tuple[int, int]:
        return (min(self.nx - 1, max(0, round((p[0] - self.x0) / CELL))),
                min(self.ny - 1, max(0, round((p[1] - self.y0) / CELL))))

    def clear(self, a: Point, b: Point, ends: tuple[Point, ...] = ()) -> bool:
        for p in sample_segment(a, b, SAMPLE):
            near_end = any(dist(p, e) <= 2 * PLAN_MARGIN for e in ends)
            if self.air.violation(p, 0.0 if near_end else PLAN_MARGIN / 2) is not None:
                return False
        return True

    def leg(self, a: Point, b: Point) -> list[Point] | None:
        """Shortest safe path a->b, or None. Both ends must themselves be legal."""
        if self.air.violation(a) or self.air.violation(b):
            return None
        ends = (a, b)
        if self.clear(a, b, ends):
            return [a, b]
        s, g = self._nearest_free(a), self._nearest_free(b)
        if s is None or g is None:
            return None
        cells = self._astar(s, g)
        if cells is None:
            return None
        return self._shortcut([a] + [self.center(c) for c in cells] + [b], ends)

    def _nearest_free(self, p: Point):
        """Closest free cell connected to p by a clear straight line (searched in distance order)."""
        c0 = self.cell_of(p)
        cands = []
        for i in range(c0[0] - 5, c0[0] + 6):
            for j in range(c0[1] - 5, c0[1] + 6):
                if 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[i][j]:
                    cands.append((dist(p, self.center((i, j))), (i, j)))
        for _, c in sorted(cands):
            if self.clear(p, self.center(c), (p,)) or self.air.segment_violation(p, self.center(c)) is None:
                return c
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

    def _shortcut(self, pts: list[Point], ends) -> list[Point] | None:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.clear(pts[i], pts[j], ends):
                j -= 1
            if not self.clear(pts[i], pts[j], ends) and self.air.segment_violation(pts[i], pts[j]):
                return None
            out.append(pts[j])
            i = j
        return out


def plan_through(air: Airspace, start: Point, stops: list[Point]) -> list[Point] | None:
    """Chain safe legs start -> stops[0] -> stops[1] ..."""
    planner = GridPlanner(air)
    route = [start]
    for b in stops:
        leg = planner.leg(route[-1], b)
        if leg is None:
            return None
        route += leg[1:]
    return route


# --- escape and landing --------------------------------------------------------------------------------

def escape_path(air: Airspace, pos: Point) -> list[Point] | None:
    """If pos is inside a forbidden area, the shortest straight exit (+10 m) that never crosses
    a permanent no-fly zone or the geofence it is not already inside. [pos] if already legal."""
    if air.violation(pos) is None:
        return [pos]
    start_permanent = air.violation(pos, mission=False)
    cands = []
    for k in range(72):
        ang = 2 * math.pi * k / 72
        ux, uy = math.cos(ang), math.sin(ang)
        out_at, found = None, None
        for s in range(1, 400):
            p = (pos[0] + ux * s * 2.0, pos[1] + uy * s * 2.0)
            if start_permanent is None and air.violation(p, mission=False):
                break  # this direction runs into the jail or the fence
            if air.violation(p) is None:
                if out_at is None:
                    out_at = s * 2.0
                if s * 2.0 >= out_at + 10.0 and air.violation(p, PLAN_MARGIN / 2) is None:
                    found = (out_at, p)  # clear for 10 m beyond the exit, and not grazing the edge
                    break
            else:
                out_at = None  # back inside (a courtyard): keep looking further out
        if found:
            cands.append(found)
    if not cands:
        return None
    cands.sort()
    planner = GridPlanner(air)
    for _, p in cands:  # shortest exit from which the planner can continue (not a closed pocket)
        if planner._nearest_free(p) is not None:
            return [pos, p]
    return [pos, cands[0][1]]


def usable_pads(air: Airspace) -> dict[str, Point]:
    return {n: p for n, p in air.campus.pads.items() if air.violation(p) is None}


def route_to_nearest_pad(air: Airspace, pos: Point, wind: Wind, energy_wh: float,
                         floor_wh: float) -> tuple[str, list[Point]] | None:
    """The cheapest reachable usable pad (escaping first if needed), within energy_wh - floor_wh."""
    esc = escape_path(air, pos)
    if esc is None:
        return None
    best = None
    for name, pad in usable_pads(air).items():
        tail = plan_through(air, esc[-1], [pad])
        if tail is None:
            continue
        route = esc[:-1] + tail
        e = route_energy_wh(route, wind) + LANDING_WH
        if e <= energy_wh - floor_wh and (best is None or e < best[0]):
            best = (e, name, route)
    return None if best is None else (best[1], best[2])


# --- preflight shield ----------------------------------------------------------------------------------

@dataclass
class ShieldResult:
    ok: bool
    reasons: list[str]
    energy_wh: float


def coerce_route(route: object) -> list[Point] | None:
    """Raw proposer output -> points, or None if malformed (bounds checked before any geometry)."""
    if not isinstance(route, (list, tuple)) or not 2 <= len(route) <= 2000:
        return None
    out = []
    for p in route:
        if not (isinstance(p, (list, tuple)) and len(p) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        and abs(v) <= MAX_COORD for v in p)):
            return None
        out.append((float(p[0]), float(p[1])))
    if any(dist(a, b) > MAX_LEG for a, b in zip(out, out[1:])):
        return None
    return out


def preflight_check(route: object, air: Airspace, *, start: Point, energy_wh: float, capacity_wh: float,
                    forecast: Wind, airborne: bool, targets: list[Point], end_at: Point | None,
                    escape_first: bool = False) -> ShieldResult:
    """Check a whole route before it is flown (at take-off or when swapped in mid-flight).

    Rules: starts where the drone is; never enters forbidden space; visits the targets in
    order; ends at `end_at` (base or a pad); and the energy for the route (with a wind
    margin), landing, any take-off and the floor reserve fits in what is left.
    """
    reasons: list[str] = []
    pts = coerce_route(route)
    if pts is None:
        return ShieldResult(False, ["route is malformed, too long or out of bounds"], math.inf)
    if dist(pts[0], start) > 1.0:
        reasons.append("route does not start at the drone's position")
    if path_length(pts) == 0:
        reasons.append("route has zero length")
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        # the escape leg starts inside a mission zone; it must still respect the permanent rules
        why = air.segment_violation(a, b, mission=not (escape_first and i == 0))
        if why:
            reasons.append(f"route enters {why}")
            break
    if not visits_in_order(pts, targets):
        reasons.append("route does not visit every target in the requested order")
    if end_at is not None and dist(pts[-1], end_at) > VISIT_RADIUS:
        reasons.append("route does not end at its landing point")
    if not flyable_weather(forecast):
        reasons.append(f"wind {forecast.speed:.0f} m/s is too strong to fly")
    e = route_energy_wh(pts, forecast.scaled(FORECAST_MARGIN)) + LANDING_WH + (0.0 if airborne else TAKEOFF_WH)
    budget = energy_wh - FLOOR_FRACTION * capacity_wh
    if e > budget:
        reasons.append(f"needs {e:.1f} Wh but only {max(budget, 0):.1f} Wh is available above the reserve")
    return ShieldResult(not reasons, reasons, e)


def visits_in_order(track: list[Point], targets: list[Point], radius: float = VISIT_RADIUS) -> bool:
    """Does the track pass within `radius` of each target, in the given order?"""
    return visit_count(track, targets, radius) == len(targets)


def visit_count(track: list[Point], targets: list[Point], radius: float = VISIT_RADIUS) -> int:
    """How many of the targets the track reaches, in order (a target counts only after the previous)."""
    if not targets:
        return 0
    k = 0
    pairs = list(zip(track, track[1:])) or [(track[0], track[0])]
    for a, b in pairs:
        for p in sample_segment(a, b, SAMPLE):
            if k < len(targets) and dist(p, targets[k]) <= radius:
                k += 1  # one target per sample point, so close neighbours are not skipped
            if k == len(targets):
                return k
    return k
