"""Planar geometry in local metres.

The campus is about 1.5 km across, so an equirectangular projection around a
fixed origin is accurate to well under a metre here. x grows east, y grows north.
"""

from __future__ import annotations

import math

Point = tuple[float, float]
Polygon = list[Point]

EARTH_M_PER_DEG_LAT = 110_540.0
EARTH_M_PER_DEG_LON_EQ = 111_320.0


class Projection:
    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = EARTH_M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))

    def to_xy(self, lat: float, lon: float) -> Point:
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * EARTH_M_PER_DEG_LAT)

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return (self.lat0 + y / EARTH_M_PER_DEG_LAT, self.lon0 + x / self.kx)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Ray casting; points exactly on an edge may go either way (callers use buffers)."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return dist(p, (ax + t * dx, ay + t * dy))


def distance_to_polygon(p: Point, poly: Polygon) -> float:
    """0 inside the polygon, otherwise the distance to its boundary."""
    if point_in_polygon(p, poly):
        return 0.0
    n = len(poly)
    return min(point_segment_distance(p, poly[i], poly[(i + 1) % n]) for i in range(n))


def centroid(poly: Polygon) -> Point:
    """Area centroid; falls back to the vertex mean for degenerate polygons."""
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-9:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    return (cx / (3 * a), cy / (3 * a))


def bbox(poly: Polygon) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def sample_segment(a: Point, b: Point, step: float) -> list[Point]:
    """Points along a->b, at most `step` apart, including both ends."""
    n = max(1, math.ceil(dist(a, b) / step))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def path_length(points: list[Point]) -> float:
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))
