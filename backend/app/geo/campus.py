"""The Thapar campus map, built from OpenStreetMap data (ODbL, (c) OpenStreetMap contributors).

Named features become places a mission can refer to (inspection targets, zones
to avoid). A small set of PERMANENT no-fly zones is trusted configuration: no
mission text or model output can remove them.

Assumed policy (stated, not official): the drone may not enter the airspace of
Central Jail Patiala (a real sensitive site next to campus) or leave the mapped
area (the geofence).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .geometry import Point, Polygon, Projection, bbox, distance_to_polygon, interior_point

REPO_ROOT = Path(__file__).resolve().parents[3]
OSM_FILE = REPO_ROOT / "benchmark" / "geo" / "thapar_osm.xml"

# Permanent no-fly zones by OSM name, with the safety buffer (metres) the shield enforces.
PERMANENT_NO_FLY = {"Central Jail Patiala": 30.0}
GEOFENCE_MARGIN_M = 25.0      # stay this far inside the mapped area
BASE_NAME = "Football Ground"  # launch and recovery point
# Designated landing pads (assumed marked pads on open ground; the base is one of them).
# SAFE_HOLD and EMERGENCY may only land on these, never on roofs or arbitrary pitches.
LANDING_PADS = ("Football Ground", "Fete Area", "SBOP lawns", "Nirvana Meditation Park")


def norm(name: str) -> str:
    """Comparison key: Unicode NFKC, case-folded, letters and digits only (any script).
    Raises ValueError for names with no letters or digits, so '' never matches anything."""
    key = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(name)).casefold())
    if not key:
        raise ValueError(f"place name {name!r} has no letters or digits")
    return key


@dataclass(frozen=True)
class Place:
    name: str
    kind: str
    polygon: tuple[Point, ...]
    center: Point


@dataclass
class Campus:
    projection: Projection
    places: dict[str, Place]                  # normalised name -> place (unique names only)
    ambiguous: set[str]                       # normalised names that occur more than once
    permanent_no_fly: dict[str, tuple[Polygon, float]]  # name -> (polygon, buffer m)
    fence: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax after the margin
    base: Point
    pads: dict[str, Point] = field(default_factory=dict)  # landing pad name -> touchdown point
    osm_sha256: str = ""

    def find(self, name: str) -> Place | None:
        try:
            return self.places.get(norm(name))
        except ValueError:
            return None

    def in_fence(self, p: Point) -> bool:
        x0, y0, x1, y1 = self.fence
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def permanent_violation(self, p: Point) -> str | None:
        """Name of the permanent no-fly zone p is in (including its buffer), else None."""
        for name, (poly, buf) in self.permanent_no_fly.items():
            if distance_to_polygon(p, poly) < buf:
                return name
        return None


@lru_cache(maxsize=2)
def load_campus(path: str = str(OSM_FILE)) -> Campus:
    root = ET.parse(path).getroot()
    b = root.find("bounds")
    minlat, minlon = float(b.get("minlat")), float(b.get("minlon"))
    maxlat, maxlon = float(b.get("maxlat")), float(b.get("maxlon"))
    proj = Projection((minlat + maxlat) / 2, (minlon + maxlon) / 2)
    nodes = {n.get("id"): proj.to_xy(float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}

    places: dict[str, Place] = {}
    counts: dict[str, int] = {}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        name = tags.get("name")
        kind = tags.get("building") or tags.get("amenity") or tags.get("leisure")
        refs = [nd.get("ref") for nd in w.iter("nd")]
        if not name or not kind or len(refs) < 4 or refs[0] != refs[-1]:
            continue  # only named, closed areas
        if not all(r in nodes for r in refs):
            continue
        poly = tuple(nodes[r] for r in refs[:-1])
        try:
            key = norm(name)
        except ValueError:
            continue
        counts[key] = counts.get(key, 0) + 1
        places[key] = Place(name, kind, poly, interior_point(list(poly)))
    ambiguous = {k for k, v in counts.items() if v > 1}
    for k in ambiguous:
        del places[k]

    permanent = {}
    for name, buf in PERMANENT_NO_FLY.items():
        p = places.get(norm(name))
        if p is None:
            raise ValueError(f"permanent no-fly zone {name!r} not found in {path}")
        permanent[name] = (list(p.polygon), buf)

    x0, y0 = proj.to_xy(minlat, minlon)
    x1, y1 = proj.to_xy(maxlat, maxlon)
    fence = (x0 + GEOFENCE_MARGIN_M, y0 + GEOFENCE_MARGIN_M, x1 - GEOFENCE_MARGIN_M, y1 - GEOFENCE_MARGIN_M)
    base = places[norm(BASE_NAME)].center
    pads = {n: places[norm(n)].center for n in LANDING_PADS}
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Campus(proj, places, ambiguous, permanent, fence, base, pads, digest)


def describe_places(c: Campus) -> list[dict]:
    """Compact place list for prompts: name, kind, centre and rough radius in metres."""
    out = []
    for p in sorted(c.places.values(), key=lambda p: p.name):
        x0, y0, x1, y1 = bbox(list(p.polygon))
        out.append({"name": p.name, "kind": p.kind, "x": round(p.center[0]), "y": round(p.center[1]),
                    "radius": round(max(x1 - x0, y1 - y0) / 2)})
    return out
