"""Background map for the console: roads, paths, every building, grass, water and walls, taken from the
same pinned OSM snapshot as the safety map (so drawing and rules always agree).

This is only for looking at. Nothing here is used by the planner or the shield.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

from .campus import GEOFENCE_MARGIN_M, load_campus

OSM = Path(__file__).resolve().parents[3] / "benchmark" / "geo" / "thapar_osm.xml"
CAMPUS_NAME = "Thapar Institute of Engineering and Technology"

ROAD_CLASS = {"primary": "major", "primary_link": "major", "secondary": "major", "secondary_link": "major",
              "tertiary": "minor", "tertiary_link": "minor", "residential": "minor", "unclassified": "minor",
              "living_street": "minor", "service": "service", "footway": "path", "path": "path",
              "pedestrian": "path", "steps": "path", "cycleway": "path", "track": "path"}
GREEN_LANDUSE = {"grass", "forest", "meadow", "recreation_ground", "village_green", "orchard"}
GREEN_LEISURE = {"park", "garden", "pitch", "playground", "golf_course", "nature_reserve", "track"}


def _r(p):
    return [round(p[0], 1), round(p[1], 1)]


@functools.lru_cache(maxsize=1)
def build_basemap() -> dict:
    campus = load_campus()
    proj = campus.projection
    root = ET.parse(OSM).getroot()
    nodes = {n.get("id"): proj.to_xy(float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
    ways = {}
    for w in root.iter("way"):
        refs = [nd.get("ref") for nd in w.iter("nd")]
        if len(refs) >= 2 and all(r in nodes for r in refs):
            ways[w.get("id")] = (refs, {t.get("k"): t.get("v") for t in w.iter("tag")})

    out = {"roads": [], "paths": [], "buildings": [], "green": [], "water": [], "waterways": [], "walls": [],
           "landuse": []}

    def area_kind(tags):
        if "building" in tags:
            return "buildings"
        if tags.get("natural") == "water" or tags.get("landuse") == "reservoir":
            return "water"
        if tags.get("landuse") in GREEN_LANDUSE or tags.get("leisure") in GREEN_LEISURE or tags.get("natural") in ("wood", "scrub", "grassland"):
            return "green"
        if tags.get("landuse") in ("residential", "industrial", "commercial", "retail", "education"):
            return "landuse"
        return None

    def add_area(refs, tags):
        k = area_kind(tags)
        if k and len(refs) >= 4 and refs[0] == refs[-1]:
            out[k].append([_r(nodes[r]) for r in refs[:-1]])

    for refs, tags in ways.values():
        hw = tags.get("highway")
        if hw in ROAD_CLASS:
            cls = ROAD_CLASS[hw]
            (out["paths"] if cls == "path" else out["roads"]).append(
                {"c": cls, "name": tags.get("name"), "pts": [_r(nodes[r]) for r in refs]})
        elif tags.get("waterway"):
            out["waterways"].append([_r(nodes[r]) for r in refs])
        elif tags.get("barrier") in ("wall", "fence", "retaining_wall", "hedge"):
            out["walls"].append([_r(nodes[r]) for r in refs])
        else:
            add_area(refs, tags)
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.iter("tag")}
        if tags.get("type") != "multipolygon":
            continue
        for m in rel.iter("member"):
            if m.get("type") == "way" and m.get("role") == "outer" and m.get("ref") in ways:
                add_area(ways[m.get("ref")][0], {**ways[m.get("ref")][1], **tags})

    camp = campus.places.get(CAMPUS_NAME.lower().replace(" ", ""))
    xs = [p[0] for p in camp.polygon] if camp else [campus.fence[0], campus.fence[2]]
    ys = [p[1] for p in camp.polygon] if camp else [campus.fence[1], campus.fence[3]]
    pad = 40
    out["focus"] = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    x0, y0, x1, y1 = campus.fence
    m = GEOFENCE_MARGIN_M
    out["bounds"] = [x0 - m, y0 - m, x1 + m, y1 + m]   # where map data exists: the view never leaves it
    out["campus"] = [_r(p) for p in camp.polygon] if camp else []
    return out
