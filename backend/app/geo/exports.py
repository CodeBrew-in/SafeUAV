"""Downloadable records of one flight: JSONL audit log, track CSV and a standalone HTML report.

Everything here is a pure function of a plain-data snapshot (`Session.export_data()`), so it is easy to
test. The report is built from a fixed template and the recorded facts only. No generated text is added,
and every value is HTML-escaped. The operator's own chat lines and the AI's replies are quoted as
recorded, and labelled as such.
"""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime, timezone

DISCLAIMER = ("SIMULATION ONLY. This is a record of a simulated flight in a research prototype. "
              "It is not flight software and not evidence of real-world safety.")
ATTRIBUTION = "Map data © OpenStreetMap contributors (ODbL)."


def audit_jsonl(d: dict) -> str:
    """One JSON object per line: a meta line first, then every audit event and chat message in order."""
    lines = [{"type": "meta", "version": d["version"], "exported_utc": d["exported_utc"], "phase": d["phase"],
              "config": d["config"], "mission": d["mission"], "note": "simulation only"}]
    lines += [{"type": "audit", **e} for e in d["audit"]]
    lines += [{"type": "chat", **c} for c in d["chat"]]
    return "\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in lines) + "\n"


def _cell(v) -> str:
    """CSV cells that start with = + - @ can be run as formulas by spreadsheets; keep text inert."""
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") and not _is_number(s) else s


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def track_csv(d: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["t_s", "x_m", "y_m", "state"])
    for t, x, y, st in d["track"]:
        w.writerow([round(t, 1), round(x, 1), round(y, 1), _cell(st)])
    return out.getvalue()


def _svg_map(d: dict) -> str:
    x0, y0, x1, y1 = d["fence"]
    W, H = 640, 640 * (y1 - y0) / (x1 - x0)
    sx = lambda x: (x - x0) / (x1 - x0) * W
    sy = lambda y: H - (y - y0) / (y1 - y0) * H
    pts = lambda poly: " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in poly)
    g = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flight track on the campus map">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f6f4ee" stroke="#98a2b3"/>']
    for z in d["no_fly"]:
        g.append(f'<polygon points="{pts(z["poly"])}" fill="#d92d2033" stroke="#d92d20"/>')
    for z in d["zones"]:
        g.append(f'<polygon points="{pts(z["poly"])}" fill="#ee7a0033" stroke="#ee7a00"/>')
    track = [(x, y) for _, x, y, _ in d["track"]]
    if len(track) > 1:
        g.append(f'<polyline points="{pts(track)}" fill="none" stroke="#1d5fd1" stroke-width="2"/>')
    for n, (x, y) in d["pads"].items():
        g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#15a05a"><title>{html.escape(n)}</title></circle>')
    for i, t in enumerate(d["targets"], 1):
        g.append(f'<circle cx="{sx(t["center"][0]):.1f}" cy="{sy(t["center"][1]):.1f}" r="7" fill="#1d5fd1"/>'
                 f'<text x="{sx(t["center"][0]):.1f}" y="{sy(t["center"][1]) + 4:.1f}" font-size="10" fill="#fff" text-anchor="middle">{i}</text>')
    g.append("</svg>")
    return "".join(g)


def report_html(d: dict) -> str:
    e = html.escape
    audit = d["audit"]
    count = lambda *kinds: sum(1 for a in audit if a["kind"] in kinds)
    rows = [("Outcome", d["phase"]), ("Simulated flight time", f'{d["t"]:.0f} s'), ("Battery left", f'{d["battery_wh"]:.1f} Wh ({d["battery_pct"]:.0f} %)'),
            ("Targets reached", ", ".join(d["visited"]) or "none"),
            ("Orders refused by the shield", str(count("order_rejected", "replan_failed"))),
            ("Replans", str(count("replan"))), ("Emergency or battery events", str(count("monitor") + sum(1 for a in audit if a["kind"] == "state" and "EMERGENCY" in a["msg"])))]
    mission = d["mission"]
    av = ", ".join(a["zone"] + (f' ({a.get("from") or "now"}-{a.get("to") or "end"})' if a.get("from") or a.get("to") else "") for a in mission["avoid"]) or "none"
    cfg = d["config"]
    settings = [("Battery", f'{cfg["capacity"]:g} Wh'), ("Real wind", f'{cfg["wind_speed"]:g} m/s from {cfg["wind_from"]:g}°'),
                ("Start clock", cfg["clock"]), ("Base", cfg["base_pad"]), ("Mode", cfg["mode"]), ("AI model", cfg["model"]),
                ("Runtime battery monitor", "on" if cfg["monitor"] else "off")]
    table = lambda items: "<table>" + "".join(f"<tr><th>{e(a)}</th><td>{e(b)}</td></tr>" for a, b in items) + "</table>"
    events = "".join(f'<tr><td>{e(str(a.get("clock") or ""))}</td><td>{e(str(a["state"]))}</td><td>{e(a["kind"])}</td><td>{e(str(a["msg"]))}</td></tr>' for a in audit)
    chat = "".join(f'<tr><td>{e(str(c.get("clock", "")))}</td><td>{e(c["who"])}</td><td>{e(str(c["text"]))}</td></tr>' for c in d["chat"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SafeUAV flight report</title>
<style>
body{{font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#101828}}
h1{{margin:0}} h2{{margin-top:28px;border-bottom:1px solid #d0d5dd;padding-bottom:4px}}
.banner{{background:#fef3c7;border:1px solid #f59e0b;padding:8px 12px;border-radius:8px;margin:12px 0;font-weight:600}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #d0d5dd;padding:4px 8px;text-align:left;vertical-align:top;font-size:14px}}
th{{background:#f2f4f7;width:30%}} .ev th{{width:auto}} svg{{max-width:100%;height:auto}} footer{{margin-top:32px;color:#667085;font-size:13px}}
</style></head><body>
<h1>SafeUAV flight report</h1>
<div>Version {e(d["version"])} · exported {e(d["exported_utc"])}</div>
<div class="banner">{e(DISCLAIMER)}</div>
<h2>Summary</h2>{table(rows)}
<h2>Mission</h2>{table([("Targets, in order", " → ".join(mission["targets"]) or "none"), ("Restricted areas", av), ("After the last target", "stays there" if mission["stay_at_end"] else "returns to base")])}
<h2>Settings</h2>{table(settings)}
<h2>Track</h2>{_svg_map(d)}
<p style="font-size:13px">Red: permanent no-fly zones. Orange: areas restricted by the operator. Blue line: flown track. Green dots: landing pads.</p>
<h2>Events</h2>
<table class="ev"><tr><th>Clock</th><th>State</th><th>Kind</th><th>What happened</th></tr>{events}</table>
<h2>Chat (as recorded; the operator's words and the AI's replies)</h2>
<table class="ev"><tr><th>Clock</th><th>From</th><th>Text</th></tr>{chat}</table>
<footer>{e(ATTRIBUTION)} · {e(DISCLAIMER)}</footer>
</body></html>
"""


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
