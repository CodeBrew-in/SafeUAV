"""The local server: host/origin/token/content-type/size checks and the main flow, over real HTTP."""

import http.client
import json
import threading
import time

import pytest

from app.geo.llm import LLMReply
from app.geo.server import MAX_BODY, App, make_server
from app.geo.session import Session


class FakeLLM:
    def ask(self, system, user, schema, seed=0, max_tokens=0):
        d = {"targets": ["Nava Nalanda Library"], "avoid": [], "stay_at_end": False, "refuse": False, "reason": ""}
        return LLMReply("fake", "fake", str(d), d, 1.0, None)


@pytest.fixture(scope="module")
def srv():
    app = App(Session(llm_factory=lambda m: FakeLLM()))
    httpd = make_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app
    httpd.shutdown()


def call(app, method, path, body=None, **kw):
    for attempt in (1, 2):
        try:
            return _call(app, method, path, body, **kw)
        except ConnectionError:
            if attempt == 2:
                raise


def _call(app, method, path, body=None, *, token=True, host=None, origin=None, ctype="application/json",
          clen=None, raw=None, drone=None):
    c = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{app.port}"}
    if token:
        h["X-Session-Token"] = app.token if token is True else token
    if origin:
        h["Origin"] = origin
    if drone:
        h["X-Drone-Id"] = drone
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    if method == "POST":
        if ctype:
            h["Content-Type"] = ctype
        h["Content-Length"] = str(len(data or b"") if clen is None else clen)
    c.putrequest(method, path, skip_host=True)
    for k, v in h.items():
        c.putheader(k, v)
    c.endheaders(data if method == "POST" else None)
    r = c.getresponse()
    out = r.read()
    try:
        js = json.loads(out)
    except ValueError:
        js = None
    c.close()
    return r.status, js, out, dict(r.getheaders())


def test_page_has_token_and_security_headers(srv):
    st, _, body, hdr = call(srv, "GET", "/", token=False)
    assert st == 200 and srv.token.encode() in body and b"__TOKEN__" not in body
    assert hdr["Content-Security-Policy"].startswith("default-src 'self'") and hdr["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("host", ["evil.example.com", "127.0.0.1:1", "localhost"])
def test_bad_host_is_refused_even_for_the_page(srv, host):
    assert call(srv, "GET", "/", host=host, token=False)[0] == 403
    assert call(srv, "GET", "/api/state", host=host)[0] == 403


def test_foreign_origin_is_refused(srv):
    assert call(srv, "POST", "/api/reset", {}, origin="http://evil.example.com")[0] == 403
    assert call(srv, "POST", "/api/reset", {}, origin=f"http://localhost:{srv.port}")[0] == 200


@pytest.mark.parametrize("token", [False, "wrong", "x" * 40])
def test_missing_or_wrong_token_is_refused(srv, token):
    assert call(srv, "GET", "/api/state", token=token)[0] == 403
    assert call(srv, "POST", "/api/start", {}, token=token)[0] == 403
    assert call(srv, "GET", "/api/map", token=token)[0] == 403


def test_post_needs_json_and_sane_length(srv):
    assert call(srv, "POST", "/api/reset", {}, ctype="text/plain")[0] == 415
    assert call(srv, "POST", "/api/reset", {}, ctype=None)[0] == 415
    assert call(srv, "POST", "/api/reset", raw=b"{}", clen="abc")[0] == 400
    assert call(srv, "POST", "/api/reset", raw=b"x", clen=MAX_BODY + 1)[0] == 413
    assert call(srv, "POST", "/api/reset", raw=b"not json")[0] == 400
    assert call(srv, "POST", "/api/reset", raw=b"[1]")[0] == 400


def test_bad_inputs_get_400_not_500(srv):
    assert call(srv, "GET", "/api/state?epoch=abc")[0] == 400
    assert call(srv, "POST", "/api/speed", {"speed": 20})[0] == 400
    assert call(srv, "POST", "/api/speed", {"speed": "fast"})[0] == 400
    assert call(srv, "POST", "/api/edit", {"op": "target_add", "place": "Central Jail Patiala"})[0] == 400
    assert call(srv, "POST", "/api/edit", {"place": "x"})[0] == 400
    assert call(srv, "POST", "/api/emergency", {"kind": "explode"})[0] == 400
    assert call(srv, "POST", "/api/config", {"clock": "banana"})[0] == 400
    assert call(srv, "POST", "/api/nope", {})[0] == 404
    assert call(srv, "GET", "/static/../server.py", token=False)[0] == 404


def test_start_without_target_is_a_clear_400(srv):
    call(srv, "POST", "/api/reset", {})
    call(srv, "POST", "/api/edit", {"op": "clear"})
    st, js, _, _ = call(srv, "POST", "/api/start", {})
    assert st == 400 and "target" in js["error"]


def test_full_flow_over_http(srv):
    call(srv, "POST", "/api/reset", {})
    assert call(srv, "POST", "/api/edit", {"op": "target_add", "place": "Nava Nalanda Library"})[0] == 200
    assert call(srv, "POST", "/api/start", {})[0] == 200
    for _ in range(15):
        srv.session.tick()
    st, js, _, _ = call(srv, "GET", "/api/state")
    assert st == 200 and js["phase"] == "FLYING" and js["track_total"] > 5 and js["route"]
    assert call(srv, "POST", "/api/pause", {})[0] == 200
    assert call(srv, "POST", "/api/start", {})[0] == 400          # double start
    assert call(srv, "POST", "/api/config", {"capacity": 60})[0] == 400   # locked
    assert call(srv, "POST", "/api/reset", {})[0] == 200
    assert call(srv, "GET", "/api/state")[1]["phase"] == "DRAFT"


def test_chat_over_http_and_busy_guard(srv):
    call(srv, "POST", "/api/reset", {})
    st, _, _, _ = call(srv, "POST", "/api/chat", {"text": "inspect the library"})
    assert st == 200
    for _ in range(50):
        if srv.session.desired.targets:
            break
        time.sleep(0.05)
    assert srv.session.desired.targets == ("Nava Nalanda Library",)


def test_state_never_contains_the_token_or_a_key(srv):
    _, _, raw, _ = call(srv, "GET", "/api/state")
    assert srv.token.encode() not in raw
    _, _, raw, _ = call(srv, "GET", "/api/health")
    assert b"AIza" not in raw and srv.token.encode() not in raw


def test_concurrent_polls_and_edits_do_not_break(srv):
    call(srv, "POST", "/api/reset", {})
    errs = []

    def poll():
        for _ in range(10):
            if call(srv, "GET", "/api/state")[0] != 200:
                errs.append("state")

    def edit():
        for _ in range(5):
            call(srv, "POST", "/api/edit", {"op": "avoid_add", "place": "E Hostel"})
            call(srv, "POST", "/api/edit", {"op": "avoid_remove", "place": "E Hostel"})

    ts = [threading.Thread(target=poll) for _ in range(5)] + [threading.Thread(target=edit)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert not errs


def test_review_hardening_headers_nan_deep_json_and_strict_values(srv):
    _, _, _, hdr = call(srv, "GET", "/", token=False)
    assert "frame-ancestors 'none'" in hdr["Content-Security-Policy"] and hdr["X-Frame-Options"] == "DENY"
    call(srv, "POST", "/api/reset", {})
    for raw in (b'{"wind_from": NaN}', b'{"wind_speed": Infinity}', b"[" * 5000):
        assert call(srv, "POST", "/api/config", raw=raw)[0] == 400
    assert call(srv, "GET", "/api/state")[0] == 200             # still valid JSON afterwards
    assert call(srv, "POST", "/api/config", raw=b'{"monitor": "false"}')[0] == 400
    assert call(srv, "POST", "/api/speed", {"speed": 2.9})[0] == 400
    assert call(srv, "POST", "/api/speed", {"speed": True})[0] == 400
    assert call(srv, "POST", "/api/edit", {"op": "stay", "value": "false"})[0] == 400
    assert call(srv, "POST", "/api/reset", raw=b"{}", clen="+5")[0] == 400
    assert call(srv, "POST", "/api/reset", raw=b"{}", clen="1_0")[0] == 400
    assert call(srv, "POST", "/api/config", {"model": "gpt"})[1]["error"].startswith("Model must be one of")


A, B = "a" * 32, "b" * 32


def test_each_browser_tab_has_its_own_drone(srv):
    for d in (None, A, B):
        call(srv, "POST", "/api/reset", {}, drone=d)
        call(srv, "POST", "/api/edit", {"op": "clear"}, drone=d)
    assert call(srv, "POST", "/api/edit", {"op": "target_add", "place": "Nava Nalanda Library"}, drone=A)[0] == 200
    assert call(srv, "GET", "/api/state", drone=A)[1]["desired"]["targets"] == ["Nava Nalanda Library"]
    assert call(srv, "GET", "/api/state", drone=B)[1]["desired"]["targets"] == []
    assert call(srv, "GET", "/api/state")[1]["desired"]["targets"] == []
    assert call(srv, "POST", "/api/start", {}, drone=A)[0] == 200
    assert call(srv, "GET", "/api/state", drone=B)[1]["phase"] == "DRAFT"      # B is not flying
    assert call(srv, "POST", "/api/pause", {}, drone=B)[0] == 400
    assert call(srv, "POST", "/api/reset", {}, drone=A)[0] == 200
    for bad in ("short", "G" * 32, "a" * 33):
        assert call(srv, "GET", "/api/state", drone=bad)[0] == 400


def test_the_downloads_are_valid_and_inert(srv):
    call(srv, "POST", "/api/reset", {}, drone=A)
    call(srv, "POST", "/api/edit", {"op": "target_add", "place": "Nava Nalanda Library"}, drone=A)
    call(srv, "POST", "/api/start", {}, drone=A)
    s = srv.session_for(A)
    s.say("operator", "<script>alert(1)</script> =HYPERLINK(1)")
    for _ in range(30):
        s.tick()
    st, _, raw, hdr = call(srv, "GET", "/api/logs/audit.jsonl", drone=A)
    lines = [json.loads(x) for x in raw.decode().splitlines()]
    assert st == 200 and lines[0]["type"] == "meta" and any(x["type"] == "audit" for x in lines) and any(x["type"] == "chat" for x in lines)
    assert hdr["Content-Disposition"].startswith("attachment")
    st, _, raw, _ = call(srv, "GET", "/api/logs/track.csv", drone=A)
    rows = raw.decode().splitlines()
    assert rows[0] == "t_s,x_m,y_m,state" and len(rows) > 20
    st, _, raw, hdr = call(srv, "GET", "/api/logs/report.html", drone=A)
    page = raw.decode()
    assert "SIMULATION ONLY" in page and "<script>alert" not in page and "&lt;script&gt;" in page and "OpenStreetMap" in page
    assert call(srv, "GET", "/api/logs/secret.txt", drone=A)[0] == 404
    assert call(srv, "GET", "/api/logs/audit.jsonl", token=False)[0] == 403
    assert srv.token.encode() not in raw
    call(srv, "POST", "/api/reset", {}, drone=A)


def test_csv_cells_cannot_become_formulas():
    from app.geo.exports import _cell
    assert _cell("=1+1").startswith("'") and _cell("CRUISE") == "CRUISE" and _cell("-5") == "-5"


def test_drawn_areas_over_http(srv):
    call(srv, "POST", "/api/reset", {}, drone=B)
    cx, cy = srv.session.campus.pads["Football Ground"]
    st, js, _, _ = call(srv, "POST", "/api/circle", {"x": cx + 60, "y": cy + 60, "r": 50}, drone=B)
    assert st == 200 and js["name"] == "Area 1"
    snap = call(srv, "GET", "/api/state", drone=B)[1]
    assert [c["name"] for c in snap["circles"]] == ["Area 1"]
    assert call(srv, "GET", "/api/state", drone=A)[1]["circles"] == []                  # another tab does not see it
    assert "area 1" not in srv.session.base_campus.places                               # the shared campus is untouched
    for bad in ({"x": cx, "y": cy, "r": 5}, {"x": cx, "y": cy, "r": 900}, {"x": 1e9, "y": 0, "r": 50}, {"x": "a", "y": 0, "r": 50}):
        assert call(srv, "POST", "/api/circle", bad, drone=B)[0] == 400
    assert call(srv, "POST", "/api/edit", {"op": "avoid_add", "place": "Area 1"}, drone=B)[0] == 200
    assert call(srv, "POST", "/api/edit", {"op": "circle_remove", "place": "Area 1"}, drone=B)[0] == 400   # still in the mission
    assert call(srv, "POST", "/api/edit", {"op": "avoid_remove", "place": "Area 1"}, drone=B)[0] == 200
    assert call(srv, "POST", "/api/edit", {"op": "circle_remove", "place": "Area 1"}, drone=B)[0] == 200
    call(srv, "POST", "/api/reset", {}, drone=B)


def test_settings_file_is_validated_and_cannot_touch_safety_constants(tmp_path):
    from app.geo.server import App
    from app.geo.settings import SettingsError, load
    good = tmp_path / "ok.toml"
    good.write_text('[server]\nport = 8123\n[session]\nmodel = "qwen2.5:3b"\ncapacity = 30\nspeed = 5\n', encoding="utf-8")
    cfg = load(good)
    assert cfg == {"port": 8123, "session": {"model": "qwen2.5:3b", "capacity": 30, "speed": 5}}
    app = App(Session(llm_factory=lambda m: FakeLLM()), defaults=cfg["session"])
    assert app.session.cfg.capacity == 30 and app.session.speed == 5
    assert app.session_for("c" * 32).cfg.model == "qwen2.5:3b"            # new drones get the defaults too
    for bad in ('[session]\nreserve = 0\n', '[session]\nfloor_fraction = 0\n', '[session]\nmonitor = "no"\n',
                '[server]\nport = 80\n', '[x]\na = 1\n', '[session]\ncapacity = true\n', 'not toml ['):
        f = tmp_path / "bad.toml"
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(SettingsError):
            load(f)
