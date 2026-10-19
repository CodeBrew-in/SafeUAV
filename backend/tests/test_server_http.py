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
          clen=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{app.port}"}
    if token:
        h["X-Session-Token"] = app.token if token is True else token
    if origin:
        h["Origin"] = origin
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
