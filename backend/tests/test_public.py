"""Public-demo mode: host and origin rules, rate limits, the daily AI budget, and the hosted-model restriction."""

import http.client
import json
import threading

import pytest

from app.geo.llm import LLMReply
from app.geo.public import AiBudget, BudgetLLM, Limiter, PublicConfig, PublicConfigError, from_env
from app.geo.server import App, make_server
from app.geo.session import Session, SessionError

HOST = "demo.example.com"
D1, D2 = "1" * 32, "2" * 32


class FakeLLM:
    def ask(self, system, user, schema, seed=0, max_tokens=0):
        d = {"targets": ["Nava Nalanda Library"], "avoid": [], "stay_at_end": False, "refuse": False, "reason": ""}
        return LLMReply("fake", "fake", str(d), d, 1.0, None)


@pytest.fixture(scope="module")
def pub():
    cfg = PublicConfig(enabled=True, host=HOST, bind="127.0.0.1", max_drones=3, chat_per_min=3, chat_per_ip_min=100,
                       starts_per_hour=4, ai_daily_budget=5)
    app = App(Session(llm_factory=lambda m: FakeLLM()), public=cfg)
    httpd = make_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app
    httpd.shutdown()


def call(app, method, path, body=None, host=HOST, token=True, origin=None, drone=None, xff=None):
    c = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5)
    h = {"Host": host, "Content-Type": "application/json"}
    if token:
        h["X-Session-Token"] = app.token
    if origin:
        h["Origin"] = origin
    if drone:
        h["X-Drone-Id"] = drone
    if xff:
        h["X-Forwarded-For"] = xff
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    try:
        js = json.loads(raw)
    except ValueError:
        js = None
    c.close()
    return r.status, js, raw, dict(r.getheaders())


def test_only_the_public_host_is_accepted_and_hsts_is_sent(pub):
    st, _, raw, hdr = call(pub, "GET", "/", token=False)
    assert st == 200 and pub.token.encode() in raw and "Strict-Transport-Security" in hdr
    for bad in ("127.0.0.1:8000", "localhost", "evil.example.com", HOST + ":8000"):
        assert call(pub, "GET", "/", host=bad, token=False)[0] == 403
    assert call(pub, "POST", "/api/reset", {}, origin="https://evil.example.com")[0] == 403
    assert call(pub, "POST", "/api/reset", {}, origin=f"http://{HOST}")[0] == 403          # must be https
    assert call(pub, "POST", "/api/reset", {}, origin=f"https://{HOST}", drone=D1)[0] == 200
    assert call(pub, "GET", "/api/state", token=False)[0] == 403


def test_healthz_needs_no_token_and_leaks_nothing(pub):
    st, _, raw, _ = call(pub, "GET", "/healthz", token=False, host="anything.internal")
    assert st == 200 and raw == b"ok"


def test_public_page_data_has_the_hosted_model_only_and_the_budget_in_health(pub):
    m = call(pub, "GET", "/api/map")[1]
    assert m["public"] is True and all(x.startswith("gemini") for x in m["models"])
    h = call(pub, "GET", "/api/health")[1]
    assert h["public"] is True and h["ollama_models"] == [] and h["ai_budget_left"] == 5
    assert b"AIza" not in json.dumps(h).encode()
    st, js, _, _ = call(pub, "POST", "/api/config", {"model": "qwen2.5:7b"}, drone=D1)
    assert st == 400 and "Model must be one of" in js["error"]


def test_chat_rate_limit_per_drone_and_drone_cap(pub):
    for _ in range(3):
        assert call(pub, "POST", "/api/chat", {"text": "inspect the library"}, drone=D2)[0] in (200, 409)
    st, js, _, _ = call(pub, "POST", "/api/chat", {"text": "inspect the library"}, drone=D2)
    assert st == 429 and "Too many" in js["error"]
    assert call(pub, "POST", "/api/chat", {"text": "inspect the library"}, drone="3" * 32)[0] in (200, 409)   # another drone is fine
    # the cap is 3 drones in total and the others were active a moment ago
    st, js, _, _ = call(pub, "GET", "/api/state", drone="4" * 32)
    assert st == 400 and "busy" in js["error"]


def test_starts_per_hour_limit_by_client_address(pub):
    ok = [call(pub, "POST", "/api/start", {}, drone=D1, xff="9.9.9.9, 8.8.8.8")[0] for _ in range(6)]
    assert 429 in ok and ok.index(429) <= 4                                                    # limit 4 per address per hour
    other = call(pub, "POST", "/api/start", {}, drone=D1, xff="1.2.3.4, 7.7.7.7")[0]
    assert other != 429                                                                        # a different address is unaffected


def test_the_daily_ai_budget_runs_out_and_the_session_falls_back():
    now = [86400 * 100.0]
    b = AiBudget(2, clock=lambda: now[0])
    llm = BudgetLLM(FakeLLM(), b)
    assert llm.ask("s", "u", {}).error is None and llm.ask("s", "u", {}).error is None
    r = llm.ask("s", "u", {})
    assert r.error and "daily AI limit" in r.error and b.left() == 0
    now[0] += 86400                                                                            # next UTC day
    assert b.left() == 2 and llm.ask("s", "u", {}).error is None
    s = Session(llm_factory=lambda m: BudgetLLM(FakeLLM(), AiBudget(0)))
    s.edit("target_add", "Nava Nalanda Library")
    s.handle("also inspect the directorate")
    assert s.desired.targets == ("Nava Nalanda Library", "Directorate")                        # the basic parser answered
    assert any(c.get("status") == "fallback" for c in s.chat)


def test_limiter_window():
    lim = Limiter()
    assert all(lim.allow("k", "a", 3, 10, now=t) for t in (0, 1, 2))
    assert not lim.allow("k", "a", 3, 10, now=3)
    assert lim.allow("k", "a", 3, 10, now=11.5)                                                # old hits expired
    assert lim.allow("k", "b", 3, 10, now=3)                                                   # another key is separate


def test_env_config_is_validated():
    assert from_env({}).enabled is False
    with pytest.raises(PublicConfigError, match="SAFEUAV_PUBLIC_HOST"):
        from_env({"SAFEUAV_PUBLIC": "1"})
    for bad in ("a/b", "has space", "host:80"):
        with pytest.raises(PublicConfigError):
            from_env({"SAFEUAV_PUBLIC": "1", "SAFEUAV_PUBLIC_HOST": bad})
    with pytest.raises(PublicConfigError, match="whole number"):
        from_env({"SAFEUAV_PUBLIC": "1", "SAFEUAV_PUBLIC_HOST": HOST, "PORT": "abc"})
    with pytest.raises(PublicConfigError, match="between"):
        from_env({"SAFEUAV_PUBLIC": "1", "SAFEUAV_PUBLIC_HOST": HOST, "SAFEUAV_AI_DAILY_BUDGET": "-5"})
    c = from_env({"SAFEUAV_PUBLIC": "1", "RENDER_EXTERNAL_HOSTNAME": "SafeUAV.onrender.com", "PORT": "10000"})
    assert c.enabled and c.host == "safeuav.onrender.com" and c.bind == "0.0.0.0" and c.port == 10000
    assert c.max_drones == 20 and c.ai_daily_budget == 200


def test_local_mode_is_unchanged_and_listens_on_loopback_only():
    app = App(Session(llm_factory=lambda m: FakeLLM()))
    assert app.public.enabled is False and app.public.bind == "127.0.0.1"
    srv = make_server(app, 0)
    try:
        assert srv.server_address[0] == "127.0.0.1"
        assert app.hosts() == {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}
    finally:
        srv.server_close()
