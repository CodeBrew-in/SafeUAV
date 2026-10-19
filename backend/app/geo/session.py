"""One operator session: a draft mission, a flight, and the rules for editing both.

Why this layer exists
  * Nothing flies until `start()`. Before that, clicks and chat only change the DRAFT.
  * There is ONE desired mission (`desired`). Every click is a patch on it, and every chat
    reply is turned into a diff against the mission it was asked about, then applied to the
    CURRENT mission. A slow AI reply therefore cannot erase a click made while it was thinking.
  * Every change is validated here and again by the controller (validate -> plan -> shield).
    The controller itself is untouched, so the benchmarked behaviour is unchanged.
  * Pause and speed live here, not in the controller: the simulation always steps dt = 1 s.

All public methods are thread-safe (one RLock). `handle()` blocks on the AI, so call it from
a worker thread; the simulation keeps ticking (or holds, see `wait_for_ai`).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

from .campus import Campus, load_campus, norm
from .controller import FLYING, Command, Controller
from .flight import AVOID_BUFFER, Airspace, Wind, horizon_min
from .llm import GeminiLLM, OllamaLLM
from .operator import RegexParser, interpret
from .spec import AvoidRule, MissionSpec, parse_time, validate_spec

SPEEDS = (1, 2, 5)
DEFAULT_SPEED = 2
MODELS = ["gemini-3.5-flash-lite", "qwen2.5:7b", "qwen2.5:3b"]
MODES = ("ours", "classic", "direct")
MAX_AVOID = 20
MAX_TARGETS = 10
AI_DEADLINE_S = {"gemini": 30.0, "ollama": 60.0}
FORECAST = Wind(3, 270)  # what the planner is told; the real wind may differ


class SessionError(Exception):
    """A request that cannot be applied; the message is shown to the operator as is."""


# --- pure helpers (unit-tested) -----------------------------------------------------------------

def _k(name: str) -> str:
    return norm(name)


def merge_reply(base: MissionSpec, reply: MissionSpec, cur: MissionSpec) -> MissionSpec:
    """Apply what the AI changed (reply vs base) onto the current mission (cur).

    base  = the mission the AI was shown. reply = the mission the AI wrote.
    cur   = the mission now (clicks may have changed it while the AI was thinking).
    Only differences between base and reply are applied, so edits made meanwhile survive.
    """
    bt, rt = [_k(t) for t in base.targets], [_k(t) for t in reply.targets]
    removed = {k for k in bt if k not in rt}
    targets = [t for t in cur.targets if _k(t) not in removed]
    have = {_k(t) for t in targets}
    for t in reply.targets:
        if _k(t) not in bt and _k(t) not in have:
            targets.append(t)
            have.add(_k(t))
    if set(bt) == set(rt) and bt != rt:  # the AI only re-ordered the stops
        order = {k: i for i, k in enumerate(rt)}
        targets.sort(key=lambda t: order.get(_k(t), len(order)))

    b_av = {_k(r.zone): r for r in base.avoid}
    r_av = {_k(r.zone): r for r in reply.avoid}
    avoid = {_k(r.zone): r for r in cur.avoid}
    for k in b_av:
        if k not in r_av:
            avoid.pop(k, None)
    for k, r in r_av.items():
        if b_av.get(k) != r:
            avoid[k] = r
    stay_changed = reply.return_to_base != base.return_to_base
    rtb = reply.return_to_base if stay_changed else cur.return_to_base
    return MissionSpec(tuple(targets), tuple(avoid.values()), rtb)


def apply_parts(cur: MissionSpec, targets: list[str], avoid: list[AvoidRule], stay: bool) -> MissionSpec:
    """Add what the basic (no-AI) parser found to the current mission."""
    have = {_k(t) for t in cur.targets}
    new_t = list(cur.targets) + [t for t in targets if _k(t) not in have]
    av = {_k(r.zone): r for r in cur.avoid}
    for r in avoid:
        av[_k(r.zone)] = r
    return MissionSpec(tuple(new_t), tuple(av.values()), cur.return_to_base and not stay)


# --- the session --------------------------------------------------------------------------------

@dataclass
class Config:
    model: str = MODELS[0]
    mode: str = "ours"
    capacity: float = 40.0
    wind_speed: float = 3.0
    wind_from: float = 270.0
    clock: str = "10:00"
    monitor: bool = True
    base_pad: str = "Football Ground"
    wait_for_ai: bool = True


class Session:
    def __init__(self, llm_factory=None, campus: Campus | None = None):
        self.lock = threading.RLock()
        self.base_campus = campus or load_campus()
        self.llm_factory = llm_factory or self._default_llm
        self.cfg = Config()
        self.speed = DEFAULT_SPEED
        self.draft = MissionSpec(())
        self.desired = self.draft
        self.gen = 0
        self.epoch = 0
        self.chat: list[dict] = []
        self.direct: dict | None = None
        self.token = 0                 # id of the chat request now being answered, 0 = none
        self.started = False
        self.paused = False
        self.resume_submit = False     # an edit made while paused, applied on resume
        self._audit_seen = 0
        self._rebuild()

    # --- construction helpers -------------------------------------------------------------------
    @staticmethod
    def _default_llm(model: str):
        return GeminiLLM(model) if model.startswith("gemini") else OllamaLLM(model)

    def _rebuild(self) -> None:
        cfg = self.cfg
        pad = self.base_campus.pads[cfg.base_pad]
        self.campus = dataclasses.replace(self.base_campus, base=pad)
        self.ctl = Controller(self.campus, capacity_wh=cfg.capacity, forecast=FORECAST,
                              actual=Wind(cfg.wind_speed, cfg.wind_from),
                              clock_min=parse_time(cfg.clock), monitor=cfg.monitor)
        self._audit_seen = len(self.ctl.audit)

    # --- state ----------------------------------------------------------------------------------
    @property
    def phase(self) -> str:
        if not self.started:
            return "DRAFT"
        st = self.ctl.state
        if st == "DONE":
            return "DONE"
        if st == "CRASHED":
            return "CRASHED"
        if st == "GROUNDED_NEEDS_PICKUP":
            return "GROUNDED"
        return "PAUSED" if self.paused else "FLYING"

    @property
    def holding(self) -> bool:
        """The drone waits (hovers) while the AI reads a message."""
        return bool(self.token) and self.started and self.cfg.wait_for_ai and self.cfg.mode != "direct"

    def say(self, who: str, text: str, **extra) -> None:
        with self.lock:
            self.chat.append({"who": who, "text": text, "clock": self.ctl.hhmm(), **extra})

    # --- settings -------------------------------------------------------------------------------
    def configure(self, **kw) -> None:
        """Change settings. Only in DRAFT: a flight is never re-configured under the operator."""
        with self.lock:
            if self.started:
                raise SessionError("Settings are locked after Start. Press Reset to change them.")
            cfg = dataclasses.replace(self.cfg)
            for k, v in kw.items():
                if v is None:
                    continue
                if k == "model" and v in MODELS:
                    cfg.model = v
                elif k == "mode" and v in MODES:
                    cfg.mode = v
                elif k == "capacity":
                    cfg.capacity = min(80.0, max(5.0, float(v)))
                elif k == "wind_speed":
                    cfg.wind_speed = min(9.0, max(0.0, float(v)))
                elif k == "wind_from":
                    cfg.wind_from = float(v) % 360
                elif k == "clock":
                    if parse_time(str(v)) is None:
                        raise SessionError("Clock must look like 10:00")
                    cfg.clock = str(v)
                elif k == "monitor":
                    cfg.monitor = bool(v)
                elif k == "wait_for_ai":
                    cfg.wait_for_ai = bool(v)
                elif k == "base_pad":
                    if v not in self.base_campus.pads:
                        raise SessionError("The base must be one of the landing pads.")
                    cfg.base_pad = v
                else:
                    raise SessionError(f"Unknown or invalid setting: {k}")
            self.cfg = cfg
            self.direct = None
            self._rebuild()

    def set_speed(self, speed: int) -> None:
        with self.lock:
            if speed not in SPEEDS:
                raise SessionError(f"Speed must be one of {SPEEDS}.")
            self.speed = speed

    # --- editing the mission --------------------------------------------------------------------
    def _place(self, name: str):
        p = self.campus.find(str(name))
        if p is None or p.name in self.campus.permanent_no_fly:
            raise SessionError("That place can't be used.")
        return p

    def _editable(self) -> None:
        if self.phase in ("DONE", "GROUNDED", "CRASHED"):
            raise SessionError("This flight has ended. Press Reset to start another.")

    def edit(self, op: str, place: str | None = None, value: bool | None = None) -> str:
        """Click edits. Returns a short confirmation for the operator."""
        with self.lock:
            self._editable()
            cur = self.desired
            if op == "clear":
                if self.started:
                    raise SessionError("Can't clear a mission in flight.")
                new = MissionSpec(())
                msg = "Mission cleared."
            elif op == "stay":
                new = dataclasses.replace(cur, return_to_base=not bool(value))
                msg = "The drone will stay at the last place." if value else "The drone will return to base."
            else:
                p = self._place(place)
                if op == "target_add":
                    if any(_k(t) == _k(p.name) for t in cur.targets):
                        raise SessionError(f"{p.name} is already a target.")
                    if len(cur.targets) >= MAX_TARGETS:
                        raise SessionError(f"At most {MAX_TARGETS} targets.")
                    new = dataclasses.replace(cur, targets=cur.targets + (p.name,))
                    msg = f"Target added: {p.name}."
                elif op == "target_remove":
                    new = dataclasses.replace(cur, targets=tuple(t for t in cur.targets if _k(t) != _k(p.name)))
                    msg = f"Target removed: {p.name}."
                elif op == "avoid_add":
                    if len(cur.avoid) >= MAX_AVOID:
                        raise SessionError("Too many restricted areas.")
                    rules = tuple(r for r in cur.avoid if _k(r.zone) != _k(p.name)) + (AvoidRule(p.name, None, None),)
                    new = dataclasses.replace(cur, avoid=rules)
                    msg = f"Restricted: {p.name}."
                elif op == "avoid_remove":
                    new = dataclasses.replace(cur, avoid=tuple(r for r in cur.avoid if _k(r.zone) != _k(p.name)))
                    msg = f"No longer restricted: {p.name}."
                else:
                    raise SessionError("Unknown edit.")
            self._commit(new, source="click", text=msg)
            return msg

    def _commit(self, new: MissionSpec, *, source: str, text: str) -> None:
        """Validate and apply a new desired mission. Before Start only the draft changes."""
        if self.started:
            chk = validate_spec(new, self.campus, now_min=self.ctl.clock(),
                                horizon_min=horizon_min(self.cfg.capacity), position=self.ctl.pos,
                                visited=frozenset(self.ctl.visited))
            if not chk.ok:
                raise SessionError("; ".join(chk.errors))
            self.desired = new
            if self.paused:
                self.resume_submit = True      # "queued: applies on resume"
            else:
                self.ctl.submit(Command("order", new, f"{source}: {text}"))
        else:
            if new.targets:
                chk = validate_spec(new, self.campus, now_min=self.ctl.clock(),
                                    horizon_min=horizon_min(self.cfg.capacity))
                if not chk.ok:
                    raise SessionError("; ".join(chk.errors))
            self.draft = self.desired = new

    # --- flight control -------------------------------------------------------------------------
    def start(self) -> None:
        with self.lock:
            if self.started:
                raise SessionError("Already started.")
            if self.token:
                raise SessionError("The AI is still reading your message. Try again in a moment.")
            spec = self.desired
            if not spec.targets:
                raise SessionError("Add at least one target first (click a building or type an order).")
            chk = validate_spec(spec, self.campus, now_min=self.ctl.clock(),
                                horizon_min=horizon_min(self.cfg.capacity))
            if not chk.ok:
                raise SessionError("; ".join(chk.errors))
            n = len(self.ctl.audit)
            self.ctl.submit(Command("order", spec, "start"))
            self.ctl.step()
            if self.ctl.state not in FLYING:
                why = [e["msg"] for e in self.ctl.audit[n:] if e["kind"] in ("replan_failed", "order_rejected")]
                self._rebuild()   # the rejected attempt must leave no trace in the controller
                raise SessionError(why[0] if why else "The shield did not accept this mission.")
            self.started, self.paused, self.resume_submit = True, False, False

    def pause(self) -> None:
        with self.lock:
            if self.phase != "FLYING":
                raise SessionError("Nothing to pause.")
            self.paused = True

    def resume(self) -> None:
        with self.lock:
            if self.phase != "PAUSED":
                raise SessionError("Not paused.")
            self.paused = False
            if self.resume_submit:
                self.resume_submit = False
                self.ctl.submit(Command("order", self.desired, "queued edit applied on resume"))

    def reset(self) -> None:
        """Back to DRAFT. Settings and the draft mission are kept; anything pending is dropped."""
        with self.lock:
            self.gen += 1
            self.epoch += 1
            self.token = 0
            self.started = self.paused = self.resume_submit = False
            self.desired = self.draft
            self.chat, self.direct = [], None
            self._rebuild()

    def emergency(self, kind: str) -> None:
        """'return' or 'land': works without the AI, only while flying."""
        with self.lock:
            if self.phase not in ("FLYING", "PAUSED"):
                raise SessionError("Nothing is flying.")
            self.paused = False
            self.ctl.submit(Command(kind, None, f"operator: {kind} now"))

    # --- simulation tick ------------------------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            if not self.started or self.paused or self.holding:
                return
            ctl = self.ctl
            if ctl.state in FLYING or ctl.state == "LANDED_WAIT" or ctl.inbox:
                ctl.step()
            for e in ctl.audit[self._audit_seen:]:
                if e["kind"] in ("order_rejected", "replan_failed"):
                    self.say("system", e["msg"], level="bad")
            self._audit_seen = len(ctl.audit)

    # --- chat -----------------------------------------------------------------------------------
    def handle(self, text: str) -> None:
        """Understand one operator message and apply it. Blocks while the AI answers."""
        text = text.strip()[:2000]
        if not text:
            return
        with self.lock:
            if self.token:
                raise SessionError("Still working on your previous message.")
            gen = self.gen
            self.token = token = int(time.monotonic() * 1e6) or 1
            base = self.desired
            visited = [self.campus.places[v].name for v in self.ctl.visited if v in self.campus.places]
            hh, cfg, campus = self.ctl.hhmm(), self.cfg, self.campus
            self.say("operator", text)
        try:
            if cfg.mode == "direct":
                self._direct(text, hh, gen)
                return
            interp = self._interpret(text, base, visited, hh, cfg, campus)
            with self.lock:
                if gen != self.gen:
                    return                      # the session was reset while the AI was thinking
                self._apply_interp(interp, text, base)
        finally:
            with self.lock:
                if self.token == token:
                    self.token = 0

    def _interpret(self, text, base, visited, hh, cfg, campus):
        if cfg.mode == "classic":
            return ("regex", None)
        llm = self.llm_factory(cfg.model)
        deadline = AI_DEADLINE_S["gemini" if cfg.model.startswith("gemini") else "ollama"]
        r = interpret(text, llm, campus, current=base if base.targets else None, visited=visited,
                      hhmm=hh, deadline_s=deadline)
        return ("ai", r)

    def _apply_interp(self, interp, text: str, base: MissionSpec) -> None:
        kind, r = interp
        try:
            if kind == "ai" and r.status == "keyword":
                if not self.started:
                    self.say("ai", "Nothing is flying yet.", status="keyword")
                    return
                self.ctl.submit(r.command)
                self.say("ai", r.message, status="keyword")
                return
            if kind == "ai" and r.status in ("applied",):
                new = merge_reply(base, r.command.spec, self.desired)
                self._commit(new, source="chat", text=text)
                self.say("ai", "Understood.", status="applied", spec=new.to_json(),
                         latency_ms=None if r.latency_ms is None else round(r.latency_ms))
                return
            if kind == "ai" and r.status in ("refused", "clarify"):
                self.say("ai", r.message, status=r.status)
                return
            # AI unavailable (error/timeout) or classic mode: the basic parser takes over
            fallback = kind == "ai"
            parser = RegexParser(self.campus)
            from .operator import keyword_command
            kw = keyword_command(text)
            if kw is not None:
                if not self.started:
                    self.say("ai", "Nothing is flying yet.", status="keyword")
                else:
                    self.ctl.submit(kw)
                    self.say("ai", f"'{kw.kind}' command (handled without the AI)", status="keyword")
                return
            targets, avoid, stay = parser.parse_parts(text, split_and=True)
            if not (targets or avoid or stay):
                why = r.message if fallback else "I couldn't find a place name in that."
                self.say("ai", why, status="error" if fallback else "clarify")
                return
            new = apply_parts(self.desired, targets, avoid, stay)
            self._commit(new, source="chat", text=text)
            self.say("ai", ("AI unavailable: basic parser used. " if fallback else "Basic parser: ") + "Understood.",
                     status="fallback" if fallback else "applied", spec=new.to_json())
        except SessionError as e:
            self.say("system", f"Not applied: {e}", level="bad")

    def _direct(self, text: str, hh: str, gen: int) -> None:
        from .bench import ask_direct, fly_direct
        llm = self.llm_factory(self.cfg.model)
        d = ask_direct(llm, text, hh)
        with self.lock:
            if gen != self.gen:
                return
            if d["refused"] or not d["waypoints"]:
                self.say("system", f"AI-direct: no route ({d['error'] or 'refused'})")
                return
        res = fly_direct(d["waypoints"], parse_time(hh), None, cap=self.cfg.capacity,
                         actual=Wind(self.cfg.wind_speed, self.cfg.wind_from))
        with self.lock:
            if gen != self.gen:
                return
            self.direct = {"route": d["waypoints"], **{k: res.get(k) for k in
                           ("perm_violation", "fence_violation", "crashed", "energy_left", "malformed")}}
            bad = [k for k in ("perm_violation", "fence_violation", "crashed", "malformed") if res.get(k)]
            self.say("system", "AI-direct route (no shield, no animation): " + (", ".join(bad) if bad else "no violation"),
                     level="bad" if bad else "ok")

    # --- what the browser sees ------------------------------------------------------------------
    def zones_signature(self) -> str:
        spec = self.ctl.spec if self.started and self.ctl.spec else self.desired
        air = Airspace.for_spec(self.campus, spec, self.ctl.clock(),
                                self.ctl.clock() + horizon_min(self.cfg.capacity))
        return ",".join(sorted(n for n, _, _ in air.avoid_polys))

    def snapshot(self, since: dict | None = None) -> dict:
        """Delta feed: only what changed since the client's offsets (epoch mismatch = send everything)."""
        since = since or {}
        with self.lock:
            fresh = since.get("epoch") != self.epoch
            g = (lambda k: 0 if fresh else int(since.get(k) or 0))
            ctl, snap = self.ctl, self.ctl.snapshot()
            out = {
                "epoch": self.epoch, "reset": fresh, "phase": self.phase, "paused": self.paused,
                "holding": self.holding, "busy": bool(self.token), "speed": self.speed,
                "queued": len(ctl.inbox) + (1 if self.resume_submit else 0),
                "config": dataclasses.asdict(self.cfg), "state": snap["state"], "t": snap["t"],
                "clock": snap["clock"], "pos": snap["pos"], "battery_wh": snap["battery_wh"],
                "battery_pct": snap["battery_pct"], "visited": snap["visited"],
                "route_index": snap["route_index"], "version": snap["version"],
                "desired": self.desired.to_json(), "base": list(self.campus.base),
                "visited_names": [self.campus.places[v].name for v in ctl.visited if v in self.campus.places],
                "targets": [{"name": t, "center": [round(self.campus.find(t).center[0], 1), round(self.campus.find(t).center[1], 1)],
                             "done": _k(t) in ctl.visited} for t in self.desired.targets],
                "chat": self.chat[g("chat"):], "chat_total": len(self.chat),
                "audit": ctl.audit[g("audit"):], "audit_total": len(ctl.audit),
                "track": [[round(x, 1), round(y, 1), s] for _, x, y, s in ctl.track[g("track"):]],
                "track_total": len(ctl.track), "direct": self.direct,
            }
            if fresh or since.get("route_rev") != snap["version"]:
                out["route"] = snap["route"]
            out["route_rev"] = snap["version"]
            sig = self.zones_signature()
            out["zones_sig"] = sig
            if fresh or since.get("zones_sig") != sig:
                spec = ctl.spec if self.started and ctl.spec else self.desired
                air = Airspace.for_spec(self.campus, spec, ctl.clock(), ctl.clock() + horizon_min(self.cfg.capacity))
                out["zones"] = [{"name": n, "poly": [[round(x, 1), round(y, 1)] for x, y in poly], "buffer": b}
                                for n, poly, b in air.avoid_polys]
            return out
