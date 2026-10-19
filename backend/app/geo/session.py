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
import math
import re
import threading
import time
from dataclasses import dataclass

from .campus import Campus, Place, load_campus, norm
from .controller import FLYING, Command, Controller
from .flight import AVOID_BUFFER, HOVER_W, Airspace, Wind, horizon_min
from .llm import GeminiLLM, OllamaLLM
from .operator import RegexParser, interpret, strict_keyword
from .spec import AvoidRule, MissionSpec, parse_time, validate_spec

SPEEDS = (1, 2, 5)
DEFAULT_SPEED = 2
MODELS = ["gemini-3.5-flash-lite", "qwen2.5:7b", "qwen2.5:3b"]
MODES = ("ours", "classic", "direct")
MAX_AVOID = 20
MAX_CIRCLES = 8
CIRCLE_R = (20.0, 300.0)
END_WORDS = re.compile(r"^\s*(please\s+)?(end|finish|stop|cancel)\s+(the\s+|this\s+)?(mission|flight)\W*$", re.I)
LAND_AT = re.compile(r"^\s*(?:please\s+)?land\s+(?:and\s+(?:stop|stay|wait)\s+)?(?:at|on|in)\s+(?:the\s+)?(.+?)\W*$", re.I)
STAY_RE = re.compile(r"\b(?:don'?t|do not|never|no need to)\s+(?:return|go|come|fly)\s+(?:back|home|to\s+base)(?:\s+to\s+base)?\b"
                     r"|\b(?:stop|stay|remain|finish|end|wait)\s+(?:at|there|here)\b|\bland\s+and\s+(?:stop|stay)\b", re.I)
DIVERT_RE = re.compile(r"\b(divert|detour|go first|first go|go now|right now|immediately|urgent)\b", re.I)
ONLY_WORDS = re.compile(r"\b(only|instead)\b", re.I)
HOLD_MAX_TICKS = 20      # the drone waits for the AI at most this many simulated seconds
MAX_CHAT = 400
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
        self.models = list(MODELS)     # the public demo narrows this to the hosted AI only
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
        self.end_after_land = False    # 'End mission' while airborne: land first, then finish
        self.safety_pending = False    # Land/Return/End is under way: orders wait until the drone is down
        self.hold_ticks = 0
        self.chat_dropped = 0
        self._acc = 0.0                # real seconds * speed not yet turned into simulation ticks
        self.circles: list[dict] = []  # areas the operator drew: name, x, y, r
        self.circle_n = 0
        self._audit_seen = 0
        self._rebuild()

    # --- construction helpers -------------------------------------------------------------------
    @staticmethod
    def _default_llm(model: str):
        return GeminiLLM(model) if model.startswith("gemini") else OllamaLLM(model)

    def _rebuild(self) -> None:
        cfg = self.cfg
        pad = self.base_campus.pads[cfg.base_pad]
        # own copy of the places, so a drawn area never leaks into the shared campus
        self.campus = dataclasses.replace(self.base_campus, base=pad, places=dict(self.base_campus.places))
        self._install_circles()
        self.ctl = Controller(self.campus, capacity_wh=cfg.capacity, forecast=FORECAST,
                              actual=Wind(cfg.wind_speed, cfg.wind_from),
                              clock_min=parse_time(cfg.clock), monitor=cfg.monitor)
        self._audit_seen = len(self.ctl.audit)

    # --- areas the operator draws ---------------------------------------------------------------
    @staticmethod
    def _circle_poly(c: dict) -> tuple:
        return tuple((c["x"] + c["r"] * math.cos(2 * math.pi * i / 24), c["y"] + c["r"] * math.sin(2 * math.pi * i / 24))
                     for i in range(24))

    def _install_circles(self) -> None:
        for c in self.circles:
            self.campus.places[norm(c["name"])] = Place(c["name"], "area", self._circle_poly(c), (c["x"], c["y"]))

    def add_circle(self, x, y, r) -> str:
        """A round area the operator drew. It is a normal place afterwards: it can be restricted or made a
        target, and every such change still goes through the shield."""
        with self.lock:
            for v in (x, y, r):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise SessionError("The area needs numbers for its centre and radius.")
            if not CIRCLE_R[0] <= r <= CIRCLE_R[1]:
                raise SessionError(f"The radius must be between {CIRCLE_R[0]:.0f} and {CIRCLE_R[1]:.0f} m.")
            if not self.campus.in_fence((x, y)):
                raise SessionError("The centre is outside the flight area.")
            if len(self.circles) >= MAX_CIRCLES:
                raise SessionError(f"At most {MAX_CIRCLES} drawn areas. Delete one first.")
            self.circle_n += 1
            name = f"Area {self.circle_n}"
            if self.campus.find(name) is not None:
                raise SessionError("That name is taken.")
            self.circles.append({"name": name, "x": round(float(x), 1), "y": round(float(y), 1), "r": round(float(r), 1)})
            self._install_circles()
            return name

    def remove_circle(self, name) -> str:
        with self.lock:
            c = next((c for c in self.circles if _k(c["name"]) == _k(str(name))), None)
            if c is None:
                raise SessionError("That is not a drawn area.")
            used = [t for t in self.desired.targets] + [a.zone for a in self.desired.avoid]
            if self.started and self.ctl.spec:
                used += list(self.ctl.spec.targets) + [a.zone for a in self.ctl.spec.avoid]
            if any(_k(u) == _k(c["name"]) for u in used):
                raise SessionError(f"{c['name']} is still part of the mission. Remove it from the mission first.")
            self.circles.remove(c)
            self.campus.places.pop(norm(c["name"]), None)
            return f"Deleted {c['name']}."

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
        if st == "LANDED_WAIT":
            return "WAITING"
        return "PAUSED" if self.paused else "FLYING"

    @property
    def holding(self) -> bool:
        """The drone waits (hovers) while the AI reads a message."""
        return (bool(self.token) and self.started and self.cfg.wait_for_ai and self.cfg.mode != "direct"
                and not self.safety_pending)

    def say(self, who: str, text: str, **extra) -> None:
        with self.lock:
            self.chat.append({"who": who, "text": text, "clock": self.ctl.hhmm(), **extra})
            if len(self.chat) > MAX_CHAT:
                del self.chat[:100]
                self.chat_dropped += 100

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
                if k in ("capacity", "wind_speed", "wind_from"):
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                        raise SessionError(f"{k.replace('_', ' ').capitalize()} must be a number.")
                elif k in ("monitor", "wait_for_ai") and not isinstance(v, bool):
                    raise SessionError(f"{k} must be true or false.")
                if k == "model" and v in self.models:
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
                    if not isinstance(v, str) or not re.fullmatch(r"[0-9]{1,2}:[0-9]{2}", v) or parse_time(v) is None:
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
                elif k == "model":
                    raise SessionError("Model must be one of: " + ", ".join(self.models))
                elif k == "mode":
                    raise SessionError("Mode must be one of: " + ", ".join(MODES))
                else:
                    raise SessionError(f"Unknown or invalid setting: {k}")
            self.cfg = cfg
            self.direct = None
            self._rebuild()

    def set_speed(self, speed: int) -> None:
        with self.lock:
            if type(speed) is not int or speed not in SPEEDS:
                raise SessionError(f"Speed must be one of {SPEEDS}.")
            self.speed = speed

    # --- editing the mission --------------------------------------------------------------------
    def _resolve_name(self, raw: str):
        """A place from the operator's words: exact (normalised) match first, then a close spelling."""
        import difflib
        p = self.campus.find(raw)
        if p is not None:
            return p
        names = {_k(pl.name): pl for pl in self.campus.places.values()}
        hit = difflib.get_close_matches(_k(raw), list(names), n=1, cutoff=0.8)
        return names[hit[0]] if hit else None

    def _place(self, name: str):
        p = self.campus.find(str(name))
        if p is None or p.name in self.campus.permanent_no_fly:
            raise SessionError("That place can't be used.")
        return p

    def _editable(self) -> None:
        if self.phase in ("GROUNDED", "CRASHED"):
            raise SessionError("This flight has ended. Press Reset to start another.")

    def working_spec(self) -> MissionSpec:
        """The mission edits apply to. After a completed mission the finished targets are dropped (a new
        order starts a new leg from where the drone landed); restrictions and the return setting stay."""
        if self.started and self.ctl.state == "DONE":
            return MissionSpec((), self.desired.avoid, self.desired.return_to_base)
        return self.desired

    def edit(self, op: str, place: str | None = None, value: bool | None = None) -> str:
        """Click edits. Returns a short confirmation for the operator."""
        with self.lock:
            self._editable()
            if op == "circle_remove":
                return self.remove_circle(place)
            cur = self.working_spec()
            if op == "clear":
                if self.started:
                    raise SessionError("Can't clear a mission in flight.")
                new = MissionSpec(())
                msg = "Mission cleared."
            elif op == "stay":
                if not isinstance(value, bool):
                    raise SessionError("Stay needs true or false.")
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
            if self.safety_pending and self.ctl.state in FLYING:
                raise SessionError("The drone is landing or returning on your command. Give the new order once it is down.")
            self.safety_pending = False
            new_leg = self.ctl.state == "DONE"
            if new_leg and not new.targets:
                self.desired = new            # restrictions / return setting for the next leg; nothing to fly yet
                return
            chk = validate_spec(new, self.campus, now_min=self.ctl.clock(),
                                horizon_min=horizon_min(self.cfg.capacity), position=self.ctl.pos,
                                visited=frozenset() if new_leg else frozenset(self.ctl.visited))
            if not chk.ok:
                raise SessionError("; ".join(chk.errors))
            if new_leg:
                self._dry_run_new_leg(new)
                self.ctl.reopen()   # landed and idle: take the new order from here
            self.end_after_land = False
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

    def _dry_run_new_leg(self, new: MissionSpec) -> None:
        """A finished drone only takes an order the shield accepts; try it on a copy first."""
        import copy
        trial = copy.deepcopy(self.ctl)
        trial.reopen()
        n = len(trial.audit)
        trial.submit(Command("order", new, "dry run"))
        trial.step()
        if trial.state not in FLYING:
            why = [e["msg"] for e in trial.audit[n:] if e["kind"] in ("replan_failed", "order_rejected")]
            raise SessionError(why[0] if why else "The shield did not accept this order.")

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
            self.safety_pending, self.hold_ticks = False, 0

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
            self.started = self.paused = self.resume_submit = self.end_after_land = self.safety_pending = False
            self.hold_ticks = self.chat_dropped = 0
            self.desired = self.draft
            self.chat, self.direct = [], None
            self._rebuild()

    def _zone_spec(self) -> MissionSpec:
        """The rules the map should show: the flight's own, or the operator's after the mission is done."""
        if self.started and self.ctl.spec and self.ctl.state != "DONE":
            return self.ctl.spec
        return self.desired

    def emergency(self, kind: str, pad: str | None = None) -> None:
        """'return' or 'land' (optionally at a chosen pad): works without the AI, only while flying."""
        with self.lock:
            if kind not in ("return", "land"):
                raise SessionError("Unknown command.")
            if self.phase == "WAITING":
                raise SessionError("The drone is already on the ground.")
            if self.phase not in ("FLYING", "PAUSED"):
                raise SessionError("Nothing is flying.")
            if pad is not None:
                if kind != "land":
                    raise SessionError("Only a landing can name a pad.")
                if pad not in self.campus.pads:
                    raise SessionError("That is not a landing pad.")
                plan, reason = self.ctl.land_plan(pad)
                if plan is None:                  # refused here, so nothing changes and the drone flies on
                    raise SessionError(f"Can't land at {pad}: {reason}.")
            self.paused = False
            self.resume_submit = False
            self.end_after_land = False
            self.safety_pending = True
            self.gen += 1                    # answers to messages sent before this command are dropped
            self.ctl.submit(Command(kind, None, f"operator: {kind} now" + (f" at {pad}" if pad else ""), pad))

    def _drop_pending(self) -> None:
        """Ending a mission drops the targets not reached; the reached ones stay for the record."""
        done = tuple(t for t in self.desired.targets if _k(t) in self.ctl.visited)
        self.desired = dataclasses.replace(self.desired, targets=done)

    def end_mission(self) -> None:
        """The operator ends the mission: land on the nearest pad if airborne, then finish (DONE).
        A finished drone stays on the ground and accepts a new order. Works without the AI."""
        with self.lock:
            if not self.started:
                raise SessionError("There is no mission to end.")
            st = self.ctl.state
            if st in FLYING:
                self.paused = False
                self.resume_submit = False
                self.safety_pending = True
                self.gen += 1
                self.end_after_land = True
                self.ctl.submit(Command("land", None, "operator: end mission"))
            elif st == "LANDED_WAIT":
                self.paused = False
                self.resume_submit = False
                self.gen += 1                # an AI reply still on its way must not start a new leg
                self.ctl._set_state("DONE", "mission ended by the operator")
                self._drop_pending()
            elif st == "DONE":
                raise SessionError("The mission is already complete.")
            else:
                raise SessionError("This flight has ended. Press Reset to start another.")

    def advance(self, real_dt: float) -> None:
        """Called by the server loop: turns real time into simulation ticks at this session's speed."""
        with self.lock:
            self._acc = min(self._acc + real_dt * self.speed, 12.0)
            n = int(self._acc)
            self._acc -= n
        for _ in range(n):
            self.tick()

    # --- simulation tick ------------------------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            if not self.started or self.paused:
                return
            ctl = self.ctl
            if self.holding and not self.safety_pending and self.hold_ticks < HOLD_MAX_TICKS:
                self.hold_ticks += 1          # hovering costs battery and time, so the numbers stay honest
                if ctl.state in FLYING:
                    ctl._drain(HOVER_W, ctl.dt)
                    ctl.t += ctl.dt
                    ctl.track.append((ctl.t, *ctl.pos, ctl.state))
                return
            if ctl.state in FLYING or ctl.state == "LANDED_WAIT" or ctl.inbox:
                ctl.step()
            for e in ctl.audit[self._audit_seen:]:
                if e["kind"] in ("order_rejected", "replan_failed"):
                    self.say("system", e["msg"], level="bad")
            self._audit_seen = len(ctl.audit)
            if self.safety_pending and ctl.state not in FLYING:
                self.safety_pending = False
            if self.end_after_land:
                if ctl.state == "LANDED_WAIT":
                    ctl._set_state("DONE", "mission ended by the operator")
                    self._drop_pending()
                    self.end_after_land = False
                elif ctl.state in ("DONE", "GROUNDED_NEEDS_PICKUP", "CRASHED"):
                    self.end_after_land = False

    # --- chat -----------------------------------------------------------------------------------
    def handle(self, text: str) -> None:
        """Understand one operator message and apply it. Blocks while the AI answers."""
        text = text.strip()[:2000]
        if not text:
            return
        if len(text) <= 60 and END_WORDS.match(text):
            with self.lock:
                self.say("operator", text)
                try:
                    self.end_mission()
                    self.say("ai", "Ending the mission: landing, then finishing. (handled without the AI)", status="keyword")
                except SessionError as e:
                    self.say("system", f"Not applied: {e}", level="bad")
            return
        m = LAND_AT.match(text) if len(text) <= 70 else None
        land_raw = None
        if m:
            place = self.campus.find(m.group(1))        # an exact name needs no AI
            if place is not None:
                with self.lock:
                    self.say("operator", text)
                    self._land_at(place)
                return
            land_raw = m.group(1)                       # otherwise the AI reads the (maybe misspelt) place name below
        kind = strict_keyword(text)
        if kind is not None:
            with self.lock:
                self.say("operator", text)
                try:
                    if not self.started:
                        self.say("ai", "Nothing is flying yet.", status="keyword")
                    else:
                        self.emergency(kind)
                        self.say("ai", f"'{kind}' command (handled without the AI)", status="keyword")
                except SessionError as e:
                    self.say("system", f"Not applied: {e}", level="bad")
            return
        with self.lock:
            if self.token:
                raise SessionError("Still working on your previous message.")
            self._editable()   # a finished flight takes no orders: Reset first
            gen, epoch0 = self.gen, self.epoch
            self.token = token = int(time.monotonic() * 1e6) or 1
            self.hold_ticks = 0
            base = self.working_spec()
            visited = [self.campus.places[v].name for v in self.ctl.visited if v in self.campus.places]
            if self.ctl.state == "DONE":
                visited = []
            hh, cfg, campus = self.ctl.hhmm(), self.cfg, self.campus
            self.say("operator", text)
        try:
            if cfg.mode == "direct":
                self._direct(text, hh, gen)
                return
            if land_raw:      # "land at <place>": the AI only has to find which place is meant
                interp = self._interpret(f"inspect {land_raw}", MissionSpec(()), [], hh, cfg, campus)
            else:
                interp = self._interpret(text, base, visited, hh, cfg, campus)
            with self.lock:
                if gen != self.gen:
                    if self.epoch == epoch0:
                        self.say("system", "Not applied: you gave a landing command while the AI was reading this.", level="bad")
                    return                      # reset or Land/Return/End happened while the AI was thinking
                if land_raw:
                    self._apply_land(interp, land_raw)
                else:
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
        known = bool(base.targets or base.avoid or not base.return_to_base)
        r = interpret(text, llm, campus, current=base if known else None, visited=visited,
                      hhmm=hh, deadline_s=deadline, keywords=False)
        return ("ai", r)

    @staticmethod
    def _short_reason(msg: str | None) -> str:
        """Why the AI did not answer, in a few safe words: no keys, no addresses, no stack traces."""
        m = re.sub(r"^AI error \(|\): message not applied\.?$", "", str(msg or "no answer"))
        m = re.sub(r"AIza[0-9A-Za-z_\-]{10,}|key=[^&\s]+|https?://\S+", "[hidden]", m)
        return m.strip()[:140] or "no answer"

    @staticmethod
    def _prioritize(new: MissionSpec, old: MissionSpec, text: str) -> MissionSpec:
        """"Divert to X" / "go to X now": the new stops come first, the rest of the mission follows."""
        if not DIVERT_RE.search(text):
            return new
        old_keys = {_k(t) for t in old.targets}
        added = [t for t in new.targets if _k(t) not in old_keys]
        if not added:
            return new
        return dataclasses.replace(new, targets=tuple(added + [t for t in new.targets if _k(t) in old_keys]))

    def _land_at(self, place) -> None:
        """Land at a named place: only an approved pad qualifies, and the shield still checks the route."""
        try:
            if place.name in self.campus.pads:
                if not self.started:
                    self.say("ai", "Nothing is flying yet.", status="keyword")
                else:
                    self.emergency("land", place.name)
                    self.say("ai", f"Landing at {place.name}.", status="keyword")
            else:
                pads = ", ".join(sorted(self.campus.pads))
                self.say("system", f"{place.name} is not a landing pad, so the drone cannot land there. "
                         f"Pads: {pads}. Say \"land now\" for the nearest pad, or \"land at <pad>\".", level="bad")
        except SessionError as e:
            self.say("system", f"Not applied: {e}", level="bad")

    def _apply_land(self, interp, land_raw: str) -> None:
        """Which place did 'land at <land_raw>' mean? The AI reads the spelling; offline, a close-spelling match."""
        kind, r = interp
        names: list[str] = []
        if kind == "ai" and r.status == "applied":
            names = list(r.command.spec.targets)
        elif kind == "ai" and r.status in ("refused", "clarify"):
            self.say("ai", r.message, status=r.status)
            return
        else:
            place = self._resolve_name(land_raw)
            names = [place.name] if place else RegexParser(self.campus).parse_parts(f"inspect {land_raw}")[0]
        place = self.campus.find(names[0]) if names else None
        if place is None:
            self.say("system", f"I couldn't tell which place \"{land_raw}\" is. Click a pad on the map and press Land here.", level="bad")
            return
        self._land_at(place)

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
                new = merge_reply(base, r.command.spec, self.working_spec())
                new = self._prioritize(new, base, text)
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
            targets, avoid, stay = parser.parse_parts(text, split_and=True)
            stay = stay or bool(STAY_RE.search(text))
            if not (targets or avoid or stay):
                why = ("The AI is offline and the basic parser could not read that. Name a place (for example "
                       "\"inspect the library\"), or use the buttons on the map." if fallback
                       else "I couldn't find a place name in that.")
                self.say("ai", why, status="error" if fallback else "clarify")
                return
            cur = self.working_spec()
            if targets and ONLY_WORDS.search(text):
                cur = dataclasses.replace(cur, targets=())     # "only" / "instead": replace, don't append
            new = self._prioritize(apply_parts(cur, targets, avoid, stay), cur, text)
            self._commit(new, source="chat", text=text)
            note = f"AI unavailable ({self._short_reason(r.message)}): basic parser used. " if fallback else "Basic parser: "
            self.say("ai", note + "Understood.",
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
        spec = self._zone_spec()
        air = Airspace.for_spec(self.campus, spec, self.ctl.clock(),
                                self.ctl.clock() + horizon_min(self.cfg.capacity))
        return ",".join(sorted(n for n, _, _ in air.avoid_polys))

    def export_data(self) -> dict:
        """Plain data for the downloads (see exports.py)."""
        from . import exports
        from .version import VERSION
        with self.lock:
            ctl, snap = self.ctl, self.snapshot()
            return {
                "version": VERSION, "exported_utc": exports.stamp(), "phase": self.phase,
                "config": dataclasses.asdict(self.cfg), "mission": self.desired.to_json(),
                "audit": [dict(a) for a in ctl.audit], "chat": [dict(c) for c in self.chat],
                "track": [(t, x, y, st) for t, x, y, st in ctl.track], "fence": self.campus.fence,
                "no_fly": [{"name": n, "poly": list(poly)} for n, (poly, _) in self.campus.permanent_no_fly.items()],
                "zones": snap["zones"], "pads": dict(self.campus.pads), "targets": snap["targets"],
                "t": ctl.t, "battery_wh": snap["battery_wh"], "battery_pct": snap["battery_pct"],
                "visited": snap["visited_names"],
            }

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
                "chat": self.chat[max(0, g("chat") - self.chat_dropped):], "chat_total": self.chat_dropped + len(self.chat),
                "audit": ctl.audit[g("audit"):], "audit_total": len(ctl.audit),
                "track": [[round(x, 1), round(y, 1), s] for _, x, y, s in ctl.track[g("track"):]],
                "track_total": len(ctl.track), "direct": self.direct,
                "circles": [{"name": c["name"], "center": [c["x"], c["y"]], "r": c["r"],
                             "poly": [[round(x, 1), round(y, 1)] for x, y in self._circle_poly(c)]} for c in self.circles],
            }
            if fresh or since.get("route_rev") != snap["version"]:
                out["route"] = snap["route"]
            out["route_rev"] = snap["version"]
            sig = self.zones_signature()
            out["zones_sig"] = sig
            if fresh or since.get("zones_sig") != sig:
                spec = self._zone_spec()
                air = Airspace.for_spec(self.campus, spec, ctl.clock(), ctl.clock() + horizon_min(self.cfg.capacity))
                out["zones"] = [{"name": n, "poly": [[round(x, 1), round(y, 1)] for x, y in poly], "buffer": b}
                                for n, poly, b in air.avoid_polys]
            return out
