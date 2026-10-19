"""Operator interface: plain-English text -> a controller Command.

Order of handling:
  1. Keyword fallback, no LLM: "return now/home", "land now", "abort". These always work,
     even when the model is slow or down.
  2. The LLM (any provider) turns the text into a full MissionSpec, with a hard deadline.
     A timeout, error, refusal or clarification never touches the controller: the drone
     keeps its current, already-verified state.
  3. `RegexParser` is the no-LLM baseline used in the benchmark (and as a UI option).
"""

from __future__ import annotations

import concurrent.futures as cf
import difflib
import re
from dataclasses import dataclass

from .campus import Campus, describe_places, norm
from .controller import Command
from .llm import CHANGE_SYSTEM, SPEC_SCHEMA, SPEC_SYSTEM, change_prompt, spec_prompt
from .spec import (AvoidRule, MissionSpec, NeedsClarification, SpecRefused, parse_time, spec_from_json)

KEYWORDS = [
    (re.compile(r"\b(return|come back|go)\s+(now|home|to base)\b|\bRTB\b", re.I), "return"),
    (re.compile(r"\b(land|abort)\b(\s+(now|immediately|here))?", re.I), "land"),
]
_POOL = cf.ThreadPoolExecutor(max_workers=2)


@dataclass
class Interpretation:
    command: Command | None        # None = nothing to apply
    status: str                    # applied | keyword | refused | clarify | error | timeout
    message: str                   # plain-English reply for the operator
    raw: str | None = None
    latency_ms: float | None = None


def keyword_command(text: str) -> Command | None:
    t = text.strip()
    if len(t) > 60:
        return None  # long messages go to the parser; keywords are for short commands
    for pat, kind in KEYWORDS:
        if pat.search(t):
            return Command(kind, None, text)
    return None


_STRICT = [
    (re.compile(r"^(please\s+)?(return(\s+(now|home|to\s+base))?|come\s+back(\s+(now|home))?|go\s+(home|back)|rtb)\W*$", re.I), "return"),
    (re.compile(r"^(please\s+)?(land(\s+(now|immediately|here))?|abort(\s+(now|mission))?)\W*$", re.I), "land"),
]


def strict_keyword(text: str) -> str | None:
    """Whole-message safety commands only ("land now", "return home"). A longer order that merely
    contains such words ("inspect X and return to base", "do not land") goes to the parser."""
    t = text.strip()
    if len(t) > 40:
        return None
    for pat, kind in _STRICT:
        if pat.match(t):
            return kind
    return None


def interpret(text: str, llm, campus: Campus, *, current: MissionSpec | None, visited: list[str],
              hhmm: str, deadline_s: float = 20.0, seed: int = 0, keywords: bool = True) -> Interpretation:
    text = text[:2000]
    kw = keyword_command(text) if keywords else None
    if kw is not None:
        return Interpretation(kw, "keyword", f"'{kw.kind}' command (handled without the AI)")
    if current is None:
        system, user = SPEC_SYSTEM, spec_prompt(campus, text, hhmm)
    else:
        system, user = CHANGE_SYSTEM, change_prompt(campus, current.to_json(), visited, text, hhmm)
    fut = _POOL.submit(llm.ask, system, user, SPEC_SCHEMA, seed)
    try:
        reply = fut.result(timeout=deadline_s)
    except cf.TimeoutError:
        return Interpretation(None, "timeout", f"AI did not answer within {deadline_s:.0f} s: message not applied")
    if reply.error or reply.data is None:
        return Interpretation(None, "error", f"AI error ({reply.error}): message not applied", reply.text,
                              reply.latency_ms)
    return _to_command(reply.data, text, reply.text, reply.latency_ms)


def _to_command(data: dict, text: str, raw, latency) -> Interpretation:
    try:
        spec = spec_from_json(data)
    except SpecRefused as e:
        return Interpretation(None, "refused", f"Refused: {e}", raw, latency)
    except NeedsClarification as e:
        return Interpretation(None, "clarify", f"Question: {e}", raw, latency)
    except ValueError as e:
        return Interpretation(None, "error", f"Could not understand the AI's answer ({e}): message not applied",
                              raw, latency)
    return Interpretation(Command("order", spec, text), "applied", "Parsed order", raw, latency)


class RegexParser:
    """No-LLM baseline: fuzzy place-name matching plus keyword and time patterns.

    Each clause is classified by the words around the place name: "avoid / stay away /
    don't fly over / keep clear / skip" marks a zone to avoid, everything else is a target
    (in order of mention). "don't come back / stay there / leave it at" disables return.
    """

    name = "regex"
    AVOID = re.compile(r"\b(avoid|away from|stay off|keep (clear|out)|don'?t (fly|go)|do not (fly|go)|"
                       r"never|skip|not over|no[- ]fly)\b", re.I)
    STAY = re.compile(r"\b(don'?t|do not|no need to) (come|return|fly) back\b|\bstay (there|at)\b|"
                      r"\bleave it (there|at)\b|\bend (the route )?at\b", re.I)
    TIME = re.compile(r"(?:between|from)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(?:and|to|-)\s+"
                      r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|until\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", re.I)

    def __init__(self, campus: Campus):
        self.c = campus
        self.names = [p["name"] for p in describe_places(campus)]

    def parse_parts(self, text: str, split_and: bool = False) -> "tuple[list[str], list[AvoidRule], bool]":
        """(targets in order, avoid rules, stay_at_end) found in the text. No validation.

        split_and=True (live fallback only) also splits '... and avoid X' into two clauses. The
        benchmarked baseline keeps it False, so its published numbers do not change."""
        if split_and:
            text = re.sub(r"\band\b(?=\s+(?:avoid|stay (?:away|off)|keep (?:clear|out)|don'?t|do not|never|skip))", ".", text, flags=re.I)
        clauses = re.split(r"[.;\n]|,\s*(?:and\s+)?(?:then\s+)?|\bthen\b|\bbut\b", text)
        targets, avoid = [], []
        for cl in clauses:
            found = self._places(cl)
            if not found:
                continue
            if self.AVOID.search(cl):
                tm = self.TIME.search(cl)
                s = e = None
                if tm:
                    try:
                        s = parse_time(tm.group(1)) if tm.group(1) else None
                        e = parse_time(tm.group(2) or tm.group(3))
                    except ValueError:
                        s = e = None
                avoid += [AvoidRule(n, s, e) for n in found]
            else:
                targets += [n for n in found if n not in targets]
        return targets, avoid, bool(self.STAY.search(text))

    def parse(self, text: str) -> Interpretation:
        targets, avoid, stay = self.parse_parts(text)
        if not targets:
            return Interpretation(None, "clarify", "Question: which place should the drone inspect?")
        spec = MissionSpec(tuple(targets), tuple(avoid), not stay)
        return Interpretation(Command("order", spec, text), "applied", "Parsed order (regex)")

    def _places(self, clause: str) -> list[str]:
        """Place names mentioned in the clause, in order of appearance (exact, then fuzzy)."""
        low = clause.lower()
        hits = []
        for n in self.names:
            i = low.find(n.lower())
            if i >= 0:
                hits.append((i, n))
        if not hits:
            words = re.findall(r"[a-z0-9']+", low)
            for size in (4, 3, 2):
                for k in range(len(words) - size + 1):
                    cand = " ".join(words[k:k + size])
                    m = difflib.get_close_matches(cand, [n.lower() for n in self.names], n=1, cutoff=0.85)
                    if m:
                        hits.append((low.find(words[k]), next(n for n in self.names if n.lower() == m[0])))
        hits.sort()
        out = []
        for _, n in hits:  # drop names contained in a longer hit ("G Block" inside "Canteen(G Block)")
            if n not in out and not any(n != o and norm(n) in norm(o) for _, o in hits):
                out.append(n)
        return out
