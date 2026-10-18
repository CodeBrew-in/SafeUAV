"""Runs one mission under one method and records what happened.

Methods ("arms"):
  * astar                   -- conventional planner. Its route is checked by the shield, then flown.
  * llm_only                -- the LLM's proposal is scored but never flown. Its outcome is
                               hypothetical, so it has its own status values (PROPOSAL_*).
  * llm_shield              -- the proposal is validated. If it is not usable, A* is the
                               fallback, and the fallback route is validated again before it is flown.
  * llm_shield_no_fallback  -- as llm_shield, but a rejected proposal means nothing is flown.
                               This shows how much of llm_shield's success is the LLM's own.

Paired design: ``run_llm_arms`` asks the model once and scores that single
proposal under every LLM arm, so differences between arms come from the policy,
not from different model samples.

Latency fields:
  * proposal_latency_ms -- the LLM call(s), including retries. None for astar and scripted proposals.
  * decision_latency_ms -- the work after the proposal: shield checks plus any A* search
                           (for astar: the A* search plus its shield check).
  * total_latency_ms    -- proposal + decision: time until a route was approved or refused.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .astar import plan_mission
from .llm_planner import LLMProposal, scripted_proposal
from .mission import Cell, MissionContext
from .safety_shield import ValidationResult, approve, is_adjacent, validate_route
from .simulator import Simulator


class Method(str, Enum):
    ASTAR = "astar"
    LLM_ONLY = "llm_only"
    LLM_SHIELD = "llm_shield"
    LLM_SHIELD_NO_FALLBACK = "llm_shield_no_fallback"


LLM_METHODS = (Method.LLM_ONLY, Method.LLM_SHIELD, Method.LLM_SHIELD_NO_FALLBACK)


class Status(str, Enum):
    # Flown arms (astar, llm_shield, llm_shield_no_fallback)
    COMPLETED = "completed"                              # the route the method planned was flown
    COMPLETED_WITH_FALLBACK = "completed_with_fallback"  # the LLM's route was not used; A*'s was flown
    INFEASIBLE_PROVEN = "infeasible_proven"              # A* proved no valid route exists; nothing flown
    NOT_FLOWN = "not_flown"                              # no-fallback arm: proposal not usable; see rejection_cause
    INTERNAL_ERROR = "internal_error"                    # should be impossible: an A* route failed the shield
    # Hypothetical arm (llm_only): the proposal is scored, never flown
    PROPOSAL_VALID = "proposal_valid"
    PROPOSAL_UNSAFE = "proposal_unsafe"                  # flying it would break a safety rule
    PROPOSAL_MISSION_INVALID = "proposal_mission_invalid"  # safe, but misses the goal or the return
    NO_PROPOSAL = "no_proposal"                          # model output could not be parsed, or the call failed
    DECLARED_INFEASIBLE = "declared_infeasible"          # model said no route exists


FLOWN_STATUSES = frozenset({Status.COMPLETED, Status.COMPLETED_WITH_FALLBACK, Status.INFEASIBLE_PROVEN,
                            Status.NOT_FLOWN, Status.INTERNAL_ERROR})


class RejectionCause(str, Enum):
    """Why the LLM's route was not the one flown (llm_shield arms)."""
    SHIELD_REJECTED_UNSAFE = "shield_rejected_unsafe"
    SHIELD_REJECTED_MISSION_INVALID = "shield_rejected_mission_invalid"
    PARSE_FAILURE = "parse_failure"
    TRANSPORT_FAILURE = "transport_failure"
    DECLARED_INFEASIBLE = "declared_infeasible"


SHIELD_CAUSES = frozenset({RejectionCause.SHIELD_REJECTED_UNSAFE, RejectionCause.SHIELD_REJECTED_MISSION_INVALID})


@dataclass(frozen=True)
class Event:
    timestamp: str
    elapsed_ms: float
    kind: str
    message: str
    data: dict = field(default_factory=dict)


class EventLog:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.events: list[Event] = []

    def add(self, kind: str, message: str, **data) -> None:
        self.events.append(Event(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_ms=round((time.perf_counter() - self._t0) * 1000, 3),
            kind=kind,
            message=message,
            data=data,
        ))

    def fork(self) -> "EventLog":
        """A copy that keeps the same clock, for scoring one proposal under several arms."""
        other = EventLog()
        other._t0 = self._t0
        other.events = list(self.events)
        return other


@dataclass
class RunResult:
    run_id: str
    scenario_id: str
    method: Method
    status: Status = Status.INTERNAL_ERROR
    seed: int | None = None
    # ground truth, from A* (complete and optimal on this grid)
    ground_truth_feasible: bool | None = None
    optimal_steps: int | None = None          # shortest full mission, ignoring the budget; None if unreachable
    # the proposal and how the shield judged it
    proposal: object | None = None            # raw waypoints as returned by the planner
    proposal_validation: ValidationResult | None = None
    proposal_unsafe: bool | None = None       # None when there was no route to judge
    proposal_mission_invalid: bool | None = None
    false_infeasible: bool = False            # LLM said "infeasible" about a feasible mission
    rejection_cause: RejectionCause | None = None
    shield_intervened: bool = False           # the shield itself rejected a parsed route
    fallback_used: bool = False
    # what was flown (flown arms) or would have been flown (llm_only)
    accepted_route: list[Cell] | None = None
    flown_route: list[Cell] | None = None
    hypothetical_route: list[Cell] | None = None
    executed_violations: int | None = None    # measured on the flown trail; None if nothing was flown
    # timing (see module docstring)
    proposal_latency_ms: float | None = None
    decision_latency_ms: float | None = None
    total_latency_ms: float | None = None
    events: list[Event] = field(default_factory=list)
    llm: dict | None = None


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _new_log(ctx: MissionContext) -> EventLog:
    log = EventLog()
    log.add("mission_loaded", f"scenario {ctx.scenario_id} loaded", condition=ctx.condition.value)
    return log


def _ground_truth(ctx: MissionContext, res: RunResult):
    plan = plan_mission(ctx.constraints)
    res.ground_truth_feasible = plan.feasible
    res.optimal_steps = plan.required_steps
    return plan


def count_trail_violations(trail: list[Cell], ctx: MissionContext) -> int:
    """Forbidden cells entered plus impossible moves, measured on a trail that was flown."""
    c = ctx.constraints
    bad_cells = sum(1 for p in trail if not c.is_traversable(p))
    bad_moves = sum(1 for a, b in zip(trail, trail[1:]) if not is_adjacent(a, b))
    return bad_cells + bad_moves


def _fly(ctx: MissionContext, check: ValidationResult, res: RunResult, log: EventLog) -> bool:
    """Execute an approved route and record the measured trail. Returns mission_satisfied."""
    sim = Simulator(approve(check, ctx.constraints))  # the simulator re-validates and re-checks each move
    res.accepted_route = list(check.route)
    log.add("execution_started", f"executing {len(check.route) - 1} approved moves")
    trail = sim.run()
    res.flown_route = trail
    res.executed_violations = count_trail_violations(trail, ctx)
    log.add("execution_finished", f"drone finished at {sim.position}",
            remaining_budget=sim.remaining_budget, mission_satisfied=sim.mission_satisfied,
            executed_violations=res.executed_violations)
    return sim.mission_satisfied


def _finish(res: RunResult, log: EventLog, status: Status) -> RunResult:
    res.status = status
    if res.decision_latency_ms is not None:
        res.total_latency_ms = (res.proposal_latency_ms or 0.0) + res.decision_latency_ms
    log.add("mission_finished", status.value)
    res.events = log.events
    return res


def _as_proposal(proposal: LLMProposal | object | None) -> LLMProposal:
    """Accept a planner result, or raw waypoints from a scripted test."""
    if isinstance(proposal, LLMProposal):
        return proposal
    return scripted_proposal(proposal)


def _log_proposal(log: EventLog, prop: LLMProposal, *, request_logged: bool = False) -> None:
    if not request_logged:
        log.add("route_request_submitted", f"route requested from {prop.model}")
    if prop.transport_error:
        log.add("llm_error", prop.transport_error)
    elif prop.parse_error:
        log.add("parse_failed", f"{prop.parse_error} (after {prop.attempts} attempt(s))")
    else:
        log.add("llm_response_parsed", f"{prop.status}: {prop.reason or ''}"[:300],
                latency_ms=round(prop.latency_ms, 1), attempts=prop.attempts)


def _judge_proposal(ctx: MissionContext, prop: LLMProposal, res: RunResult) -> RejectionCause | None:
    """Fill the proposal fields. Returns why the LLM's route cannot be flown, or None if it can."""
    res.proposal = prop.waypoints
    res.llm = prop.summary()
    res.proposal_latency_ms = prop.latency_ms if prop.model != "scripted" else None
    if prop.transport_error:
        return RejectionCause.TRANSPORT_FAILURE
    if prop.parse_error:
        return RejectionCause.PARSE_FAILURE
    if prop.declared_infeasible:
        res.false_infeasible = bool(res.ground_truth_feasible)
        return RejectionCause.DECLARED_INFEASIBLE
    check = validate_route(prop.waypoints, ctx.constraints)
    res.proposal_validation = check
    if check.malformed:
        return RejectionCause.PARSE_FAILURE  # valid JSON, but not a list of [x, y] pairs
    res.proposal_unsafe = check.unsafe
    res.proposal_mission_invalid = check.mission_invalid
    if check.valid:
        return None
    return RejectionCause.SHIELD_REJECTED_UNSAFE if check.unsafe else RejectionCause.SHIELD_REJECTED_MISSION_INVALID


# --- arms -----------------------------------------------------------------------------

def run_astar(ctx: MissionContext) -> RunResult:
    log = _new_log(ctx)
    res = RunResult(_new_id(), ctx.scenario_id, Method.ASTAR)
    t0 = time.perf_counter()
    plan = _ground_truth(ctx, res)
    if not plan.feasible:
        res.decision_latency_ms = (time.perf_counter() - t0) * 1000
        log.add("mission_infeasible", plan.message, outcome=plan.outcome.value)
        return _finish(res, log, Status.INFEASIBLE_PROVEN)
    check = validate_route(plan.route, ctx.constraints)
    res.decision_latency_ms = (time.perf_counter() - t0) * 1000
    res.proposal, res.proposal_validation = plan.route, check
    log.add("route_planned", plan.message, route=plan.route)
    if not check.valid:  # would mean a bug in A*; recorded, never hidden
        log.add("shield_rejected", check.reason, codes=check.code_list)
        return _finish(res, log, Status.INTERNAL_ERROR)
    log.add("shield_accepted", check.reason)
    ok = _fly(ctx, check, res, log)
    return _finish(res, log, Status.COMPLETED if ok else Status.INTERNAL_ERROR)


def run_unshielded(ctx: MissionContext, proposal: LLMProposal | object | None,
                   *, log: EventLog | None = None) -> RunResult:
    """LLM-only baseline: scores the proposal, never flies it."""
    prop = _as_proposal(proposal)
    if log is None:
        log = _new_log(ctx)
        _log_proposal(log, prop)
    res = RunResult(_new_id(), ctx.scenario_id, Method.LLM_ONLY)
    _ground_truth(ctx, res)
    t0 = time.perf_counter()
    cause = _judge_proposal(ctx, prop, res)
    res.decision_latency_ms = (time.perf_counter() - t0) * 1000
    if cause in (RejectionCause.TRANSPORT_FAILURE, RejectionCause.PARSE_FAILURE):
        return _finish(res, log, Status.NO_PROPOSAL)
    if cause is RejectionCause.DECLARED_INFEASIBLE:
        log.add("planner_declared_infeasible", "planner reported the mission as infeasible",
                actually_feasible=res.ground_truth_feasible)
        return _finish(res, log, Status.DECLARED_INFEASIBLE)
    check = res.proposal_validation
    res.hypothetical_route = list(check.route)
    log.add("proposal_scored", check.reason, valid=check.valid, unsafe=check.unsafe, codes=check.code_list,
            swept_forbidden=[list(p) for p in check.swept_forbidden])
    if check.valid:
        return _finish(res, log, Status.PROPOSAL_VALID)
    return _finish(res, log, Status.PROPOSAL_UNSAFE if check.unsafe else Status.PROPOSAL_MISSION_INVALID)


def run_shielded(ctx: MissionContext, proposal: LLMProposal | object | None, *, allow_fallback: bool = True,
                 log: EventLog | None = None) -> RunResult:
    prop = _as_proposal(proposal)
    if log is None:
        log = _new_log(ctx)
        _log_proposal(log, prop)
    method = Method.LLM_SHIELD if allow_fallback else Method.LLM_SHIELD_NO_FALLBACK
    res = RunResult(_new_id(), ctx.scenario_id, method)
    t0 = time.perf_counter()
    plan = _ground_truth(ctx, res)
    cause = _judge_proposal(ctx, prop, res)

    if cause is None:
        check = res.proposal_validation
        res.decision_latency_ms = (time.perf_counter() - t0) * 1000
        log.add("shield_accepted", check.reason)
        ok = _fly(ctx, check, res, log)
        return _finish(res, log, Status.COMPLETED if ok else Status.INTERNAL_ERROR)

    res.rejection_cause = cause
    res.shield_intervened = cause in SHIELD_CAUSES
    if res.shield_intervened:
        check = res.proposal_validation
        log.add("shield_rejected", check.reason, codes=check.code_list, cause=cause.value)
    else:
        log.add("no_usable_proposal", cause.value)

    if not allow_fallback:
        res.decision_latency_ms = (time.perf_counter() - t0) * 1000
        return _finish(res, log, Status.NOT_FLOWN)

    log.add("fallback_requested", "A* fallback requested", cause=cause.value)
    res.fallback_used = True
    if not plan.feasible:
        res.decision_latency_ms = (time.perf_counter() - t0) * 1000
        log.add("mission_infeasible", plan.message, outcome=plan.outcome.value)
        return _finish(res, log, Status.INFEASIBLE_PROVEN)
    fb = validate_route(plan.route, ctx.constraints)
    res.decision_latency_ms = (time.perf_counter() - t0) * 1000
    if not fb.valid:  # would mean a bug in A*
        log.add("fallback_rejected", fb.reason, codes=fb.code_list)
        return _finish(res, log, Status.INTERNAL_ERROR)
    log.add("fallback_accepted", f"valid alternative found ({fb.steps} steps)", route=plan.route)
    ok = _fly(ctx, fb, res, log)
    return _finish(res, log, Status.COMPLETED_WITH_FALLBACK if ok else Status.INTERNAL_ERROR)


def run_llm_arms(ctx: MissionContext, planner, *, seed: int | None = None,
                 methods=LLM_METHODS) -> dict[Method, RunResult]:
    """Ask the model once, then score that same proposal under each LLM arm."""
    log = _new_log(ctx)
    log.add("route_request_submitted", f"route requested from {planner.model}", seed=seed)
    prop: LLMProposal = planner.propose(ctx, seed=seed)
    _log_proposal(log, prop, request_logged=True)

    results: dict[Method, RunResult] = {}
    for m in methods:
        arm_prop = copy.deepcopy(prop)  # arms must not share mutable proposal data
        if m is Method.LLM_ONLY:
            r = run_unshielded(ctx, arm_prop, log=log.fork())
        elif m is Method.LLM_SHIELD:
            r = run_shielded(ctx, arm_prop, allow_fallback=True, log=log.fork())
        elif m is Method.LLM_SHIELD_NO_FALLBACK:
            r = run_shielded(ctx, arm_prop, allow_fallback=False, log=log.fork())
        else:
            raise ValueError(f"run_llm_arms does not handle {m}")
        r.seed = seed
        results[m] = r
    return results
