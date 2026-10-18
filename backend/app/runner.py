"""Runs one mission under one method and records what happened.

Methods:
  * astar       -- conventional planner; its route is still checked by the shield.
  * llm_only    -- the proposal is measured but never executed by the simulator.
                   Its "trail" is a hypothetical trajectory, flagged as such.
  * llm_shield  -- the proposal is validated; on rejection A* is the fallback,
                   and the fallback route is validated again before execution.

The LLM call itself happens elsewhere; these functions take its parsed output
so the same logic works for live model output and for scripted test fixtures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .astar import plan_mission
from .mission import Cell, MissionContext
from .safety_shield import MOVEMENT_VIOLATIONS, ValidationResult, approve, validate_route
from .simulator import Simulator


class Method(str, Enum):
    ASTAR = "astar"
    LLM_ONLY = "llm_only"
    LLM_SHIELD = "llm_shield"


class MissionStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_FALLBACK = "completed_with_fallback"
    INFEASIBLE = "infeasible"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # LLM-only baseline: the hypothetical trajectory breaks a constraint or misses the goal.
    UNSAFE = "unsafe"


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


@dataclass
class RunResult:
    run_id: str
    scenario_id: str
    method: Method
    status: MissionStatus
    proposal: object | None = None                  # raw parsed proposer output
    proposal_validation: ValidationResult | None = None
    accepted_route: list[Cell] | None = None
    executed_trail: list[Cell] | None = None
    trail_is_hypothetical: bool = False
    executed_violations: int = 0
    shield_intervened: bool = False
    fallback_used: bool = False
    planning_latency_ms: float | None = None        # shield + fallback time, excluding the LLM call
    proposal_latency_ms: float | None = None        # LLM call time, when supplied
    shortest_route_length: int | None = None
    events: list[Event] = field(default_factory=list)
    error: str | None = None


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _execute(ctx: MissionContext, route: tuple[Cell, ...], log: EventLog) -> tuple[list[Cell], bool]:
    sim = Simulator(approve(validate_route(list(route), ctx.constraints), ctx.constraints))
    log.add("execution_started", f"executing {len(route) - 1} approved moves")
    trail = sim.run()
    log.add("execution_finished", f"drone finished at {sim.position}",
            remaining_budget=sim.remaining_budget, mission_satisfied=sim.mission_satisfied)
    return trail, sim.mission_satisfied


def run_astar(ctx: MissionContext) -> RunResult:
    log = EventLog()
    log.add("mission_loaded", f"scenario {ctx.scenario_id} loaded", condition=ctx.condition.value)
    t0 = time.perf_counter()
    plan = plan_mission(ctx.constraints)
    res = RunResult(_new_id(), ctx.scenario_id, Method.ASTAR, MissionStatus.FAILED,
                    shortest_route_length=plan.required_steps if plan.feasible else None)
    if not plan.feasible:
        res.planning_latency_ms = (time.perf_counter() - t0) * 1000
        res.status = MissionStatus.INFEASIBLE
        log.add("mission_infeasible", plan.message, outcome=plan.outcome.value)
        res.events = log.events
        return res
    check = validate_route(plan.route, ctx.constraints)
    res.planning_latency_ms = (time.perf_counter() - t0) * 1000
    res.proposal, res.proposal_validation = plan.route, check
    log.add("route_planned", plan.message, route=plan.route)
    if not check.valid:  # would indicate a bug in A*; record it rather than hide it
        res.status = MissionStatus.REJECTED
        log.add("validator_rejected", check.reason, codes=sorted(c.value for c in check.codes))
        res.events = log.events
        return res
    log.add("validator_accepted", check.reason)
    res.accepted_route = list(check.route)
    res.executed_trail, ok = _execute(ctx, check.route, log)
    res.status = MissionStatus.COMPLETED if ok else MissionStatus.FAILED
    log.add("mission_finished", res.status.value)
    res.events = log.events
    return res


def run_unshielded(ctx: MissionContext, proposal: object | None, *, declared_infeasible: bool = False,
                   proposal_latency_ms: float | None = None) -> RunResult:
    """LLM-only baseline. Measures the proposal; never runs it on the simulator."""
    log = EventLog()
    log.add("mission_loaded", f"scenario {ctx.scenario_id} loaded", condition=ctx.condition.value)
    plan = plan_mission(ctx.constraints)
    res = RunResult(_new_id(), ctx.scenario_id, Method.LLM_ONLY, MissionStatus.FAILED,
                    proposal=proposal, proposal_latency_ms=proposal_latency_ms,
                    shortest_route_length=plan.required_steps if plan.feasible else None)
    if declared_infeasible and proposal is None:
        res.status = MissionStatus.INFEASIBLE
        log.add("planner_declared_infeasible", "planner reported the mission as infeasible",
                actually_feasible=plan.feasible)
        res.events = log.events
        return res
    t0 = time.perf_counter()
    check = validate_route(proposal, ctx.constraints)
    res.planning_latency_ms = (time.perf_counter() - t0) * 1000
    res.proposal_validation = check
    log.add("route_measured", check.reason, valid=check.valid,
            codes=sorted(c.value for c in check.codes))
    if check.route is None:
        res.status = MissionStatus.FAILED
        res.error = "proposal could not be parsed into waypoints"
        log.add("mission_finished", res.status.value)
        res.events = log.events
        return res
    # Hypothetical trajectory: what the drone would have flown with no shield.
    res.executed_trail = list(check.route)
    res.trail_is_hypothetical = True
    res.executed_violations = sum(1 for v in check.violations if v.code in MOVEMENT_VIOLATIONS)
    res.status = MissionStatus.COMPLETED if check.valid else MissionStatus.UNSAFE
    log.add("mission_finished", f"{res.status.value} (hypothetical, not executed)")
    res.events = log.events
    return res


def run_shielded(ctx: MissionContext, proposal: object | None, *, declared_infeasible: bool = False,
                 proposal_latency_ms: float | None = None, allow_fallback: bool = True) -> RunResult:
    log = EventLog()
    log.add("mission_loaded", f"scenario {ctx.scenario_id} loaded", condition=ctx.condition.value)
    res = RunResult(_new_id(), ctx.scenario_id, Method.LLM_SHIELD, MissionStatus.FAILED,
                    proposal=proposal, proposal_latency_ms=proposal_latency_ms)
    t0 = time.perf_counter()

    if declared_infeasible and proposal is None:
        log.add("planner_declared_infeasible", "planner reported the mission as infeasible; checking with A*")
        res.shield_intervened = False
    else:
        check = validate_route(proposal, ctx.constraints)
        res.proposal_validation = check
        if check.valid:
            res.planning_latency_ms = (time.perf_counter() - t0) * 1000
            res.shortest_route_length = plan_mission(ctx.constraints).required_steps
            log.add("validator_accepted", check.reason)
            res.accepted_route = list(check.route)
            res.executed_trail, ok = _execute(ctx, check.route, log)
            res.status = MissionStatus.COMPLETED if ok else MissionStatus.FAILED
            log.add("mission_finished", res.status.value)
            res.events = log.events
            return res
        res.shield_intervened = True
        log.add("validator_rejected", check.reason, codes=sorted(c.value for c in check.codes))

    plan = plan_mission(ctx.constraints)
    res.shortest_route_length = plan.required_steps if plan.feasible else None
    if not allow_fallback:
        res.planning_latency_ms = (time.perf_counter() - t0) * 1000
        res.status = MissionStatus.INFEASIBLE if not plan.feasible else MissionStatus.REJECTED
        log.add("mission_finished", res.status.value)
        res.events = log.events
        return res

    log.add("fallback_requested", "A* fallback requested")
    res.fallback_used = True
    if not plan.feasible:
        res.planning_latency_ms = (time.perf_counter() - t0) * 1000
        res.status = MissionStatus.INFEASIBLE
        log.add("mission_infeasible", plan.message, outcome=plan.outcome.value)
        res.events = log.events
        return res
    fb = validate_route(plan.route, ctx.constraints)
    res.planning_latency_ms = (time.perf_counter() - t0) * 1000
    if not fb.valid:
        res.status = MissionStatus.REJECTED
        log.add("fallback_rejected", fb.reason)
        res.events = log.events
        return res
    log.add("fallback_accepted", f"valid alternative found ({fb.steps} steps)", route=plan.route)
    res.accepted_route = list(fb.route)
    res.executed_trail, ok = _execute(ctx, fb.route, log)
    res.status = MissionStatus.COMPLETED_WITH_FALLBACK if ok else MissionStatus.FAILED
    log.add("mission_finished", res.status.value)
    res.events = log.events
    return res
