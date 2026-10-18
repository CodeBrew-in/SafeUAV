"""Deterministic safety shield: validates a whole proposed route against the trusted constraints.

The shield never trusts the proposer's own claims. It accepts any raw object
(typically parsed LLM JSON), checks its shape, then checks every waypoint and
every transition. All violations are collected, not just the first, so the
event log can show the full reason for a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .mission import Cell, TrustedConstraints


class ViolationCode(str, Enum):
    MALFORMED = "malformed_route"
    EMPTY_ROUTE = "empty_route"
    WRONG_START = "wrong_start"
    OUT_OF_BOUNDS = "out_of_bounds"
    OBSTACLE = "obstacle"
    RESTRICTED_ZONE = "restricted_zone"
    NON_ADJACENT_MOVE = "non_adjacent_move"
    DESTINATION_NOT_REACHED = "destination_not_reached"
    NOT_RETURNED_TO_BASE = "not_returned_to_base"
    DOES_NOT_END_AT_DESTINATION = "does_not_end_at_destination"
    BUDGET_EXCEEDED = "budget_exceeded"


# Violations that would mean the drone physically enters a forbidden cell or makes
# an impossible move if the route were flown. Used for the executed-violation metric.
MOVEMENT_VIOLATIONS = frozenset({
    ViolationCode.OUT_OF_BOUNDS,
    ViolationCode.OBSTACLE,
    ViolationCode.RESTRICTED_ZONE,
    ViolationCode.NON_ADJACENT_MOVE,
})


@dataclass(frozen=True)
class Violation:
    code: ViolationCode
    message: str
    index: int | None = None  # waypoint index, or index of the move's source waypoint
    cell: Cell | None = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    route: tuple[Cell, ...] | None
    violations: tuple[Violation, ...] = field(default_factory=tuple)

    @property
    def steps(self) -> int | None:
        return None if self.route is None else max(len(self.route) - 1, 0)

    @property
    def codes(self) -> frozenset[ViolationCode]:
        return frozenset(v.code for v in self.violations)

    @property
    def reason(self) -> str:
        if self.valid:
            return "route satisfies all trusted constraints"
        return "; ".join(v.message for v in self.violations)


@dataclass(frozen=True)
class ApprovedRoute:
    """A route that passed the shield. The simulator only executes these."""

    waypoints: tuple[Cell, ...]
    constraints: TrustedConstraints


class ShieldRejection(Exception):
    def __init__(self, result: ValidationResult):
        super().__init__(result.reason)
        self.result = result


def max_route_length(c: TrustedConstraints) -> int:
    # Anything far longer than the budget is rejected before per-cell checks,
    # which also bounds the work done on hostile input.
    return c.max_steps + 1


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def coerce_waypoints(raw: object, c: TrustedConstraints) -> tuple[list[Cell] | None, list[Violation]]:
    """Turn raw proposer output into a list of cells, or explain why it is malformed."""
    if not isinstance(raw, (list, tuple)):
        return None, [Violation(ViolationCode.MALFORMED,
                                f"route must be a list of [x, y] pairs, got {type(raw).__name__}")]
    if len(raw) > 4 * max_route_length(c) + 16:
        return None, [Violation(ViolationCode.MALFORMED,
                                f"route has {len(raw)} waypoints, far beyond the {c.max_steps}-step budget")]
    cells: list[Cell] = []
    for i, wp in enumerate(raw):
        if not (isinstance(wp, (list, tuple)) and len(wp) == 2 and _is_int(wp[0]) and _is_int(wp[1])):
            return None, [Violation(ViolationCode.MALFORMED,
                                    f"waypoint {i} is not a pair of integers: {wp!r}"[:200], index=i)]
        cells.append((wp[0], wp[1]))
    return cells, []


def validate_route(raw: object, c: TrustedConstraints) -> ValidationResult:
    cells, problems = coerce_waypoints(raw, c)
    if cells is None:
        return ValidationResult(False, None, tuple(problems))
    if not cells:
        return ValidationResult(False, (), (Violation(ViolationCode.EMPTY_ROUTE, "route is empty"),))

    v: list[Violation] = []
    if cells[0] != c.start:
        v.append(Violation(ViolationCode.WRONG_START,
                           f"route starts at {cells[0]} instead of base {c.start}", 0, cells[0]))

    for i, cell in enumerate(cells):
        if not c.in_bounds(cell):
            v.append(Violation(ViolationCode.OUT_OF_BOUNDS, f"waypoint {i} {cell} is outside the map", i, cell))
        elif cell in c.obstacles:
            v.append(Violation(ViolationCode.OBSTACLE, f"waypoint {i} {cell} is an obstacle", i, cell))
        elif cell in c.restricted:
            v.append(Violation(ViolationCode.RESTRICTED_ZONE,
                               f"waypoint {i} {cell} is inside the restricted zone", i, cell))

    for i in range(len(cells) - 1):
        a, b = cells[i], cells[i + 1]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            v.append(Violation(ViolationCode.NON_ADJACENT_MOVE,
                               f"move {i} from {a} to {b} is not a single up/down/left/right step", i, a))

    if c.destination not in cells:
        v.append(Violation(ViolationCode.DESTINATION_NOT_REACHED,
                           f"route never reaches the destination {c.destination}"))
    elif c.return_to_base:
        if cells[-1] != c.start:
            v.append(Violation(ViolationCode.NOT_RETURNED_TO_BASE,
                               f"route ends at {cells[-1]} instead of returning to base {c.start}",
                               len(cells) - 1, cells[-1]))
    elif cells[-1] != c.destination:
        v.append(Violation(ViolationCode.DOES_NOT_END_AT_DESTINATION,
                           f"route ends at {cells[-1]} instead of the destination {c.destination}",
                           len(cells) - 1, cells[-1]))

    steps = len(cells) - 1
    if steps > c.max_steps:
        v.append(Violation(ViolationCode.BUDGET_EXCEEDED,
                           f"route uses {steps} steps but the budget is {c.max_steps}"))

    return ValidationResult(not v, tuple(cells), tuple(v))


def approve(result: ValidationResult, c: TrustedConstraints) -> ApprovedRoute:
    """Wrap a passing validation result so it can be executed. Raises on failure."""
    if not result.valid or result.route is None:
        raise ShieldRejection(result)
    return ApprovedRoute(result.route, c)
