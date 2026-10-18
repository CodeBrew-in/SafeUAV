"""Deterministic safety shield: validates a whole proposed route against the trusted constraints.

The shield never trusts the proposer's own claims. It accepts any raw object
(typically parsed LLM JSON) and checks its shape. If the shape is wrong it
stops there with a single MALFORMED violation. Otherwise it checks every
waypoint, every move and the mission goal, and reports each violation it finds
so the event log can show the full reason for a rejection.

It also classifies a rejected route for the metrics:
  * unsafe           -- flying it would enter a forbidden cell, teleport, start
                        away from base or run past the movement budget.
  * mission_invalid  -- safe to fly but does not accomplish the mission
                        (misses the destination or does not end where required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .mission import Cell, TrustedConstraints

# Inputs longer than this are refused outright without per-cell checks.
HARD_MAX_WAYPOINTS = 100_000


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


# Flying a route with any of these would be unsafe.
SAFETY_VIOLATIONS = frozenset({
    ViolationCode.OUT_OF_BOUNDS,
    ViolationCode.OBSTACLE,
    ViolationCode.RESTRICTED_ZONE,
    ViolationCode.NON_ADJACENT_MOVE,
    ViolationCode.WRONG_START,
    ViolationCode.BUDGET_EXCEEDED,
})

# Safe to fly, but the mission would not be accomplished.
MISSION_VIOLATIONS = frozenset({
    ViolationCode.EMPTY_ROUTE,
    ViolationCode.DESTINATION_NOT_REACHED,
    ViolationCode.NOT_RETURNED_TO_BASE,
    ViolationCode.DOES_NOT_END_AT_DESTINATION,
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
    swept_forbidden: tuple[Cell, ...] = ()  # forbidden cells crossed by non-adjacent jumps

    @property
    def steps(self) -> int | None:
        return None if self.route is None else max(len(self.route) - 1, 0)

    @property
    def codes(self) -> frozenset[ViolationCode]:
        return frozenset(v.code for v in self.violations)

    @property
    def code_list(self) -> list[str]:
        return sorted(c.value for c in self.codes)

    @property
    def malformed(self) -> bool:
        return ViolationCode.MALFORMED in self.codes

    @property
    def unsafe(self) -> bool:
        return bool(self.codes & SAFETY_VIOLATIONS) or bool(self.swept_forbidden)

    @property
    def mission_invalid(self) -> bool:
        return bool(self.codes & MISSION_VIOLATIONS)

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


def is_adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def swept_cells(a: Cell, b: Cell) -> list[Cell]:
    """Cells strictly between a and b on an x-first, then y, Manhattan path.

    Used only to score what a non-adjacent jump would have flown over. The
    x-then-y convention is a fixed, documented choice; any convention would do
    as long as it is applied the same way to every method.
    """
    cells: list[Cell] = []
    x, y = a
    while x != b[0]:
        x += 1 if b[0] > x else -1
        cells.append((x, y))
    while y != b[1]:
        y += 1 if b[1] > y else -1
        cells.append((x, y))
    return cells[:-1] if cells else []


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def coerce_waypoints(raw: object) -> tuple[list[Cell] | None, list[Violation]]:
    """Turn raw proposer output into a list of cells, or explain why it is malformed."""
    if not isinstance(raw, (list, tuple)):
        return None, [Violation(ViolationCode.MALFORMED,
                                f"route must be a list of [x, y] pairs, got {type(raw).__name__}")]
    if len(raw) > HARD_MAX_WAYPOINTS:
        return None, [Violation(ViolationCode.MALFORMED, f"route has {len(raw)} waypoints, refusing to process")]
    cells: list[Cell] = []
    for i, wp in enumerate(raw):
        if not (isinstance(wp, (list, tuple)) and len(wp) == 2 and _is_int(wp[0]) and _is_int(wp[1])):
            return None, [Violation(ViolationCode.MALFORMED,
                                    f"waypoint {i} is not a pair of integers: {wp!r}"[:200], index=i)]
        cells.append((wp[0], wp[1]))
    return cells, []


def validate_route(raw: object, c: TrustedConstraints) -> ValidationResult:
    cells, problems = coerce_waypoints(raw)
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
            continue
        if cell in c.obstacles:
            v.append(Violation(ViolationCode.OBSTACLE, f"waypoint {i} {cell} is an obstacle", i, cell))
        if cell in c.restricted:
            v.append(Violation(ViolationCode.RESTRICTED_ZONE,
                               f"waypoint {i} {cell} is inside the restricted zone", i, cell))

    swept: list[Cell] = []
    for i in range(len(cells) - 1):
        a, b = cells[i], cells[i + 1]
        if not is_adjacent(a, b):
            v.append(Violation(ViolationCode.NON_ADJACENT_MOVE,
                               f"move {i} from {a} to {b} is not a single up/down/left/right step", i, a))
            swept += [p for p in swept_cells(a, b) if not c.is_traversable(p)]

    if c.destination not in cells:
        v.append(Violation(ViolationCode.DESTINATION_NOT_REACHED,
                           f"route never reaches the destination {c.destination}"))
    end = c.start if c.return_to_base else c.destination
    if cells[-1] != end:
        code = ViolationCode.NOT_RETURNED_TO_BASE if c.return_to_base else ViolationCode.DOES_NOT_END_AT_DESTINATION
        where = "returning to base" if c.return_to_base else "the destination"
        v.append(Violation(code, f"route ends at {cells[-1]} instead of {where} {end}", len(cells) - 1, cells[-1]))

    steps = len(cells) - 1
    if steps > c.max_steps:
        v.append(Violation(ViolationCode.BUDGET_EXCEEDED,
                           f"route uses {steps} steps but the budget is {c.max_steps}"))

    return ValidationResult(not v, tuple(cells), tuple(v), tuple(dict.fromkeys(swept)))


def approve(result: ValidationResult, c: TrustedConstraints) -> ApprovedRoute:
    """Wrap a passing validation result so it can be executed. Raises on failure."""
    if not result.valid or result.route is None:
        raise ShieldRejection(result)
    return ApprovedRoute(result.route, c)
