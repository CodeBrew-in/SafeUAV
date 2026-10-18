"""Step-by-step executor for routes the shield has approved.

The simulator only accepts an ``ApprovedRoute`` and re-validates it on
construction (defence in depth). Every single move is checked again before it
is applied, so a forbidden cell can never be entered even if a bug elsewhere
let a bad route through.
"""

from __future__ import annotations

from enum import Enum

from .mission import Cell
from .safety_shield import ApprovedRoute, ShieldRejection, validate_route


class Phase(str, Enum):
    OUTBOUND = "outbound"
    RETURNING = "returning"
    DONE = "done"


class ExecutionHalted(Exception):
    """Raised if a move would break a constraint. Should be unreachable for approved routes."""


class Simulator:
    def __init__(self, approved: ApprovedRoute):
        if not isinstance(approved, ApprovedRoute):
            raise TypeError("Simulator only executes routes approved by the safety shield")
        recheck = validate_route(list(approved.waypoints), approved.constraints)
        if not recheck.valid:
            raise ShieldRejection(recheck)
        self.c = approved.constraints
        self.route: tuple[Cell, ...] = approved.waypoints
        self.reset()

    def reset(self) -> None:
        """Return the drone to base. Does not touch any recorded run results."""
        self.position: Cell = self.c.start
        self.trail: list[Cell] = [self.c.start]
        self.index = 0
        self.remaining_budget = self.c.max_steps
        self.phase = Phase.OUTBOUND
        self.reached_destination = False
        self.cancelled = False

    @property
    def finished(self) -> bool:
        return self.cancelled or self.index >= len(self.route) - 1

    @property
    def mission_satisfied(self) -> bool:
        if self.cancelled or not self.finished or not self.reached_destination:
            return False
        return self.position == (self.c.start if self.c.return_to_base else self.c.destination)

    def step(self) -> Cell:
        if self.finished:
            raise RuntimeError("route already finished")
        nxt = self.route[self.index + 1]
        x, y = self.position
        if abs(nxt[0] - x) + abs(nxt[1] - y) != 1:
            raise ExecutionHalted(f"non-adjacent move {self.position} -> {nxt}")
        if not self.c.is_traversable(nxt):
            raise ExecutionHalted(f"move into forbidden cell {nxt}")
        if self.remaining_budget <= 0:
            raise ExecutionHalted("movement budget exhausted")

        self.position = nxt
        self.trail.append(nxt)
        self.index += 1
        self.remaining_budget -= 1
        if nxt == self.c.destination and not self.reached_destination:
            self.reached_destination = True
            self.phase = Phase.RETURNING if self.c.return_to_base else Phase.DONE
        if self.index >= len(self.route) - 1:
            self.phase = Phase.DONE
        return nxt

    def run(self) -> list[Cell]:
        while not self.finished:
            self.step()
        return list(self.trail)

    def cancel(self) -> None:
        self.cancelled = True

    def snapshot(self) -> dict:
        return {
            "position": list(self.position),
            "trail": [list(c) for c in self.trail],
            "remaining_budget": self.remaining_budget,
            "phase": self.phase.value,
            "reached_destination": self.reached_destination,
            "finished": self.finished,
        }
