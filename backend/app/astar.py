"""A* search on the 4-connected grid. Used as the baseline planner and as the shield's fallback."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum

from .mission import Cell, TrustedConstraints

# Fixed neighbour order keeps A* deterministic across runs.
MOVES: tuple[Cell, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shortest_path(c: TrustedConstraints, start: Cell, goal: Cell) -> list[Cell] | None:
    """Shortest traversable path from start to goal (inclusive), or None."""
    if not c.is_traversable(start) or not c.is_traversable(goal):
        return None
    counter = 0
    open_heap: list[tuple[int, int, int, Cell]] = [(manhattan(start, goal), 0, counter, start)]
    came_from: dict[Cell, Cell] = {}
    best_g: dict[Cell, int] = {start: 0}
    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        if g > best_g.get(current, g):
            continue  # stale heap entry
        for dx, dy in MOVES:
            nxt = (current[0] + dx, current[1] + dy)
            if not c.is_traversable(nxt):
                continue
            ng = g + 1
            if ng < best_g.get(nxt, ng + 1):
                best_g[nxt] = ng
                came_from[nxt] = current
                counter += 1
                heapq.heappush(open_heap, (ng + manhattan(nxt, goal), ng, counter, nxt))
    return None


class PlanOutcome(str, Enum):
    OK = "ok"
    DESTINATION_UNREACHABLE = "destination_unreachable"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True)
class PlanResult:
    outcome: PlanOutcome
    route: list[Cell] | None
    required_steps: int | None  # shortest possible mission length, if reachable
    message: str

    @property
    def feasible(self) -> bool:
        return self.outcome is PlanOutcome.OK


def plan_mission(c: TrustedConstraints) -> PlanResult:
    """Shortest route for the whole mission, including the return leg when required.

    Movement is symmetric, so the shortest return leg is the outbound leg
    reversed and the shortest round trip is twice the outbound length. Because
    A* is optimal, a route over budget here means no valid route exists.
    """
    outbound = shortest_path(c, c.start, c.destination)
    if outbound is None:
        return PlanResult(PlanOutcome.DESTINATION_UNREACHABLE, None, None,
                          "no traversable path from base to destination")
    route = outbound + outbound[-2::-1] if c.return_to_base else outbound
    steps = len(route) - 1
    if steps > c.max_steps:
        return PlanResult(PlanOutcome.BUDGET_EXCEEDED, None, steps,
                          f"shortest valid route needs {steps} steps but the budget is {c.max_steps}")
    return PlanResult(PlanOutcome.OK, route, steps, f"route found with {steps} steps")
