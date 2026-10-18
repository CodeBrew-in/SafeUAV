"""Loading scenario records from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .astar import plan_mission
from .mission import Scenario, build_mission

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCENARIOS = REPO_ROOT / "benchmark" / "demo_scenarios.json"


def load_scenarios(path: Path = DEMO_SCENARIOS) -> dict[str, Scenario]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(r) for r in records]
    by_id = {s.id: s for s in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError(f"duplicate scenario ids in {path}")
    for s in scenarios:
        # A hand-written label that disagrees with A* is a scenario bug: refuse to load it.
        feasible = plan_mission(build_mission(s).constraints).feasible
        if s.expected_feasible is not None and s.expected_feasible != feasible:
            raise ValueError(f"scenario {s.id}: expected_feasible={s.expected_feasible} but A* says {feasible}")
    return by_id
