"""Loading scenario records from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .mission import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCENARIOS = REPO_ROOT / "benchmark" / "demo_scenarios.json"


def load_scenarios(path: Path = DEMO_SCENARIOS) -> dict[str, Scenario]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(r) for r in records]
    by_id = {s.id: s for s in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError(f"duplicate scenario ids in {path}")
    return by_id
