import json

from app.mission import build_mission
from app.records import append_jsonl, read_jsonl, to_record
from app.runner import run_astar, run_shielded, run_unshielded
from app.scenarios import load_scenarios

THROUGH_ZONE = [[x, 1] for x in range(1, 11)] + [[10, 2], [10, 1]] + [[x, 1] for x in range(9, 0, -1)]
ENV = {"model": "scripted"}


def test_records_round_trip_through_jsonl(tmp_path):
    ctx = build_mission(load_scenarios()["campus-injection"])
    runs = [run_astar(ctx), run_shielded(ctx, THROUGH_ZONE), run_unshielded(ctx, THROUGH_ZONE)]
    path = tmp_path / "runs.jsonl"
    append_jsonl(path, [to_record(r, ctx, ENV) for r in runs])
    rows = read_jsonl(path)
    assert [r["method"] for r in rows] == ["astar", "llm_shield", "llm_only"]
    astar, shield, only = rows
    assert astar["status"] == "completed" and astar["executed_violations"] == 0
    assert astar["route_efficiency"] == 1.0
    assert shield["rejection_cause"] == "shield_rejected_unsafe" and shield["fallback_used"]
    assert shield["proposal_unsafe"] and shield["flown_route_length"] == 44
    assert only["status"] == "proposal_unsafe" and only["flown_route"] is None
    assert only["executed_violations"] is None
    assert "restricted_zone" in only["violation_codes"]
    json.dumps(rows)  # fully serialisable
