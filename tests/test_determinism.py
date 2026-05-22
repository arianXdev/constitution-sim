"""Tests for end-to-end determinism with a fixed seed."""

from pathlib import Path

from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.analysis.metrics import MetricsCollector
from constitution_sim.app.cli import load_constitution, load_shocks
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.state import WorldState
from constitution_sim.scenarios.engine import ScenarioEngine


def _run_once(tmp_path: Path, seed: int) -> list[str]:
    constitution = load_constitution(Path("examples/advanced_constitution.yaml"))
    shocks = load_shocks(Path("examples/scenario.yaml"))

    state = WorldState(variables={"public_trust": 0.5, "budget": 1000.0})
    rules = RulesEngine(constitution)
    agents = {}
    for idx, (role_name, role) in enumerate(constitution.roles.items()):
        agents[f"agent_{role_name.lower()}"] = DeterministicHeuristicAgent(
            f"agent_{role_name.lower()}",
            role_name,
            seed + idx,
            utility_weights=role.utility_weights,
        )
    scheduler = Scheduler(list(agents.keys()))
    logger = EventLogger(tmp_path / "events.jsonl")
    scenario_engine = ScenarioEngine(shocks, seed)
    metrics = MetricsCollector()
    engine = SimulationEngine(
        state=state,
        rules=rules,
        scheduler=scheduler,
        logger=logger,
        agents=agents,
        scenario_engine=scenario_engine,
        metrics_collector=metrics,
        constitution=constitution,
    )
    for _ in range(15):
        engine.run_turn()
    return (tmp_path / "events.jsonl").read_text().splitlines()


def test_two_runs_with_same_seed_produce_identical_logs(tmp_path):
    a = _run_once(tmp_path / "a", seed=123)
    b = _run_once(tmp_path / "b", seed=123)
    # `timestamp` field is wall-clock; strip it before comparing.
    import json

    def strip_ts(lines):
        out = []
        for line in lines:
            d = json.loads(line)
            d.pop("timestamp", None)
            out.append(json.dumps(d, sort_keys=True))
        return out

    assert strip_ts(a) == strip_ts(b)


def test_different_seeds_produce_different_logs(tmp_path):
    a = _run_once(tmp_path / "a", seed=1)
    b = _run_once(tmp_path / "b", seed=999)
    assert a != b
