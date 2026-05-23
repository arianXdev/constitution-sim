"""End-to-end test of scenarios + roles + the advanced constitution.

Verifies that loading the advanced constitution + the example scenario
and running 5 turns through every role produces a coherent event log,
metrics history, and a mutated world state.
"""

from pathlib import Path

from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.analysis.metrics import MetricsCollector
from constitution_sim.app.cli import load_constitution, load_shocks
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.state import WorldState
from constitution_sim.scenarios.engine import ScenarioEngine


def test_advanced_constitution_with_scenario(tmp_path: Path):
    constitution = load_constitution(Path("constitutions/advanced_constitution.yaml"))
    shocks = load_shocks(Path("constitutions/scenario.yaml"))

    state = WorldState(variables=dict(constitution.initial_state.variables))
    rules = RulesEngine(constitution)

    agents = {}
    for role_name in constitution.roles.keys():
        agent_id = f"agent_{role_name.lower()}"
        agents[agent_id] = DeterministicHeuristicAgent(
            agent_id=agent_id,
            role_name=role_name,
            seed=42,
            utility_weights=constitution.roles[role_name].utility_weights,
        )

    scheduler = Scheduler(list(agents.keys()))
    logger = EventLogger(tmp_path / "events.jsonl")
    scenario_engine = ScenarioEngine(shocks, seed=42)
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

    for _ in range(5):
        engine.run_turn()

    assert state.turn == 5
    assert len(metrics.history) == 5
    # The economic crisis shock fires at turn 2, so trust must have moved.
    assert state.variables["public_trust"] != 0.5
