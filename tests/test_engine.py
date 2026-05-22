"""End-to-end test of the simulation engine with a handcrafted constitution."""

import json
from pathlib import Path

from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.constitution import Constitution, Role, Rule
from constitution_sim.models.state import WorldState


def _sample_constitution() -> Constitution:
    return Constitution(
        name="Test",
        version="1.0",
        roles={
            "Executive": Role(
                name="Executive", permissions=["ProposeLaw", "DoNothing"]
            ),
            "Legislature": Role(
                name="Legislature", permissions=["VoteLaw", "DoNothing"]
            ),
        },
        rules=[
            Rule(name="Exec Rule", description="", allowed_actions=["ProposeLaw"]),
            Rule(name="Leg Rule", description="", allowed_actions=["VoteLaw"]),
        ],
    )


def test_engine_logs_each_turn(tmp_path: Path):
    state = WorldState()
    rules = RulesEngine(_sample_constitution())

    agents = {
        "exec": DeterministicHeuristicAgent("exec", "Executive", seed=42),
        "leg": DeterministicHeuristicAgent("leg", "Legislature", seed=42),
    }
    scheduler = Scheduler(["exec", "leg"])
    log_file = tmp_path / "events.jsonl"
    logger = EventLogger(log_file)

    engine = SimulationEngine(state, rules, scheduler, logger, agents)
    engine.run_turn()
    engine.run_turn()

    assert state.turn == 2
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2

    event1 = json.loads(lines[0])
    event2 = json.loads(lines[1])
    assert event1["actor_id"] == "exec"
    assert event1["action_type"] == "ProposeLaw"
    assert event1["is_legal"] is True
    assert event2["actor_id"] == "leg"
    assert event2["action_type"] == "VoteLaw"
    assert event2["is_legal"] is True
