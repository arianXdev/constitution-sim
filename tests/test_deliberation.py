"""Tests for the engine deliberation phase."""

from pathlib import Path
from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.constitution import Constitution, Role
from constitution_sim.models.state import WorldState

def _sample_constitution() -> Constitution:
    return Constitution(
        name="Test",
        version="1.0",
        roles={
            "Executive": Role(name="Executive"),
            "Legislature": Role(name="Legislature"),
        }
    )

class MockAgent(DeterministicHeuristicAgent):
    def communicate(self, state_view, inbox):
        from constitution_sim.models.messages import Message
        # Always send a test message
        return [Message(sender=self.agent_id, recipient="__broadcast__", turn=state_view.turn, channel="signal", content="Test message")]

def test_engine_deliberation_phase(tmp_path: Path):
    state = WorldState()
    rules = RulesEngine(_sample_constitution())

    agents = {
        "agent_executive": MockAgent("agent_executive", "Executive", seed=42),
        "agent_legislature": MockAgent("agent_legislature", "Legislature", seed=42),
    }
    scheduler = Scheduler(["agent_executive", "agent_legislature"])
    logger = EventLogger(tmp_path / "events.jsonl")

    engine = SimulationEngine(state, rules, scheduler, logger, agents)
    engine.run_turn()
    
    # In turn 1, both agents should have sent a message during deliberation
    messages = engine.message_bus.get_all_messages()
    assert len(messages) == 2
    assert messages[0].content == "Test message"
    
    # Active agent was executive, who should have received the legislature's message
    inbox = engine.message_bus.get_inbox("agent_executive")
    assert len(inbox) == 1
    assert inbox[0].sender == "agent_legislature"
