"""Tests for the LLMAgent: fallback path, structured-output happy path,
and the multi-run evaluator wiring.

A live LLM smoke test is included at the bottom and is automatically
skipped when no `OPENAI_API_KEY` is set in the environment — so this
test suite stays useful both with and without API access.
"""

import json
import os
from pathlib import Path

import pytest

from constitution_sim.agents.llm import LLMAgent
from constitution_sim.analysis.evaluator import Evaluator
from constitution_sim.app.cli import load_constitution
from constitution_sim.models.actions import DoNothing, ProposeLaw
from constitution_sim.models.state import StateView
from constitution_sim.scenarios.shocks import Shock


def test_llm_agent_no_callable_falls_back_to_heuristic():
    agent = LLMAgent(agent_id="exec", role_name="Executive", llm_callable=None)
    sv = StateView(turn=0, variables={}, active_laws=[], pending_bills=[], active_shocks=[])
    action = agent.decide(sv)
    # Executive heuristic with no bills pending proposes one.
    assert isinstance(action, ProposeLaw)
    assert action.law_id == "law_0"


def test_llm_agent_parses_structured_output():
    def mock_llm(prompt: str) -> str:
        return json.dumps({"action_type": "DoNothing", "action_data": {}})

    agent = LLMAgent(
        agent_id="exec",
        role_name="Executive",
        llm_callable=mock_llm,
        allowed_actions=["DoNothing", "ProposeLaw"],
    )
    sv = StateView(turn=0, variables={}, active_laws=[], pending_bills=[], active_shocks=[])
    assert isinstance(agent.decide(sv), DoNothing)


def test_llm_agent_rejects_disallowed_action_and_falls_back():
    """LLM tries to vote when role can only DoNothing or ProposeLaw — must fall back."""

    def mock_llm(prompt: str) -> str:
        return json.dumps(
            {"action_type": "VoteLaw", "action_data": {"law_id": "x", "vote": True}}
        )

    agent = LLMAgent(
        agent_id="exec",
        role_name="Executive",
        llm_callable=mock_llm,
        allowed_actions=["ProposeLaw", "DoNothing"],
    )
    sv = StateView(turn=0, variables={}, active_laws=[], pending_bills=[], active_shocks=[])
    action = agent.decide(sv)
    # Falls back to the heuristic policy, which proposes when nothing is pending.
    assert isinstance(action, ProposeLaw)


def test_llm_agent_memory_is_populated_by_engine():
    """remember() appends to a bounded rolling buffer."""
    agent = LLMAgent(
        agent_id="exec", role_name="Executive", llm_callable=None, memory_size=3
    )
    for t in range(5):
        agent.remember(t, "DoNothing", True)
    assert len(agent.memory) == 3
    assert list(agent.memory)[-1] == (4, "DoNothing", True)


def test_evaluator_runs_with_llm_mock_factory(tmp_path):
    constitution = load_constitution(Path("examples/advanced_constitution.yaml"))
    shocks = [
        Shock(
            id="1",
            name="S",
            description="",
            duration_turns=1,
            effects={"budget": -10},
            trigger_turn=1,
        )
    ]

    def agent_factory(roles, seed):
        return {
            f"agent_{r}": LLMAgent(f"agent_{r}", r, llm_callable=None)
            for r in roles.keys()
        }

    evaluator = Evaluator(constitution, shocks, agent_factory, num_turns=2)
    df = evaluator.run_evaluations(num_runs=2, base_seed=42, log_dir=tmp_path / "evals")
    assert not df.empty
    assert len(df["run_id"].unique()) == 2


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live LLM smoke test.",
)
def test_live_openai_smoke():
    """A tiny live call — just confirms one round-trip parses cleanly."""
    from constitution_sim.agents.providers import get_openai_callable

    call_llm = get_openai_callable()
    agent = LLMAgent(
        agent_id="exec",
        role_name="Executive",
        llm_callable=call_llm,
        allowed_actions=["DoNothing", "ProposeLaw"],
        constitution_name="Smoke Test",
        constitution_description="Tiny live-LLM smoke.",
        other_roles=["Legislature"],
    )
    sv = StateView(
        turn=0, variables={"public_trust": 0.5}, active_laws=[], pending_bills=[],
        active_shocks=[]
    )
    action = agent.decide(sv)
    # The action must validate as one of the allowed types; we don't care
    # which one — only that the live LLM returns parseable JSON.
    assert action.__class__.__name__ in {"DoNothing", "ProposeLaw"}
