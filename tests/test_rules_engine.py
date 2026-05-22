"""Tests for the RulesEngine: legality decisions and reasons."""

import pytest

from constitution_sim.core.rules import RulesEngine
from constitution_sim.models.actions import (
    DeclareEmergency,
    DoNothing,
    ProposeLaw,
    StrikeDownLaw,
    VoteLaw,
)
from constitution_sim.models.constitution import Constitution, Role
from constitution_sim.models.state import WorldState


@pytest.fixture
def constitution():
    return Constitution(
        name="T",
        version="1",
        allow_emergency_powers=True,
        roles={
            "Executive": Role(
                name="Executive",
                permissions=["ProposeLaw", "DeclareEmergency", "DoNothing"],
            ),
            "Legislature": Role(
                name="Legislature", permissions=["VoteLaw", "DoNothing"]
            ),
            "Judiciary": Role(
                name="Judiciary", permissions=["StrikeDownLaw", "DoNothing"]
            ),
        },
    )


def test_unknown_role_is_illegal(constitution):
    rules = RulesEngine(constitution)
    is_legal, reason = rules.is_legal("Lobbyist", DoNothing(), WorldState())
    assert not is_legal
    assert "Lobbyist" in reason


def test_action_outside_role_permissions_is_illegal(constitution):
    rules = RulesEngine(constitution)
    is_legal, reason = rules.is_legal(
        "Legislature",
        ProposeLaw(law_id="x", content="y"),
        WorldState(),
    )
    assert not is_legal
    assert "permission" in reason.lower()


def test_vote_requires_pending_bill(constitution):
    rules = RulesEngine(constitution)
    is_legal, reason = rules.is_legal(
        "Legislature",
        VoteLaw(law_id="ghost", vote=True),
        WorldState(),
    )
    assert not is_legal
    assert "ghost" in reason


def test_strike_requires_active_law(constitution):
    rules = RulesEngine(constitution)
    is_legal, reason = rules.is_legal(
        "Judiciary",
        StrikeDownLaw(law_id="not_a_law", reason="x"),
        WorldState(active_laws=["another"]),
    )
    assert not is_legal


def test_declare_emergency_requires_constitutional_authorisation():
    forbidden = Constitution(
        name="F",
        version="1",
        allow_emergency_powers=False,
        roles={
            "Executive": Role(
                name="Executive",
                permissions=["DeclareEmergency", "DoNothing"],
            )
        },
    )
    rules = RulesEngine(forbidden)
    is_legal, reason = rules.is_legal(
        "Executive", DeclareEmergency(reason="x"), WorldState()
    )
    assert not is_legal
    assert "emergency" in reason.lower()


def test_legal_propose(constitution):
    rules = RulesEngine(constitution)
    is_legal, _ = rules.is_legal(
        "Executive", ProposeLaw(law_id="a", content="b"), WorldState()
    )
    assert is_legal
