"""Tests for the constitution schema and YAML loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from constitution_sim.app.cli import load_constitution
from constitution_sim.models.constitution import (
    Constitution,
    ObservationLimits,
    Role,
    Rule,
)


def test_role_key_must_match_role_name():
    with pytest.raises(ValidationError):
        Constitution(
            name="X",
            version="1",
            roles={"Executive": Role(name="Wrong", permissions=["DoNothing"])},
            rules=[],
        )


def test_observation_limits_defaults():
    role = Role(name="R")
    assert role.observation_limits.see_variables is True
    assert role.observation_limits.see_pending_bills is True
    assert role.observation_limits.variable_allowlist == []


def test_load_simple_constitution():
    c = load_constitution(Path("constitutions/simple_constitution.yaml"))
    assert c.name == "Simple Constitution"
    assert set(c.roles) == {"Executive", "Legislature"}
    assert "ProposeLaw" in c.roles["Executive"].permissions


def test_load_advanced_constitution_has_utilities_and_limits():
    c = load_constitution(Path("constitutions/advanced_constitution.yaml"))
    assert c.allow_emergency_powers is True
    assert c.roles["Executive"].utility_weights["public_trust"] == 1.0
    assert c.roles["Bureaucracy"].observation_limits.see_pending_bills is False
    assert c.roles["Judiciary"].observation_limits.variable_allowlist == [
        "public_trust",
        "state_capacity",
    ]


def test_initial_state_loads_from_yaml():
    c = load_constitution(Path("constitutions/advanced_constitution.yaml"))
    assert c.initial_state.variables["public_trust"] == 0.5
    assert c.initial_state.variables["state_capacity"] == 0.5
    assert c.initial_state.variables["budget"] == 1000.0


def test_initial_state_defaults_when_missing():
    c = Constitution(
        name="X",
        version="1",
        roles={"Executive": Role(name="Executive", permissions=["DoNothing"])},
    )
    assert c.initial_state.variables["public_trust"] == 0.5
    assert c.initial_state.variables["budget"] == 1000.0


def test_rule_applies_to_roles_field():
    rule = Rule(
        name="x",
        description="",
        allowed_actions=["ProposeLaw"],
        applies_to_roles=["Executive"],
    )
    assert rule.applies_to_roles == ["Executive"]
