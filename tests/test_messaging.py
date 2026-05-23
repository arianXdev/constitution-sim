"""Tests for the inter-agent messaging bus and models."""

import pytest

from constitution_sim.core.message_bus import MessageBus
from constitution_sim.models.constitution import Constitution, ObservationLimits, Role
from constitution_sim.models.messages import Message


def test_message_bus_delivery():
    bus = MessageBus()
    
    msg1 = Message(sender="agent_exec", recipient="agent_leg", turn=1, channel="signal", content="Hello")
    bus.send(msg1, ["agent_exec", "agent_leg"])
    
    assert len(bus.get_inbox("agent_leg")) == 1
    assert bus.get_inbox("agent_leg")[0].content == "Hello"
    assert len(bus.get_inbox("agent_exec")) == 0
    assert len(bus.get_turn_log()) == 1


def test_message_bus_broadcast():
    bus = MessageBus()
    
    msg = Message(sender="agent_exec", recipient="__broadcast__", turn=1, channel="public_statement", content="Emergency!")
    bus.send(msg, ["agent_exec", "agent_leg", "agent_jud"])
    
    assert len(bus.get_inbox("agent_exec")) == 0
    assert len(bus.get_inbox("agent_leg")) == 1
    assert len(bus.get_inbox("agent_jud")) == 1


def test_message_bus_constitutional_restrictions():
    constitution = Constitution(
        name="Authoritarian",
        version="1.0",
        roles={
            "Executive": Role(name="Executive"),
            "Judiciary": Role(
                name="Judiciary",
                observation_limits=ObservationLimits(
                    can_send_messages=False  # Gag order on judges
                )
            ),
            "Bureaucracy": Role(
                name="Bureaucracy",
                observation_limits=ObservationLimits(
                    allowed_message_recipients=["Executive"]  # Can only talk to Executive
                )
            )
        }
    )
    bus = MessageBus(constitution)
    all_agents = ["agent_executive", "agent_judiciary", "agent_bureaucracy"]
    
    # Exec can send to anyone
    assert bus.send(Message(sender="agent_executive", recipient="agent_judiciary", turn=1), all_agents)
    
    # Judiciary cannot send
    assert not bus.send(Message(sender="agent_judiciary", recipient="agent_executive", turn=1), all_agents)
    
    # Bureaucracy can send to Executive
    assert bus.send(Message(sender="agent_bureaucracy", recipient="agent_executive", turn=1), all_agents)
    
    # Bureaucracy cannot send to Judiciary
    assert not bus.send(Message(sender="agent_bureaucracy", recipient="agent_judiciary", turn=1), all_agents)
    
    # The attempted blocked messages are still logged
    assert len(bus.get_turn_log()) == 4
    
    # Inboxes reflect actual delivery
    assert len(bus.get_inbox("agent_executive")) == 1  # From Bureaucracy
    assert len(bus.get_inbox("agent_judiciary")) == 1  # From Executive
