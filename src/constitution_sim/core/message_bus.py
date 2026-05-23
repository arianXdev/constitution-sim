"""Message bus for inter-agent communication.

The MessageBus is the central router for all agent-to-agent messages.
It enforces constitutional communication restrictions (can_send_messages,
allowed_message_recipients) and maintains per-agent inboxes that respect
observation limits.

Design principles:
  - Messages never mutate state. They influence agent cognition only.
  - Every message attempt is logged (successful or filtered).
  - The bus is stateless across turns — inboxes are cleared each turn
    so agents only see messages from the current deliberation round.
  - Broadcast messages go to all agents except the sender.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

from constitution_sim.models.messages import BROADCAST, Message

if TYPE_CHECKING:
    from constitution_sim.models.constitution import Constitution


class MessageBus:
    """Routes messages between agents, enforcing constitutional restrictions."""

    def __init__(self, constitution: Optional[Constitution] = None):
        self._constitution = constitution
        # Per-agent inbox: actor_id -> list of messages received this turn.
        self._inboxes: Dict[str, List[Message]] = defaultdict(list)
        # Full log of all messages sent (for metrics / event logging).
        self._turn_log: List[Message] = []
        # Cumulative log across the entire simulation (for metrics).
        self._all_messages: List[Message] = []

    def clear_turn(self) -> None:
        """Clear per-turn inboxes. Called at the start of each turn."""
        self._inboxes.clear()
        self._turn_log.clear()

    def send(self, message: Message, all_agent_ids: List[str]) -> bool:
        """Attempt to deliver a message. Returns True if delivered.

        Checks constitutional communication permissions before delivery.
        Filtered messages are still recorded in the turn log (with a
        note) so researchers can see attempted but blocked communications.
        """
        # Always log the attempt
        self._turn_log.append(message)
        self._all_messages.append(message)

        # Check if sender is allowed to send messages.
        if not self._can_send(message.sender):
            return False

        if message.is_broadcast():
            recipients = [aid for aid in all_agent_ids if aid != message.sender]
        else:
            # Check if sender can message this specific recipient.
            if not self._can_reach(message.sender, message.recipient):
                return False
            recipients = [message.recipient]

        for recipient in recipients:
            self._inboxes[recipient].append(message)

        return True

    def get_inbox(self, agent_id: str) -> List[Message]:
        """Get all messages received by an agent this turn."""
        return list(self._inboxes.get(agent_id, []))

    def get_turn_log(self) -> List[Message]:
        """All messages sent this turn (for event logging)."""
        return list(self._turn_log)

    def get_all_messages(self) -> List[Message]:
        """All messages sent across the entire simulation (for metrics)."""
        return list(self._all_messages)

    # --- constitutional permission checks ---

    def _get_role_for_agent(self, agent_id: str) -> Optional[str]:
        """Extract role name from agent_id convention (agent_<role_lower>)."""
        if self._constitution is None:
            return None
        # Convention: agent_id = "agent_<role_lower>"
        for role_name in self._constitution.roles:
            if agent_id == f"agent_{role_name.lower()}":
                return role_name
        return None

    def _can_send(self, sender_id: str) -> bool:
        """Check if the sender's role allows sending messages."""
        if self._constitution is None:
            return True
        role_name = self._get_role_for_agent(sender_id)
        if role_name is None:
            return True  # unknown role = permissive default
        role = self._constitution.roles.get(role_name)
        if role is None:
            return True
        return role.observation_limits.can_send_messages

    def _can_reach(self, sender_id: str, recipient_id: str) -> bool:
        """Check if the sender is allowed to message this recipient."""
        if self._constitution is None:
            return True
        role_name = self._get_role_for_agent(sender_id)
        if role_name is None:
            return True
        role = self._constitution.roles.get(role_name)
        if role is None:
            return True
        allowed = role.observation_limits.allowed_message_recipients
        if not allowed:
            return True  # empty = unrestricted
        # Check if recipient's role name is in the allowed list.
        recipient_role = self._get_role_for_agent(recipient_id)
        if recipient_role is None:
            return True
        return recipient_role in allowed
