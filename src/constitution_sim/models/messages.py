"""Typed inter-agent messages.

Messages are the communication primitive that lets agents interact
politically — negotiate, threaten, signal, lobby, and make deals.

Every message is logged to the event log for full auditability.
Messages never mutate world state directly; they only influence
agents' subsequent decisions.

Channels:
  - negotiation:      private deal-making ("I'll vote yes if you drop the emergency")
  - public_statement:  broadcast to all agents (press conferences, official positions)
  - threat:           coercive signalling ("I'll strike down your law if you don't...")
  - offer:            cooperative signalling ("I'll support your bill")
  - signal:           soft information ("I plan to vote no on bill X")
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Valid message channels. Restricting these keeps the simulation
# analysable — each channel type has different strategic implications.
MESSAGE_CHANNELS = frozenset({
    "negotiation",
    "public_statement",
    "threat",
    "offer",
    "signal",
})

BROADCAST = "__broadcast__"


class DealProposal(BaseModel):
    """A structured quid-pro-quo proposal embedded in a message.

    Example: "I will do X if you do Y."
    """

    i_will: str            # what the sender commits to
    if_you: str            # what the sender asks of the recipient
    related_law_id: Optional[str] = None  # optional bill/law this concerns


class Message(BaseModel):
    """A single inter-agent communication."""

    sender: str                         # actor_id of the sender
    recipient: str                      # actor_id or BROADCAST
    turn: int                           # turn the message was sent
    channel: str = "signal"             # one of MESSAGE_CHANNELS
    content: str = ""                   # natural language content
    proposal: Optional[DealProposal] = None  # optional structured deal

    def is_broadcast(self) -> bool:
        return self.recipient == BROADCAST
