"""World state and observable state views.

`WorldState` is the single canonical source of truth. `StateView` is the
partially observable projection an agent receives. Engines must never
hand out the raw `WorldState` to agents — they always go through
`StateView`.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class WorldState(BaseModel):
    """The canonical truth of the simulation at a given turn."""

    turn: int = 0
    variables: Dict[str, float] = Field(default_factory=dict)
    active_laws: List[str] = Field(default_factory=list)
    pending_bills: List[Dict[str, Any]] = Field(default_factory=list)
    active_shocks: List[Dict[str, Any]] = Field(default_factory=list)
    # Track which actor proposed which active law — used for power
    # concentration / corruption-style metrics.
    law_authors: Dict[str, str] = Field(default_factory=dict)
    # Count of how many illegal actions each actor has attempted.
    illegal_action_counts: Dict[str, int] = Field(default_factory=dict)
    # Number of consecutive turns the legislature has been deadlocked
    # (pending bills not advancing).
    deadlock_counter: int = 0
    # Whether emergency powers are currently active.
    emergency_active: bool = False
    # Number of turns emergency powers have been active in this run.
    emergency_turns: int = 0


class StateView(BaseModel):
    """A partially observable view of the WorldState given to an agent.

    Fields default to empty/None — the engine fills only what the role's
    `ObservationLimits` allow.
    """

    turn: int
    variables: Dict[str, float] = Field(default_factory=dict)
    active_laws: List[str] = Field(default_factory=list)
    pending_bills: List[Dict[str, Any]] = Field(default_factory=list)
    active_shocks: List[Dict[str, Any]] = Field(default_factory=list)
    emergency_active: bool = False
