"""Event log record schema."""

import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime.datetime:
    """UTC-aware now() compatible with Python 3.10+.

    `datetime.UTC` only exists from 3.11; using `timezone.utc` works on all
    supported versions.
    """
    return datetime.datetime.now(datetime.timezone.utc)


class EventRecord(BaseModel):
    """Structured record for one event in the simulation log."""

    timestamp: datetime.datetime = Field(default_factory=_utc_now)
    turn: int
    actor_id: str
    action_type: str
    action_data: Dict[str, Any]
    is_legal: bool
    reason: Optional[str] = None
