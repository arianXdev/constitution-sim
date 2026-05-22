from pydantic import BaseModel
from typing import Dict, Optional

class Shock(BaseModel):
    id: str
    name: str
    description: str
    duration_turns: int
    effects: Dict[str, float]  # e.g. {"public_trust": -0.2, "budget": -50.0}
    trigger_turn: Optional[int] = None
    trigger_probability: Optional[float] = None
