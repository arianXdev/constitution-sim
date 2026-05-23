from abc import ABC, abstractmethod
from constitution_sim.models.state import StateView
from constitution_sim.models.actions import Action

from typing import List
from constitution_sim.models.messages import Message

class BaseAgent(ABC):
    def __init__(self, agent_id: str, role_name: str):
        self.agent_id = agent_id
        self.role_name = role_name
        
    def communicate(self, state_view: StateView, inbox: List[Message]) -> List[Message]:
        """Optional deliberation phase: return messages to send to others."""
        return []

    def decide_with_messages(self, state_view: StateView, inbox: List[Message]) -> Action:
        """Decide on the next action, optionally using received messages.
        
        Default implementation ignores messages and calls decide().
        """
        return self.decide(state_view)
        
    @abstractmethod
    def decide(self, state_view: StateView) -> Action:
        """Decide on the next action based on the partial state view."""
        pass
