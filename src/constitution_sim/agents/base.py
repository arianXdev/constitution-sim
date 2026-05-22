from abc import ABC, abstractmethod
from constitution_sim.models.state import StateView
from constitution_sim.models.actions import Action

class BaseAgent(ABC):
    def __init__(self, agent_id: str, role_name: str):
        self.agent_id = agent_id
        self.role_name = role_name
        
    @abstractmethod
    def decide(self, state_view: StateView) -> Action:
        """Decide on the next action based on the partial state view."""
        pass
