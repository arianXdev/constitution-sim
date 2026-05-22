from typing import List

class Scheduler:
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.current_index = 0
    
    def get_next_actor(self) -> str:
        if not self.agent_ids:
            raise RuntimeError("No agents in scheduler.")
        actor = self.agent_ids[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agent_ids)
        return actor
