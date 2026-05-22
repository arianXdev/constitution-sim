import random
from typing import List
from constitution_sim.scenarios.shocks import Shock
from constitution_sim.models.state import WorldState

class ScenarioEngine:
    def __init__(self, shocks: List[Shock], seed: int):
        self.shocks = shocks
        self.rng = random.Random(seed)
        
    def tick(self, state: WorldState):
        """
        Evaluate if any new shocks trigger this turn, 
        apply their effects, and decrement duration of active shocks.
        """
        # Process active shocks
        surviving_shocks = []
        for shock_data in state.active_shocks:
            shock_data["remaining_turns"] -= 1
            if shock_data["remaining_turns"] > 0:
                surviving_shocks.append(shock_data)
        state.active_shocks = surviving_shocks
        
        # Trigger new shocks
        triggered_ids = set()
        for shock in self.shocks:
            triggered = False
            if shock.trigger_turn is not None and shock.trigger_turn == state.turn:
                triggered = True
            elif shock.trigger_probability is not None:
                if self.rng.random() < shock.trigger_probability:
                    triggered = True
                    
            if triggered:
                state.active_shocks.append({
                    "id": shock.id,
                    "name": shock.name,
                    "remaining_turns": shock.duration_turns,
                    "effects": shock.effects
                })
                # Apply initial effects immediately
                for var, delta in shock.effects.items():
                    current = state.variables.get(var, 0.0)
                    state.variables[var] = current + delta
                triggered_ids.add(shock.id)
                    
        # Remove triggered shocks from the pool so they don't trigger again
        self.shocks = [s for s in self.shocks if s.id not in triggered_ids]
