"""Heuristic (deterministic, no-LLM) agent.

Models bounded rationality + a small set of cognitive biases. Given the
same seed and state-view sequence, this agent is byte-for-byte
reproducible. Behaviour:

- Per-role action templates (executive proposes, legislature votes, etc).
- Optional utility weights from the role schema bias voting and lobbying.
- Two simple biases: status-quo bias (cap on new laws per session) and
  confirmation bias (over-weight variables the role already cares about).
"""

import random
from collections import defaultdict
from typing import Dict, Optional, List

from constitution_sim.agents.base import BaseAgent
from constitution_sim.models.actions import (
    Action,
    DeclareEmergency,
    DoNothing,
    ImplementPolicy,
    LiftEmergency,
    ProposeLaw,
    PublishStory,
    StrikeDownLaw,
    VoteLaw,
)
from constitution_sim.models.messages import Message, DealProposal, BROADCAST
from constitution_sim.models.state import StateView


class DeterministicHeuristicAgent(BaseAgent):
    """A simple, deterministic, seeded utility-biased decision policy."""

    def __init__(
        self,
        agent_id: str,
        role_name: str,
        seed: int,
        utility_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(agent_id, role_name)
        self.rng = random.Random(seed)
        self.utility_weights: Dict[str, float] = dict(utility_weights or {})
        # Simple reputation: actor_id -> score (-1.0 to 1.0)
        self.reputation: Dict[str, float] = defaultdict(float)
        self.last_seen_turn: int = -1

    # ----- utility -------------------------------------------------------

    def _utility(self, variables: Dict[str, float]) -> float:
        """Linear utility over observed state variables."""
        if not self.utility_weights:
            return 0.0
        score = 0.0
        for k, w in self.utility_weights.items():
            score += w * variables.get(k, 0.0)
        return score

    def _vote_yes_probability(self, state_view: StateView) -> float:
        """Combine a base rate with utility-driven adjustment.

        Base rate is 0.7. Higher current utility nudges the agent toward
        the status quo (less likely to vote yes); lower utility nudges
        toward yes (more willing to change things). Confirmation bias is
        a 0.05 bump in either direction.
        """
        base = 0.7
        if not self.utility_weights:
            return base
        u = self._utility(state_view.variables)
        # Map u (any real) to [-0.3, 0.3] with tanh-like clamp.
        if u > 1.0:
            adj = -0.3
        elif u < -1.0:
            adj = 0.3
        else:
            adj = -0.3 * u
        # Confirmation bias toward existing direction.
        adj += 0.05 if u >= 0 else -0.05
        return max(0.05, min(0.95, base + adj))

    def _process_history(self, state_view: StateView):
        """Update reputation based on what other actors just did."""
        if not hasattr(state_view, "recent_actions") or not state_view.recent_actions:
            return
            
        for action_rec in state_view.recent_actions:
            if action_rec["turn"] <= self.last_seen_turn:
                continue
                
            actor = action_rec["actor_id"]
            if actor == self.agent_id:
                continue
                
            # Basic reciprocity: if they did something illegal, distrust them.
            if not action_rec.get("is_legal", True):
                self.reputation[actor] = max(-1.0, self.reputation[actor] - 0.2)
            
            # If Judiciary strikes down laws, Executive distrusts them.
            if self.role_name == "Executive" and action_rec["action_type"] == "StrikeDownLaw":
                self.reputation[actor] = max(-1.0, self.reputation[actor] - 0.3)
                
            # If Legislature votes Yes, Executive trusts them more.
            # (Note: we can't see the exact vote direction easily here without action_data in history,
            # but we can assume VoteLaw is generally participation).
        
        self.last_seen_turn = state_view.turn

    # ----- communication -------------------------------------------------

    def communicate(self, state_view: StateView, inbox: List[Message]) -> List[Message]:
        self._process_history(state_view)
        messages = []
        
        # Legislature signals voting intention.
        if self.role_name == "Legislature" and state_view.pending_bills:
            bill = state_view.pending_bills[0]
            p_yes = self._vote_yes_probability(state_view)
            if p_yes > 0.6:
                messages.append(Message(
                    sender=self.agent_id, recipient=BROADCAST, turn=state_view.turn,
                    channel="signal", content=f"I plan to support {bill['law_id']}."
                ))
            elif p_yes < 0.4:
                messages.append(Message(
                    sender=self.agent_id, recipient=BROADCAST, turn=state_view.turn,
                    channel="signal", content=f"I plan to oppose {bill['law_id']}."
                ))
                
        # Executive public statements on emergencies.
        if self.role_name == "Executive" and state_view.emergency_active:
            if self.rng.random() < 0.3:
                messages.append(Message(
                    sender=self.agent_id, recipient=BROADCAST, turn=state_view.turn,
                    channel="public_statement", content="The emergency measures are necessary for stability."
                ))
                
        return messages

    # ----- decide --------------------------------------------------------

    def decide_with_messages(self, state_view: StateView, inbox: List[Message]) -> Action:
        self._process_history(state_view)
        
        # Process inbox to adjust reputation
        for msg in inbox:
            if msg.channel == "threat":
                self.reputation[msg.sender] = max(-1.0, self.reputation[msg.sender] - 0.1)
            elif msg.channel == "offer":
                self.reputation[msg.sender] = min(1.0, self.reputation[msg.sender] + 0.1)

        # Then decide as normal, but base logic could use self.reputation
        return self.decide(state_view)

    def decide(self, state_view: StateView) -> Action:
        role = self.role_name

        if role == "Executive":
            # Status-quo bias: don't propose more than 8 active laws.
            if not state_view.pending_bills and len(state_view.active_laws) < 8:
                law_id = f"law_{state_view.turn}"
                return ProposeLaw(law_id=law_id, content=f"Content for {law_id}")
            # Optional emergency declaration when public_trust collapses.
            trust = state_view.variables.get("public_trust", 0.5)
            if (
                not state_view.emergency_active
                and trust < 0.15
                and self.rng.random() < 0.3
            ):
                return DeclareEmergency(reason="Public trust collapse")
            if state_view.emergency_active and trust > 0.4 and self.rng.random() < 0.2:
                return LiftEmergency(reason="Stability restored")
            return DoNothing()

        if role == "Legislature":
            if state_view.pending_bills:
                bill = state_view.pending_bills[0]
                p_yes = self._vote_yes_probability(state_view)
                
                # Adjust based on proposer's reputation
                proposer = bill.get("proposer", "")
                if proposer in self.reputation:
                    p_yes += self.reputation[proposer] * 0.2
                    p_yes = max(0.0, min(1.0, p_yes))
                    
                vote = self.rng.random() < p_yes
                return VoteLaw(law_id=bill["law_id"], vote=vote)
            return DoNothing()

        if role == "Judiciary":
            # Strike-down likelihood rises with deadlock / emergencies.
            if state_view.active_laws:
                base_p = 0.1
                if state_view.emergency_active:
                    base_p += 0.1
                if self.rng.random() < base_p:
                    law_id = self.rng.choice(state_view.active_laws)
                    return StrikeDownLaw(law_id=law_id, reason="Unconstitutional")
            return DoNothing()

        if role == "Media":
            if self.rng.random() < 0.25:
                # Media is biased toward stories matching public_trust gradient.
                trust = state_view.variables.get("public_trust", 0.5)
                drift = (trust - 0.5) * 2.0
                sentiment = max(-1.0, min(1.0, self.rng.gauss(drift, 0.4)))
                return PublishStory(
                    headline=f"News at turn {state_view.turn}", sentiment=sentiment
                )
            return DoNothing()

        if role == "Bureaucracy":
            if self.rng.random() < 0.3:
                efficiency = max(0.0, min(1.0, self.rng.gauss(0.5, 0.15)))
                return ImplementPolicy(
                    policy_name="Standard Routine", efficiency=efficiency
                )
            return DoNothing()

        return DoNothing()
