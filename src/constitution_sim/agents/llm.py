"""LLM-backed agent with strict structured outputs and deterministic fallback.

Design rules:

1. The LLM is the *primary* cognition path. Heuristic is a deterministic
   safety net used when (a) no LLM is configured, or (b) the LLM returns
   garbage that can't be parsed/validated.
2. LLM responses must validate against the typed `Action` schemas. Any
   parse / validation failure falls back to `DeterministicHeuristicAgent`.
3. The LLM never mutates world state directly — it can only return a
   typed action that still has to pass the RulesEngine.
4. Prompts are kept here so the rest of the simulator stays prompt-free.
5. Agents keep a short rolling memory of their own recent decisions so
   the LLM can reason about continuity ("I just proposed law_3 — what now?").
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from constitution_sim.agents.base import BaseAgent
from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.models.actions import (
    Action,
    DeclareEmergency,
    DoNothing,
    ImplementPolicy,
    LiftEmergency,
    Lobby,
    ProposeLaw,
    PublishStory,
    StrikeDownLaw,
    VetoLaw,
    VoteLaw,
)
from constitution_sim.models.state import StateView

logger = logging.getLogger(__name__)

# Map the action_type string from JSON back to a typed Pydantic class.
ACTION_MAP: Dict[str, type] = {
    "ProposeLaw": ProposeLaw,
    "VoteLaw": VoteLaw,
    "VetoLaw": VetoLaw,
    "StrikeDownLaw": StrikeDownLaw,
    "PublishStory": PublishStory,
    "ImplementPolicy": ImplementPolicy,
    "Lobby": Lobby,
    "DeclareEmergency": DeclareEmergency,
    "LiftEmergency": LiftEmergency,
    "DoNothing": DoNothing,
}

# Role-specific persona blurbs injected into the system prompt. Keep them
# short, opinionated, and self-interested — they're what make the
# simulator agentic rather than just utility-driven.
ROLE_PERSONAS: Dict[str, str] = {
    "Executive": (
        "You are the head of state. You are politically ambitious and want "
        "to consolidate your agenda through legislation. You strongly value "
        "public_trust because a collapse means revolt, but you will use "
        "emergency powers if trust is destroyed. You do NOT vote on bills."
    ),
    "Legislature": (
        "You are a senior parliamentarian. You vote on the executive's bills. "
        "You represent constituents and are cautious about new laws, but you "
        "also do not want to look obstructionist. You vote against bills that "
        "lower public_trust; you vote for bills that raise state_capacity."
    ),
    "Judiciary": (
        "You are a constitutional judge. You do not initiate; you react. You "
        "strike down active laws that you judge to be unconstitutional or "
        "destabilising. You weigh long-term legitimacy over short-term gain. "
        "You only act when you see active laws in the state view."
    ),
    "Media": (
        "You are an editor-in-chief. You shape public discourse by publishing "
        "stories whose sentiment (-1.0 .. +1.0) directly nudges public_trust. "
        "You are biased toward stories that move the needle — you rarely "
        "publish neutral content. You match the current trust direction."
    ),
    "Bureaucracy": (
        "You are a senior civil servant. You implement policies without "
        "ideology. Your efficiency (0.0 .. 1.0) raises state_capacity. You "
        "implement steadily; you do not strike or veto."
    ),
    "InterestGroup": (
        "You are an interest-group lobbyist. You lobby for or against pending "
        "bills with some intensity. You never legislate, vote, or rule."
    ),
}


class LLMAgent(BaseAgent):
    """An agent that asks an LLM for a typed action, with deterministic fallback.

    Memory: keeps the last `memory_size` of its own (turn, action_type,
    is_legal) tuples so the LLM can reason about continuity. The engine
    populates this via `remember()` after each turn.
    """

    def __init__(
        self,
        agent_id: str,
        role_name: str,
        llm_callable: Optional[Callable[[str], str]] = None,
        fallback_seed: int = 42,
        goals: Optional[List[str]] = None,
        utility_weights: Optional[Dict[str, float]] = None,
        allowed_actions: Optional[List[str]] = None,
        constitution_name: str = "(unnamed constitution)",
        constitution_description: str = "",
        other_roles: Optional[List[str]] = None,
        memory_size: int = 5,
    ):
        super().__init__(agent_id, role_name)
        self.llm_callable = llm_callable
        self.goals = list(goals or [])
        self.utility_weights = dict(utility_weights or {})
        self.allowed_actions = list(allowed_actions or list(ACTION_MAP.keys()))
        self.constitution_name = constitution_name
        self.constitution_description = constitution_description
        self.other_roles = [r for r in (other_roles or []) if r != role_name]
        self.memory: Deque[Tuple[int, str, bool]] = deque(maxlen=memory_size)
        self.fallback_agent = DeterministicHeuristicAgent(
            agent_id,
            role_name,
            fallback_seed,
            utility_weights=self.utility_weights,
        )

    # ----- engine integration --------------------------------------------

    def remember(self, turn: int, action_type: str, is_legal: bool) -> None:
        """Record the outcome of this agent's most recent decision."""
        self.memory.append((turn, action_type, is_legal))

    # ----- decision -------------------------------------------------------

    def decide(self, state_view: StateView) -> Action:
        if self.llm_callable is None:
            return self.fallback_agent.decide(state_view)

        prompt = self._build_prompt(state_view)
        try:
            response_text = self.llm_callable(prompt)
            parsed = json.loads(response_text)

            action_type = parsed.get("action_type")
            action_data = parsed.get("action_data", {}) or {}

            if action_type not in ACTION_MAP:
                raise ValueError(f"Unknown action_type: {action_type!r}")
            if action_type not in self.allowed_actions:
                raise ValueError(
                    f"Action {action_type!r} not in role's allowed action set."
                )

            action_cls = ACTION_MAP[action_type]
            # Pydantic v2 will raise ValidationError on schema mismatch.
            return action_cls(**action_data)
        except Exception as exc:  # noqa: BLE001 — fallback by design
            logger.warning(
                "[%s] LLM parse/validation failed (%s). Falling back.",
                self.agent_id,
                exc,
            )
            return self.fallback_agent.decide(state_view)

    # ----- prompt construction -------------------------------------------

    def _build_prompt(self, state_view: StateView) -> str:
        persona = ROLE_PERSONAS.get(
            self.role_name, "You are a political actor in a simulated state."
        )
        memory_block = self._render_memory()
        return (
            f"You are simulating the {self.role_name} of the country governed "
            f"by '{self.constitution_name}'.\n"
            f"Constitution context: {self.constitution_description or '(no description)'}\n"
            f"Other roles in this state: {', '.join(self.other_roles) or '(none)'}.\n"
            f"\n"
            f"PERSONA: {persona}\n"
            f"\n"
            f"Your declared goals: {self.goals or '(none specified)'}\n"
            f"Your utility weights (variable -> weight): "
            f"{self.utility_weights or '(none)'}\n"
            f"\n"
            f"World state (your partial view at turn {state_view.turn}):\n"
            f"  variables:       {state_view.variables}\n"
            f"  active_laws:     {state_view.active_laws}\n"
            f"  pending_bills:   {state_view.pending_bills}\n"
            f"  active_shocks:   {state_view.active_shocks}\n"
            f"  emergency_active: {state_view.emergency_active}\n"
            f"\n"
            f"{memory_block}"
            f"You may pick ONE of these typed actions (and only these):\n"
            f"  {self.allowed_actions}\n"
            f"\n"
            f"Reply with a single JSON object of the form:\n"
            f'  {{"action_type": "<one of the above>", "action_data": {{...}}}}\n'
            f"\n"
            f"Action_data shapes by action_type:\n"
            f'  ProposeLaw       => {{"law_id": "law_X", "content": "..."}}\n'
            f'  VoteLaw          => {{"law_id": "law_X", "vote": true|false}}\n'
            f'  VetoLaw          => {{"law_id": "law_X"}}\n'
            f'  StrikeDownLaw    => {{"law_id": "law_X", "reason": "..."}}\n'
            f'  PublishStory     => {{"headline": "...", "sentiment": -1.0..1.0}}\n'
            f'  ImplementPolicy  => {{"policy_name": "...", "efficiency": 0.0..1.0}}\n'
            f'  Lobby            => {{"law_id": "law_X", "support": true|false, "intensity": 0.0..1.0}}\n'
            f'  DeclareEmergency => {{"reason": "..."}}\n'
            f'  LiftEmergency    => {{"reason": "..."}}\n'
            f'  DoNothing        => {{}}\n'
            f"\n"
            f"Return ONLY the JSON object. No commentary. No markdown."
        )

    def _render_memory(self) -> str:
        if not self.memory:
            return "Your recent decisions: (none yet — this is your first turn).\n\n"
        lines = ["Your recent decisions (most recent last):"]
        for turn, atype, ok in self.memory:
            tag = "OK" if ok else "REJECTED"
            lines.append(f"  turn {turn}: {atype}  [{tag}]")
        return "\n".join(lines) + "\n\n"
