"""Simulation engine and event logger.

The engine owns the per-turn loop: pull the next actor, build a
role-appropriate state view, ask the agent to decide, ask the rules
engine for legality, mutate state on legal actions, log every attempt,
tick the scenario engine, and collect metrics.
"""

from pathlib import Path
from typing import Optional

from constitution_sim.models.actions import Action
from constitution_sim.models.constitution import Constitution
from constitution_sim.models.events import EventRecord
from constitution_sim.models.state import StateView, WorldState
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler


class EventLogger:
    """Append-only JSONL event logger."""

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Truncate / create the file.
        self.log_file.write_text("")

    def log(self, event: EventRecord) -> None:
        with self.log_file.open("a") as f:
            f.write(event.model_dump_json() + "\n")


class SimulationEngine:
    """Per-turn loop driver.

    The engine intentionally does not own any randomness — that lives in
    the scenario engine and the agents. Given the same agents,
    constitution, scheduler, and scenario engine, runs are byte-for-byte
    reproducible.
    """

    def __init__(
        self,
        state: WorldState,
        rules: RulesEngine,
        scheduler: Scheduler,
        logger: EventLogger,
        agents: dict,
        scenario_engine=None,
        metrics_collector=None,
        constitution: Optional[Constitution] = None,
    ):
        self.state = state
        self.rules = rules
        self.scheduler = scheduler
        self.logger = logger
        self.agents = agents  # actor_id -> agent instance
        self.scenario_engine = scenario_engine
        self.metrics_collector = metrics_collector
        # Constitution is optional but recommended: needed to honour
        # per-role observation limits when building StateViews.
        self.constitution = constitution or rules.constitution

    def get_state_view(self, role_name: str) -> StateView:
        """Project the world state through the role's observation limits."""
        role = self.constitution.roles.get(role_name) if self.constitution else None
        limits = role.observation_limits if role else None

        if limits is None:
            return StateView(
                turn=self.state.turn,
                variables=dict(self.state.variables),
                active_laws=list(self.state.active_laws),
                pending_bills=[dict(b) for b in self.state.pending_bills],
                active_shocks=[dict(s) for s in self.state.active_shocks],
                emergency_active=self.state.emergency_active,
            )

        if limits.see_variables:
            if limits.variable_allowlist:
                variables = {
                    k: v
                    for k, v in self.state.variables.items()
                    if k in limits.variable_allowlist
                }
            else:
                variables = dict(self.state.variables)
        else:
            variables = {}

        return StateView(
            turn=self.state.turn,
            variables=variables,
            active_laws=list(self.state.active_laws) if limits.see_active_laws else [],
            pending_bills=(
                [dict(b) for b in self.state.pending_bills]
                if limits.see_pending_bills
                else []
            ),
            active_shocks=(
                [dict(s) for s in self.state.active_shocks]
                if limits.see_active_shocks
                else []
            ),
            emergency_active=self.state.emergency_active,
        )

    def apply_action(self, actor_id: str, action: Action, state: WorldState) -> None:
        """Mutate `state` based on a *legal* action.

        Called only after the rules engine has approved the action.
        """
        action_type = type(action).__name__

        if action_type == "ProposeLaw":
            law_id = getattr(action, "law_id", "")
            content = getattr(action, "content", "")
            # Suppress duplicate proposals of the same id.
            if not any(b.get("law_id") == law_id for b in state.pending_bills):
                state.pending_bills.append(
                    {"law_id": law_id, "content": content, "proposer": actor_id}
                )
        elif action_type == "VoteLaw":
            vote = bool(getattr(action, "vote", False))
            law_id = getattr(action, "law_id", "")
            # Remove from pending regardless — a vote resolves the bill.
            removed = [b for b in state.pending_bills if b.get("law_id") == law_id]
            state.pending_bills = [
                b for b in state.pending_bills if b.get("law_id") != law_id
            ]
            if vote and law_id and law_id not in state.active_laws:
                state.active_laws.append(law_id)
                if removed:
                    state.law_authors[law_id] = removed[0].get("proposer", actor_id)
        elif action_type == "StrikeDownLaw":
            law_id = getattr(action, "law_id", "")
            if law_id in state.active_laws:
                state.active_laws.remove(law_id)
                state.law_authors.pop(law_id, None)
        elif action_type == "VetoLaw":
            law_id = getattr(action, "law_id", "")
            state.pending_bills = [
                b for b in state.pending_bills if b.get("law_id") != law_id
            ]
        elif action_type == "PublishStory":
            sentiment = float(getattr(action, "sentiment", 0.0))
            state.variables["public_trust"] = (
                state.variables.get("public_trust", 0.5) + sentiment * 0.1
            )
        elif action_type == "ImplementPolicy":
            eff = float(getattr(action, "efficiency", 0.0))
            state.variables["state_capacity"] = (
                state.variables.get("state_capacity", 0.5) + eff * 0.1
            )
        elif action_type == "Lobby":
            # Lobbying nudges public_trust very slightly based on intensity
            # and support direction — captures the "soft power" of interest
            # groups without letting them mutate laws directly.
            support = bool(getattr(action, "support", True))
            intensity = float(getattr(action, "intensity", 0.0))
            delta = (0.02 if support else -0.02) * intensity
            state.variables["public_trust"] = (
                state.variables.get("public_trust", 0.5) + delta
            )
        elif action_type == "AppointJudge":
            # Appointment is recorded as an active law-like marker so
            # downstream metrics can see judiciary turnover.
            nominee = getattr(action, "nominee", "")
            if nominee:
                state.variables["judge_appointments"] = (
                    state.variables.get("judge_appointments", 0.0) + 1.0
                )
        elif action_type == "DeclareEmergency":
            state.emergency_active = True
        elif action_type == "LiftEmergency":
            state.emergency_active = False
        # DoNothing intentionally has no effect.

    def run_turn(self) -> None:
        actor_id = self.scheduler.get_next_actor()
        agent = self.agents[actor_id]

        state_view = self.get_state_view(agent.role_name)
        action = agent.decide(state_view)

        is_legal, reason = self.rules.is_legal(agent.role_name, action, self.state)

        # Capture pending-bills count before mutation for deadlock tracking.
        pending_before = len(self.state.pending_bills)

        event = EventRecord(
            turn=self.state.turn,
            actor_id=actor_id,
            action_type=type(action).__name__,
            action_data=action.model_dump(),
            is_legal=is_legal,
            reason=reason,
        )
        self.logger.log(event)

        if is_legal:
            self.apply_action(actor_id, action, self.state)
        else:
            self.state.illegal_action_counts[actor_id] = (
                self.state.illegal_action_counts.get(actor_id, 0) + 1
            )

        # Let the agent record its own outcome (used by LLM agents for
        # memory). Agents that don't expose `remember` simply ignore it.
        if hasattr(agent, "remember"):
            agent.remember(self.state.turn, type(action).__name__, is_legal)

        # Update deadlock counter: if pending bills didn't shrink and we
        # still have pending bills, deadlock grows.
        if (
            self.state.pending_bills
            and len(self.state.pending_bills) >= pending_before
            and type(action).__name__ in {"DoNothing", "PublishStory", "ImplementPolicy", "Lobby"}
        ):
            self.state.deadlock_counter += 1
        else:
            self.state.deadlock_counter = 0

        if self.state.emergency_active:
            self.state.emergency_turns += 1

        if self.scenario_engine is not None:
            self.scenario_engine.tick(self.state)

        if self.metrics_collector is not None:
            self.metrics_collector.collect(self.state)

        self.state.turn += 1
