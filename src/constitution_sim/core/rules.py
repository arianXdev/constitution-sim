"""Rules engine.

The RulesEngine is the *sole* arbiter of action legality. Agents propose
actions; the engine accepts or rejects. The engine never lets an LLM (or
any other component) mutate state outside this path.
"""

from typing import Tuple

from constitution_sim.models.actions import Action
from constitution_sim.models.constitution import Constitution
from constitution_sim.models.state import WorldState


class RulesEngine:
    def __init__(self, constitution: Constitution):
        self.constitution = constitution

    def is_legal(
        self, actor_role: str, action: Action, state: WorldState
    ) -> Tuple[bool, str]:
        """Decide whether `action` is legal for `actor_role` given `state`.

        Returns (is_legal, reason). The reason is always present so the
        event log captures *why* an action was rejected.
        """
        role = self.constitution.roles.get(actor_role)
        if role is None:
            return False, f"Role '{actor_role}' is not defined in the constitution."

        action_type = type(action).__name__

        # 1. Role-level permission check.
        if action_type not in role.permissions:
            return (
                False,
                f"Role '{actor_role}' has no permission for action '{action_type}'.",
            )

        # 2. Rule-level constraints. A rule applies to this action if its
        #    `allowed_actions` lists this action_type AND (no role
        #    restriction OR this role is in the rule's role list).
        applicable_rules = [
            r
            for r in self.constitution.rules
            if action_type in r.allowed_actions
            and (not r.applies_to_roles or actor_role in r.applies_to_roles)
        ]
        # MVP semantics: if any rule explicitly allows this action_type,
        # we honour it. Extending this to denial rules / preconditions
        # would happen here.
        # (No-op for now; reserved for future structured rules.)
        _ = applicable_rules

        # 3. State-level legality. A few common-sense checks. These are
        #    deliberately conservative — illegal action attempts are
        #    logged so the simulation can be analysed for institutional
        #    stress.
        if action_type == "VoteLaw":
            law_id = getattr(action, "law_id", "")
            if not any(b.get("law_id") == law_id for b in state.pending_bills):
                return False, f"No pending bill with law_id '{law_id}'."
        elif action_type == "StrikeDownLaw":
            law_id = getattr(action, "law_id", "")
            if law_id not in state.active_laws:
                return False, f"Law '{law_id}' is not active and cannot be struck down."
        elif action_type == "VetoLaw":
            law_id = getattr(action, "law_id", "")
            if not any(b.get("law_id") == law_id for b in state.pending_bills):
                return False, f"No pending bill with law_id '{law_id}' to veto."
        elif action_type == "DeclareEmergency":
            if not self.constitution.allow_emergency_powers:
                return False, "Constitution does not allow emergency powers."
            if state.emergency_active:
                return False, "Emergency is already active."
        elif action_type == "LiftEmergency":
            if not state.emergency_active:
                return False, "No emergency is active to lift."

        return True, "Action is legal."
