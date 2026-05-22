"""Constitution schema.

Constitutions are strict, structured data — never free-form text. The
`Constitution` model is loaded from YAML and is consumed by the
`RulesEngine`, the agents (for their goals/utilities), and the engine
(for observation limits / state-view shaping).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class InitialState(BaseModel):
    """Starting values for the world's mutable state variables.

    Carried on the Constitution so a YAML can pin down what a simulation
    looks like at turn 0 (e.g. starting trust, starting budget). Default
    matches a neutral baseline so the simplest constitutions don't have
    to specify anything.
    """

    variables: Dict[str, float] = Field(
        default_factory=lambda: {"public_trust": 0.5, "budget": 1000.0}
    )


class ObservationLimits(BaseModel):
    """What a role can see of the world.

    Roles with constrained visibility model partial observability — e.g.
    the bureaucracy may not see pending bills, voters may not see private
    judicial deliberations, etc.
    """

    see_variables: bool = True
    see_active_laws: bool = True
    see_pending_bills: bool = True
    see_active_shocks: bool = True
    # Optional allow-list of variable names; empty = see all (when
    # see_variables is True). Concrete role configs can pick which
    # state variables a role is allowed to observe.
    variable_allowlist: List[str] = Field(default_factory=list)


class Role(BaseModel):
    """A constitutional role.

    Roles differ by:
    - permissions: which typed actions are legal for this role.
    - goals: short human-readable strings (used by LLM prompts and as
      keys into the utility weights).
    - utility_weights: numeric weights mapped to world-state variables,
      driving the heuristic agent's utility-based action selection.
    - observation_limits: what fraction of the world the role can see.
    """

    name: str
    permissions: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    utility_weights: Dict[str, float] = Field(default_factory=dict)
    observation_limits: ObservationLimits = Field(default_factory=ObservationLimits)


class Rule(BaseModel):
    """A named structured rule.

    For the MVP, a Rule mostly documents intent and lists which action
    types it permits. The RulesEngine evaluates legality from the union
    of role permissions and any rule-level constraints.
    """

    name: str
    description: str = ""
    allowed_actions: List[str] = Field(default_factory=list)
    # Optional: which roles this rule applies to. Empty = applies to all.
    applies_to_roles: List[str] = Field(default_factory=list)


class Constitution(BaseModel):
    """Top-level constitution document."""

    name: str
    version: str
    description: Optional[str] = None
    roles: Dict[str, Role]
    rules: List[Rule] = Field(default_factory=list)
    # Constitutional emergency switch. When True, certain rules may be
    # suspended (see RulesEngine).
    allow_emergency_powers: bool = False
    # Starting world-state variables. Defaults are baked in for the
    # legacy short YAMLs; richer constitutions override per-scenario.
    initial_state: InitialState = Field(default_factory=InitialState)

    @model_validator(mode="after")
    def _check_role_names_match_keys(self) -> "Constitution":
        for key, role in self.roles.items():
            if role.name != key:
                raise ValueError(
                    f"Role key '{key}' does not match role.name '{role.name}'."
                )
        return self
