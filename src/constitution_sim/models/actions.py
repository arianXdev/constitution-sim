"""Typed action objects.

All agent decisions are expressed as one of these typed Pydantic models.
The `Action` base class carries no fields; subclasses define the payload.
Action types are looked up by class name (`type(action).__name__`) in the
RulesEngine and event log, so do not rename them lightly.
"""

from pydantic import BaseModel


class Action(BaseModel):
    """Base class for all typed actions. No fields; subclasses define them."""

    pass


class ProposeLaw(Action):
    """Propose a new law / bill."""

    law_id: str
    content: str


class VoteLaw(Action):
    """Vote on a pending law. `vote=True` is yea, `False` is nay."""

    law_id: str
    vote: bool


class VetoLaw(Action):
    """Veto a law (executive)."""

    law_id: str


class AppointJudge(Action):
    """Appoint a judge to the judiciary."""

    nominee: str


class StrikeDownLaw(Action):
    """Judicial action: strike down an active law."""

    law_id: str
    reason: str


class PublishStory(Action):
    """Media action: publish a story that nudges public_trust."""

    headline: str
    sentiment: float


class Lobby(Action):
    """Interest-group action: lobby for/against a pending law."""

    law_id: str
    support: bool
    intensity: float


class ImplementPolicy(Action):
    """Bureaucracy action: implement a policy with some efficiency."""

    policy_name: str
    efficiency: float


class DeclareEmergency(Action):
    """Executive: declare a state of emergency.

    Used to surface emergency-power drift in metrics. The RulesEngine
    only honours this if the constitution sets `allow_emergency_powers`.
    """

    reason: str


class LiftEmergency(Action):
    """Executive: lift a state of emergency."""

    reason: str = ""


class FormCoalition(Action):
    """Publicly declare a coalition alignment with another role."""

    partner_role: str
    policy_area: str


class ProposeAmendment(Action):
    """Propose a constitutional amendment (meta-action)."""

    amendment_description: str
    target_rule: str


class DoNothing(Action):
    """Explicit no-op — useful for actors with no legal/useful move."""

    pass
