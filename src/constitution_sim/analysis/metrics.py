"""Per-turn metrics collection.

Captures the institutional outcomes called out in the project brief:
power concentration, deadlock, trust, volatility, legitimacy, corruption,
and emergency-power drift.

These metrics are intentionally simple — they're indicators, not models.
"""

from typing import Any, Dict, List

from constitution_sim.models.state import WorldState


class MetricsCollector:
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self._prev_trust: float = 0.5

    @staticmethod
    def _power_concentration(state: WorldState) -> float:
        """Share of active laws authored by the single most-frequent actor.

        Range 0 (perfectly distributed) → 1 (one actor authored everything).
        Returns 0 when there are no laws.
        """
        if not state.law_authors:
            return 0.0
        counts: Dict[str, int] = {}
        for actor in state.law_authors.values():
            counts[actor] = counts.get(actor, 0) + 1
        max_count = max(counts.values())
        return max_count / max(1, len(state.law_authors))

    @staticmethod
    def _legitimacy(state: WorldState) -> float:
        """Crude legitimacy proxy: trust × (1 - illegal_rate)."""
        trust = state.variables.get("public_trust", 0.5)
        total_illegal = sum(state.illegal_action_counts.values())
        # Normalise: 1.0 if zero illegal actions, decaying with attempts.
        illegal_rate = total_illegal / max(1, total_illegal + max(1, state.turn))
        return max(0.0, trust) * (1.0 - illegal_rate)

    @staticmethod
    def _corruption_proxy(state: WorldState) -> float:
        """Sum of illegal-action counts (cheap proxy for corruption pressure)."""
        return float(sum(state.illegal_action_counts.values()))

    def collect(self, state: WorldState, message_bus: Any = None) -> None:
        trust = state.variables.get("public_trust", 0.5)
        volatility = abs(trust - self._prev_trust)
        self._prev_trust = trust

        metrics: Dict[str, Any] = {
            "turn": state.turn,
            "num_active_laws": len(state.active_laws),
            "num_pending_bills": len(state.pending_bills),
            "num_active_shocks": len(state.active_shocks),
            "power_concentration": self._power_concentration(state),
            "deadlock_counter": state.deadlock_counter,
            "trust_volatility": volatility,
            "legitimacy": self._legitimacy(state),
            "corruption_proxy": self._corruption_proxy(state),
            "emergency_active": int(state.emergency_active),
            "emergency_turns": state.emergency_turns,
            "communication_volume": len(message_bus.get_turn_log()) if message_bus else 0,
            "active_coalitions": len(getattr(state, "active_coalitions", [])),
            "proposed_amendments": len(getattr(state, "proposed_amendments", [])),
        }
        for k, v in state.variables.items():
            metrics[k] = v
        self.history.append(metrics)

    def get_dataframe(self):
        try:
            import pandas as pd  # noqa: WPS433

            return pd.DataFrame(self.history)
        except ImportError:
            return self.history
