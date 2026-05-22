"""Repeated-run evaluation harness.

Runs the same constitution + scenario across N seeded trials and returns
a flat pandas DataFrame of per-turn metrics indexed by (run_id, turn).
"""

import copy
from pathlib import Path
from typing import Callable, List

import pandas as pd

from constitution_sim.analysis.metrics import MetricsCollector
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.constitution import Constitution
from constitution_sim.models.state import WorldState
from constitution_sim.scenarios.engine import ScenarioEngine
from constitution_sim.scenarios.shocks import Shock


class Evaluator:
    def __init__(
        self,
        constitution: Constitution,
        shocks: List[Shock],
        agent_factory: Callable,
        num_turns: int,
    ):
        self.constitution = constitution
        self.shocks = shocks
        self.agent_factory = agent_factory
        self.num_turns = num_turns

    def run_evaluations(
        self, num_runs: int, base_seed: int, log_dir: Path
    ) -> pd.DataFrame:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        all_metrics: List[pd.DataFrame] = []

        for i in range(num_runs):
            seed = base_seed + i
            state = WorldState(
                variables=dict(self.constitution.initial_state.variables)
            )
            rules = RulesEngine(self.constitution)

            agents = self.agent_factory(self.constitution.roles, seed)
            scheduler = Scheduler(list(agents.keys()))
            logger = EventLogger(log_dir / f"run_{i}_events.jsonl")

            run_shocks = copy.deepcopy(self.shocks)
            scenario_engine = ScenarioEngine(run_shocks, seed)
            metrics_collector = MetricsCollector()

            engine = SimulationEngine(
                state=state,
                rules=rules,
                scheduler=scheduler,
                logger=logger,
                agents=agents,
                scenario_engine=scenario_engine,
                metrics_collector=metrics_collector,
                constitution=self.constitution,
            )

            for _ in range(self.num_turns):
                engine.run_turn()

            df = metrics_collector.get_dataframe()
            if isinstance(df, pd.DataFrame):
                df["run_id"] = i
                df["seed"] = seed
                all_metrics.append(df)

        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        return pd.DataFrame()
