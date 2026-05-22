"""Command-line interface.

Subcommands:
  validate    Validate a constitution YAML against the schema.
  run         Run a single simulation (or multi-run evaluation) from YAML.
  replay      Replay a previously-recorded event log and print a summary.
  compare     Compare two evaluation result CSVs across key metrics.

Agent selection (`--agent-type`):
  auto       Pick the first available LLM provider from environment keys,
             else fall back to `heuristic`. This is the default so the
             headline experience is AI-driven.
  openai     Force the OpenAI provider (requires OPENAI_API_KEY).
  anthropic  Force the Anthropic provider (requires ANTHROPIC_API_KEY).
  heuristic  Deterministic, no-LLM agent. Use this for the reproducibility
             tests or when running on a box with no API access.
  llm-mock   `LLMAgent` with no callable wired in (heuristic via fallback).
             Used by unit tests; identical behaviour to `heuristic`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from constitution_sim.agents.heuristics import DeterministicHeuristicAgent
from constitution_sim.agents.llm import LLMAgent
from constitution_sim.analysis.evaluator import Evaluator
from constitution_sim.analysis.metrics import MetricsCollector
from constitution_sim.analysis.plot import plot_metrics
from constitution_sim.core.engine import EventLogger, SimulationEngine
from constitution_sim.core.rules import RulesEngine
from constitution_sim.core.scheduler import Scheduler
from constitution_sim.models.constitution import Constitution
from constitution_sim.models.state import WorldState
from constitution_sim.scenarios.engine import ScenarioEngine
from constitution_sim.scenarios.shocks import Shock


def load_constitution(path: Path) -> Constitution:
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    return Constitution(**data)


def load_shocks(path: Path) -> List[Shock]:
    with Path(path).open("r") as f:
        data = yaml.safe_load(f) or {}
    return [Shock(**s) for s in data.get("shocks", [])]


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.constitution)
    try:
        constitution = load_constitution(path)
    except FileNotFoundError:
        print(f"ERROR: constitution file not found: {path}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        print(f"ERROR: invalid YAML in {path}: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"ERROR: constitution schema invalid:\n{exc}", file=sys.stderr)
        return 1

    print(f"Constitution '{constitution.name}' v{constitution.version} OK.")
    print(f"  Roles ({len(constitution.roles)}): {list(constitution.roles)}")
    print(f"  Rules ({len(constitution.rules)})")
    print(f"  Emergency powers allowed: {constitution.allow_emergency_powers}")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _resolve_agent_type(agent_type: str) -> str:
    """Resolve `auto` to a concrete provider based on available API keys.

    Order of preference is OpenAI then Anthropic then heuristic. Print a
    one-line note so it's obvious which path the run is using.
    """
    if agent_type != "auto":
        return agent_type
    if os.environ.get("OPENAI_API_KEY"):
        print("Auto-selected agent type: openai (OPENAI_API_KEY detected).")
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Auto-selected agent type: anthropic (ANTHROPIC_API_KEY detected).")
        return "anthropic"
    print(
        "Auto-selected agent type: heuristic (no LLM API key found; "
        "set OPENAI_API_KEY or ANTHROPIC_API_KEY for AI agents)."
    )
    return "heuristic"


def _build_agent(
    agent_type: str,
    model: str | None,
    agent_id: str,
    role_name: str,
    seed: int,
    role,
    constitution: Constitution,
):
    """Construct an agent for a single role."""
    other_roles = [n for n in constitution.roles if n != role_name]

    if agent_type == "heuristic":
        return DeterministicHeuristicAgent(
            agent_id, role_name, seed, utility_weights=role.utility_weights
        )
    if agent_type == "openai":
        from constitution_sim.agents.providers import (
            DEFAULT_OPENAI_MODEL,
            get_openai_callable,
        )

        callable_llm = get_openai_callable(model=model or DEFAULT_OPENAI_MODEL)
        return LLMAgent(
            agent_id,
            role_name,
            llm_callable=callable_llm,
            fallback_seed=seed,
            goals=role.goals,
            utility_weights=role.utility_weights,
            allowed_actions=role.permissions,
            constitution_name=constitution.name,
            constitution_description=constitution.description or "",
            other_roles=other_roles,
        )
    if agent_type == "anthropic":
        from constitution_sim.agents.providers import (
            DEFAULT_ANTHROPIC_MODEL,
            get_anthropic_callable,
        )

        callable_llm = get_anthropic_callable(model=model or DEFAULT_ANTHROPIC_MODEL)
        return LLMAgent(
            agent_id,
            role_name,
            llm_callable=callable_llm,
            fallback_seed=seed,
            goals=role.goals,
            utility_weights=role.utility_weights,
            allowed_actions=role.permissions,
            constitution_name=constitution.name,
            constitution_description=constitution.description or "",
            other_roles=other_roles,
        )
    # llm-mock: LLMAgent with no callable => deterministic fallback.
    return LLMAgent(
        agent_id,
        role_name,
        llm_callable=None,
        fallback_seed=seed,
        goals=role.goals,
        utility_weights=role.utility_weights,
        allowed_actions=role.permissions,
        constitution_name=constitution.name,
        constitution_description=constitution.description or "",
        other_roles=other_roles,
    )


def _agent_factory_for(args: argparse.Namespace, constitution: Constitution):
    """Return an agent_factory(roles, seed) closure for the Evaluator."""
    resolved = _resolve_agent_type(args.agent_type)

    def factory(roles_dict, seed: int):
        agents = {}
        for idx, (role_name, role) in enumerate(roles_dict.items()):
            agent_id = f"agent_{role_name.lower()}"
            agents[agent_id] = _build_agent(
                resolved, args.model, agent_id, role_name, seed + idx, role,
                constitution,
            )
        return agents

    return factory


def cmd_run(args: argparse.Namespace) -> int:
    constitution = load_constitution(Path(args.constitution))
    shocks = load_shocks(Path(args.scenario)) if args.scenario else []

    factory = _agent_factory_for(args, constitution)

    if args.runs <= 1:
        state = WorldState(variables=dict(constitution.initial_state.variables))
        rules = RulesEngine(constitution)
        agents = factory(constitution.roles, args.seed)
        scheduler = Scheduler(list(agents.keys()))
        logger = EventLogger(Path(args.log))
        scenario_engine = ScenarioEngine(shocks, args.seed) if shocks else None
        metrics_collector = MetricsCollector()

        engine = SimulationEngine(
            state=state,
            rules=rules,
            scheduler=scheduler,
            logger=logger,
            agents=agents,
            scenario_engine=scenario_engine,
            metrics_collector=metrics_collector,
            constitution=constitution,
        )

        print(f"Running single simulation for {args.turns} turns...")
        for _ in range(args.turns):
            engine.run_turn()
        print(f"Simulation complete. Events logged to {args.log}.")

        if args.metrics_out:
            df = metrics_collector.get_dataframe()
            out = Path(args.metrics_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(df, "to_csv"):
                df.to_csv(out, index=False)
            print(f"Metrics CSV written to {out}.")
        return 0

    print(f"Running {args.runs} evaluation runs of {args.turns} turns each...")
    evaluator = Evaluator(constitution, shocks, factory, args.turns)
    log_dir = Path(args.log).parent / "eval_logs"
    df = evaluator.run_evaluations(args.runs, args.seed, log_dir)
    print(f"Evaluations complete. Logs in {log_dir}.")

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Metrics CSV written to {out}.")

    plot_dir = Path(args.plot_dir)
    plot_metrics(df, plot_dir)
    print(f"Plots saved to {plot_dir}.")
    return 0


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def cmd_replay(args: argparse.Namespace) -> int:
    """Read an event log and print a structured summary.

    Replay is *trace-level*: we do not re-execute the simulation, we
    walk the recorded event log and summarise. This is the right
    abstraction because runs are deterministic — to bit-exactly
    re-run, use `constitution-sim run` with the same seed/inputs.
    """
    path = Path(args.log)
    if not path.exists():
        print(f"ERROR: log file not found: {path}", file=sys.stderr)
        return 2

    events: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    if not events:
        print("Empty event log.")
        return 0

    legal = sum(1 for e in events if e.get("is_legal"))
    illegal = len(events) - legal
    by_actor: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for e in events:
        by_actor[e["actor_id"]] = by_actor.get(e["actor_id"], 0) + 1
        by_action[e["action_type"]] = by_action.get(e["action_type"], 0) + 1

    print(f"Replay of {path}")
    print(f"  Events:        {len(events)}")
    print(f"  Legal:         {legal}")
    print(f"  Illegal:       {illegal}")
    print(f"  Turns covered: {events[0]['turn']}..{events[-1]['turn']}")
    print("  By actor:")
    for actor, count in sorted(by_actor.items()):
        print(f"    {actor:<32} {count}")
    print("  By action:")
    for action, count in sorted(by_action.items()):
        print(f"    {action:<32} {count}")

    if args.show_first:
        print("\nFirst events:")
        for e in events[: args.show_first]:
            ok = "OK" if e["is_legal"] else "REJECT"
            print(
                f"  t{e['turn']:>3} {e['actor_id']:<24} "
                f"{e['action_type']:<18} {ok}  {e.get('reason','')}"
            )
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two metrics CSVs (produced by `run --metrics-out`)."""
    import pandas as pd

    a_path = Path(args.a)
    b_path = Path(args.b)
    if not a_path.exists() or not b_path.exists():
        print("ERROR: both --a and --b CSV paths must exist.", file=sys.stderr)
        return 2

    df_a = pd.read_csv(a_path)
    df_b = pd.read_csv(b_path)

    metrics_to_compare = [
        c
        for c in [
            "public_trust",
            "num_active_laws",
            "num_pending_bills",
            "power_concentration",
            "deadlock_counter",
            "trust_volatility",
            "legitimacy",
            "corruption_proxy",
            "emergency_turns",
        ]
        if c in df_a.columns and c in df_b.columns
    ]

    print(f"Comparing:\n  A = {a_path}\n  B = {b_path}\n")
    print(f"{'metric':<24} {'A_mean':>12} {'B_mean':>12} {'delta':>12}")
    print("-" * 64)
    for m in metrics_to_compare:
        a_mean = float(df_a[m].mean())
        b_mean = float(df_b[m].mean())
        delta = b_mean - a_mean
        print(f"{m:<24} {a_mean:>12.4f} {b_mean:>12.4f} {delta:>+12.4f}")
    return 0


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="constitution-sim",
        description="Constitution Simulator: stress-test constitutions with bounded-rational agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a constitution YAML.")
    p_val.add_argument("--constitution", required=True, help="Path to constitution YAML.")
    p_val.set_defaults(func=cmd_validate)

    # run
    p_run = subparsers.add_parser("run", help="Run a simulation from YAML.")
    p_run.add_argument("--constitution", required=True, help="Path to constitution YAML.")
    p_run.add_argument("--scenario", default=None, help="Optional scenario YAML.")
    p_run.add_argument("--turns", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--log", default="events.jsonl", help="Event log output path.")
    p_run.add_argument(
        "--agent-type",
        choices=["auto", "heuristic", "llm-mock", "openai", "anthropic"],
        default="auto",
        help=(
            "auto (default): pick OpenAI or Anthropic if a matching API key "
            "is set, else heuristic. heuristic: deterministic no-LLM path. "
            "llm-mock: LLMAgent wired to the heuristic fallback."
        ),
    )
    p_run.add_argument("--model", default=None, help="LLM model name (provider-specific).")
    p_run.add_argument("--runs", type=int, default=1, help="Repeated runs (>1 = eval mode).")
    p_run.add_argument("--plot-dir", default="plots", help="Where to write plots.")
    p_run.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to write per-turn metrics CSV.",
    )
    p_run.set_defaults(func=cmd_run)

    # replay
    p_rep = subparsers.add_parser("replay", help="Summarise a recorded event log.")
    p_rep.add_argument("--log", required=True, help="Path to events JSONL log.")
    p_rep.add_argument(
        "--show-first",
        type=int,
        default=0,
        help="Print the first N events in detail (default: 0).",
    )
    p_rep.set_defaults(func=cmd_replay)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare two metrics CSV files.")
    p_cmp.add_argument("--a", required=True, help="First metrics CSV.")
    p_cmp.add_argument("--b", required=True, help="Second metrics CSV.")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    # Back-compat: if no subcommand is given but --constitution is in argv,
    # treat it as `run` so old invocations keep working.
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        parser.print_help()
        return 0
    known_subcommands = {"validate", "run", "replay", "compare", "-h", "--help"}
    if args[0] not in known_subcommands:
        args = ["run", *args]

    namespace = parser.parse_args(args)
    if not hasattr(namespace, "func"):
        parser.print_help()
        return 0
    return int(namespace.func(namespace) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
