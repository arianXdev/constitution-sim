# Architecture: Constitution-Sim

## Overview

`constitution-sim` is a multi-agent Python framework for simulating
politicians' behaviour under different constitutional regimes. The
primary cognition path is **LLM-powered agents**: each role
(Executive, Legislature, Judiciary, Media, Bureaucracy) is driven by an
LLM with a role-specific persona, the constitution it operates under,
its own goals and utility weights, and a memory of its recent
decisions. A deterministic heuristic agent is available as a no-LLM
fallback so the entire simulator runs offline / in CI / with zero API
keys.

The system cleanly separates **the rules of the game (legality)** from
**the choices of the actors (cognition)**, so different constitutions
and different cognition strategies are swappable independently.

## Core design principles

1. **AI cognition is the headline.** `--agent-type auto` (the default)
   picks an LLM provider from environment keys; heuristic is the
   reproducibility-first fallback. The simulator never depends on the
   LLM being correct — every action still goes through the typed
   rules engine.
2. **Source of truth.** The `RulesEngine` is the sole arbiter of
   legality. Agents propose typed actions; the engine accepts or
   rejects. The LLM layer can never mutate `WorldState` directly.
3. **Reproducible.** Heuristic-mode runs are byte-for-byte deterministic
   given (constitution YAML, scenario YAML, seed, code version).
   `tests/test_determinism.py` asserts this.
4. **Schema-driven.** Constitutions, roles, rules, and actions are all
   strict Pydantic v2 models. YAML is the only on-disk format.
5. **Modularity.** `models/`, `core/`, `agents/`, `scenarios/`,
   `analysis/`, `app/` each have a narrow, well-defined contract.

## Directory structure

```
src/constitution_sim/
  models/
    constitution.py    Constitution, Role, Rule, ObservationLimits, InitialState
    state.py           WorldState (canonical) and StateView (per-role projection)
    actions.py         Typed action classes
    events.py          EventRecord (one row per logged decision)
    messages.py        Message and DealProposal models for inter-agent communication
  core/
    engine.py          SimulationEngine + EventLogger
    message_bus.py     MessageBus (handles message routing and filters)
    rules.py           RulesEngine (legality decisions + reasons)
    scheduler.py       Round-robin scheduler
  agents/
    base.py            BaseAgent ABC
    heuristics.py      DeterministicHeuristicAgent (utility-biased, deterministic)
    llm.py             LLMAgent (role-aware prompts, memory, deterministic fallback)
    providers.py       OpenAI / Anthropic provider adapters
  scenarios/
    shocks.py          Shock schema
    engine.py          ScenarioEngine (per-turn shock evaluation)
  analysis/
    metrics.py         MetricsCollector (per-turn snapshots)
    evaluator.py       Multi-run evaluation harness
    plot.py            Mean ± SD plots
  app/
    cli.py             validate / run / replay / compare
```

## Data flow

```
            YAML constitution    YAML scenario
                   |                  |
                   v                  v
            Constitution            [Shock]
                   |                  |
                   v                  v
            RulesEngine     ScenarioEngine
                   \              /
                    \            /
                  SimulationEngine ----> EventLogger (.jsonl)
                  /   |    |    \
                 /    |    |     \
         Scheduler Messages Agents MetricsCollector
                           |          \
                        StateView      DataFrame ----> plots + compare
```

Per turn:

1. **Deliberation Phase**: All agents generate messages (`communicate()`) which are routed via the `MessageBus`.
2. `Scheduler.get_next_actor()` returns an `actor_id`.
3. `SimulationEngine.get_state_view(role_name)` builds a partial view
   that respects the role's `ObservationLimits`, including the public political history.
4. The agent's `decide_with_messages(state_view, inbox)` returns a typed `Action`.
5. `RulesEngine.is_legal(role, action, state)` returns
   `(bool, reason)`. The reason is always stored.
6. `EventLogger.log(EventRecord)` records the attempt — legal or not.
7. If legal, `SimulationEngine.apply_action(actor_id, action, state)`
   mutates the `WorldState` and records the public action in `state.recent_actions`.
8. The engine calls `agent.remember(turn, action_type, is_legal)` so
   agents that maintain memory (LLM) can see their own history next
   turn.
9. `ScenarioEngine.tick(state)` ages active shocks and triggers new
   ones.
10. `MetricsCollector.collect(state)` snapshots metrics.
11. Turn counter increments.

## Agent cognition

### LLM agent (primary path)

`LLMAgent` is the primary cognition model. For each decision it builds
a prompt that includes:

- A **role-specific persona** (Executive, Legislature, Judiciary,
  Media, Bureaucracy) baked into `ROLE_PERSONAS` in
  `agents/llm.py`.
- The **constitution context**: name, description, list of other roles.
- The agent's declared **goals** and **utility weights** from the YAML.
- A **filtered state view** (per-role observation limits applied).
- **Public political history**: recently executed actions by all actors (if permitted).
- **Inbox**: messages sent by other actors during the current turn's deliberation phase.
- A **rolling memory** of the agent's own recent decisions and whether
  they were legal.
- The **exact set of typed actions** the role is allowed to return.

The LLM replies with a single JSON object describing one action.
`agents/providers.py` enforces JSON response_format (OpenAI) and
JSON-only system instructions (Anthropic) at `temperature=0.0`. Any
parse error or schema violation falls back to the deterministic
heuristic — the simulator never breaks on LLM failure.

### Heuristic agent (fallback)

`DeterministicHeuristicAgent` is a small utility-biased policy used
when no LLM is configured (or the LLM call fails). It implements:

- Utility-based action selection (linear utility over observed state
  variables, weighted by `Role.utility_weights`).
- Status-quo bias (Executive caps proposals at 8 active laws).
- Confirmation bias (±0.05 nudge to vote probability).
- Loss aversion / variance (Bureaucracy and Media efficiencies are
  drawn from a Gaussian around their setpoint).

All randomness flows through a single seeded `random.Random` per agent
— perfectly reproducible.

## Partial observability

`Role.observation_limits` (`ObservationLimits`) controls what fields of
the `WorldState` flow into the agent's `StateView`. Examples in
`constitutions/advanced_constitution.yaml`:

- Bureaucracy: `see_pending_bills: false` — bureaucrats don't see
  drafts.
- Judiciary: `variable_allowlist: [public_trust, state_capacity]` —
  the judiciary doesn't peek at the budget.

The view is identical for heuristic and LLM agents — the LLM does not
get more visibility than the rules say it should.

## Metrics

The `MetricsCollector` snapshots per turn:

| metric              | what it captures                                            |
|---------------------|-------------------------------------------------------------|
| power_concentration | share of active laws authored by the single top actor       |
| deadlock_counter    | consecutive turns of non-progress on pending bills          |
| trust_volatility    | abs(Δ public_trust) per turn                                |
| legitimacy          | trust × (1 − illegal-action rate)                           |
| corruption_proxy    | total illegal-action attempts                               |
| emergency_active    | 1 if a state of emergency is currently active               |
| emergency_turns     | cumulative turns spent under emergency powers               |
| communication_volume| number of messages sent over the message bus                |
| active_coalitions   | number of formally declared coalitions                      |

Plus the raw `state.variables` and counts of laws / bills / shocks.

## Reproducibility contract

- **Heuristic mode** (`--agent-type heuristic`): for a fixed
  (constitution, scenario, seed, code commit) tuple, the engine
  produces the same event log every time. The `timestamp` field on
  `EventRecord` is wall-clock — strip it before byte-comparing logs
  (`test_determinism.py` does this).
- **LLM mode** is reproducible up to provider variance.
  `temperature=0.0` and JSON response format minimise this, but
  byte-identical event logs across LLM runs are not guaranteed and not
  tested.

## CLI agent selection

`--agent-type` resolves as:

- `auto` (default): use `openai` if `OPENAI_API_KEY` is set, else
  `anthropic` if `ANTHROPIC_API_KEY` is set, else `heuristic`. Prints
  a one-line note explaining the choice.
- `openai` / `anthropic`: force that provider (errors if no key).
- `heuristic`: deterministic, no-LLM.
- `llm-mock`: `LLMAgent` with no callable wired in (behaviour
  identical to `heuristic`; used by unit tests).
