# Constitution-Sim

**Stress-test constitutions with AI-powered politicians before trying them out on a real nation.**

`constitution-sim` is a research-grade multi-agent simulator. You give it
a constitution and a scenario; it spins up an LLM-powered agent for each
political role (Executive, Legislature, Judiciary, Media, Bureaucracy)
and lets them act under the rules you wrote, turn by turn. Every action
is checked by a rules engine, every event is logged, and every run is
reproducible from a seed.

## Why

Politicians are not utility-maximisers reading from a spec — they
deliberate, bargain, posture, and reach for legitimacy. The interesting
question is *how the rules of a constitution shape that behaviour*. So
the agents here are LLMs (OpenAI / Anthropic) instructed with a
role-specific persona, the constitution they live under, their own
goals and utility weights, and a memory of their own recent decisions.
They never get to mutate the world directly — every move passes the
typed rules engine first.

A deterministic heuristic agent is still available as a no-LLM fallback,
so the project also runs offline / in CI / with zero API keys.

## Features

- **AI cognition is the default.** When `OPENAI_API_KEY` or
  `ANTHROPIC_API_KEY` is in the environment, `constitution-sim run`
  uses LLM-powered agents out of the box. With no key, it falls back to
  a deterministic heuristic — same CLI, same outputs, no setup required.
- **Role-specific personas.** Each role (Executive, Legislature,
  Judiciary, Media, Bureaucracy) gets its own LLM system prompt. The
  Executive is ambitious; the Judiciary is reactive; the Media chases a
  narrative; the Bureaucracy implements steadily.
- **Agent memory.** Each agent sees its own recent decisions (turn,
  action, legal or not) so it can reason about continuity.
- **Schema-driven constitutions.** Strict Pydantic v2 models; YAML in,
  typed objects out. Errors are explicit and structured.
- **Rules engine is source of truth.** Agents propose typed actions;
  the engine accepts or rejects with a reason. The LLM cannot mutate
  state directly.
- **Partial observability.** Each role gets a state view filtered by
  its `observation_limits`.
- **Institutional metrics.** Power concentration, deadlock, trust
  volatility, legitimacy, corruption pressure, emergency-power drift.
- **Repeated-run evaluation harness.** Multi-seed runs with pandas /
  matplotlib output.
- **Deterministic when seeded** (heuristic mode is byte-for-byte
  reproducible; LLM mode is reproducible up to provider variance).


  <img width="600" alt="corruption_proxy" src="https://github.com/user-attachments/assets/f6127e60-0db6-4c34-a996-361dfde64c27" />

  <img width="600" alt="num_active_laws" src="https://github.com/user-attachments/assets/d5959e29-70ce-4a1e-bc4e-ea216e1a08b5" />

<img width="600"  alt="power_concentration" src="https://github.com/user-attachments/assets/0bdc1b80-07d7-4c51-901e-dd38dc598483" />

<img width="600" alt="public_trust" src="https://github.com/user-attachments/assets/b004146e-51ed-43ef-a170-e8351982b702" />

<img width="600" alt="num_active_laws" src="https://github.com/user-attachments/assets/06955b78-383b-4249-b6cb-a05ce6db8080" />

<img width="600" alt="emergency_turns" src="https://github.com/user-attachments/assets/85d8bbcd-b605-423f-bf95-3891e4592c66" />


## Requirements

- Python 3.10+ (target: 3.14)
- `pydantic >= 2`, `PyYAML`, `pandas`, `matplotlib`, `seaborn`
- For AI cognition: `openai` (and/or `anthropic`)

## Install

```bash
git clone https://github.com/arianXdev/constitution-sim.git
cd constitution-sim
pip install -e ".[dev,llm]"     # core + tests + LLM SDKs (recommended)
# or, no-LLM-only install:
pip install -e ".[dev]"
```

This exposes a `constitution-sim` console entry point.

## Quickstart (AI-powered)

```bash
export OPENAI_API_KEY=sk-...
constitution-sim run \
  --constitution examples/advanced_constitution.yaml \
  --scenario     examples/scenario.yaml \
  --turns 20 --seed 42 \
  --log         /tmp/cs/events.jsonl \
  --metrics-out /tmp/cs/metrics.csv
```

That's it. The default `--agent-type auto` notices the key, spins up
LLM-powered Executive / Legislature / Judiciary / Media / Bureaucracy
agents, and runs the simulation. You'll see a one-liner telling you
which provider was picked.

Want to force a provider explicitly?

```bash
constitution-sim run --agent-type openai    --model gpt-4o-mini       ...
constitution-sim run --agent-type anthropic --model claude-sonnet-4-5 ...
```

Want deterministic, no-API runs (for tests / reproducibility)?

```bash
constitution-sim run --agent-type heuristic ...
```

## The four CLI subcommands

```bash
# 1. Validate a constitution YAML against the schema.
constitution-sim validate --constitution examples/advanced_constitution.yaml

# 2. Run a simulation (single seed or multi-seed evaluation).
constitution-sim run \
  --constitution examples/advanced_constitution.yaml \
  --scenario     examples/scenario.yaml \
  --turns 30 --runs 5 --seed 42 \
  --log         /tmp/cs/events.jsonl \
  --metrics-out /tmp/cs/metrics.csv \
  --plot-dir    /tmp/cs/plots

# 3. Replay a recorded event log (structured summary, not re-execution).
constitution-sim replay --log /tmp/cs/eval_logs/run_0_events.jsonl --show-first 5

# 4. Compare two evaluations (e.g. two constitutions).
constitution-sim compare --a /tmp/cs/metrics_A.csv --b /tmp/cs/metrics_B.csv
```

## What the LLM sees

For each turn, the LLM agent is prompted with:

- A role-specific persona (Executive / Legislature / …).
- The constitution's name, description, and the list of other roles.
- Its own declared goals and utility weights (from the YAML).
- A partial state view filtered by its `observation_limits`.
- A short memory of its own recent decisions (and whether they were
  legal).
- The exact set of typed actions it's allowed to return.

It replies with one JSON object describing a single action. If the LLM
returns malformed JSON or an action outside its permission set, the
agent silently falls back to the deterministic heuristic policy — the
simulator never breaks.

## Project structure

```
src/constitution_sim/
  models/        Pydantic schemas: Constitution, Role, Rule, WorldState, actions
  core/          SimulationEngine, RulesEngine, Scheduler, EventLogger
  agents/        BaseAgent, DeterministicHeuristicAgent, LLMAgent, providers
  scenarios/     Shock model + ScenarioEngine
  analysis/      MetricsCollector, Evaluator, plot
  app/           CLI (validate / run / replay / compare)
examples/
  simple_constitution.yaml
  advanced_constitution.yaml
  strong_executive_constitution.yaml
  scenario.yaml
docs/
  architecture.md
  tutorial.md
tests/
```

## Tests

```bash
pytest -q
```

All tests should pass. `tests/test_determinism.py` explicitly asserts
that two heuristic-mode runs with the same seed produce byte-identical
event logs. `tests/test_llm_agent.py::test_live_openai_smoke` runs a
real LLM round-trip when `OPENAI_API_KEY` is set, and is automatically
skipped otherwise.

## Headline experiment

Compare a balanced constitution against a strong-executive one (3 runs ×
12 turns, seed 11). The strong-executive YAML pushes power_concentration
from ~0.47 to ~0.92 and adds illegal-action attempts to the log: laws
written by one actor, judiciary unable to push back. That's the
framework working as intended — see `docs/tutorial.md` for a walkthrough.

## Design highlights

- `WorldState` is the single canonical truth; agents only ever see a
  `StateView`.
- Every action attempt is recorded in the JSONL event log, including
  the rules-engine reason for any rejection.
- `Role.observation_limits` lets the constitution define what each role
  can see (e.g. the Bureaucracy doesn't see pending bills in
  `advanced_constitution.yaml`).
- `Role.utility_weights` drives heuristic voting and is surfaced to
  LLM agents in their prompt as part of the persona.
- `RulesEngine` does both permission checks AND state-level legality
  checks (you can't vote on a non-existent bill, you can't declare
  emergency powers if the constitution doesn't allow them).

See [`docs/architecture.md`](docs/architecture.md) for the full design
and [`docs/tutorial.md`](docs/tutorial.md) for an end-to-end "use it
like I'm 10" walkthrough.

## Out of scope (intentional)

This is an MVP, not a finished research instrument. The following are
explicit non-goals at this stage:

- Multi-actor coalition formation / strategic communication.
- Persistent economic/demographic simulation (state variables are
  scalars, not vector economies).
- Fine-tuned LLMs or RL self-play.
