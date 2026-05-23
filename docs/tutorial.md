# Constitution-Sim Tutorial (use it like you're 10)

This is the friendly walkthrough. No prior knowledge needed.

## 1. What this thing actually does

You write a constitution in YAML (a list of roles like Executive,
Legislature, Judiciary, …; what each role is allowed to do; what they
care about). The simulator takes that constitution and runs a little
game:

- Each "turn" one role gets to act.
- That role's agent — usually an **AI** like GPT — decides what to do:
  propose a law, vote, strike one down, publish a story, call an
  emergency, or do nothing.
- A **rules engine** checks if the action is legal under the
  constitution.
- If legal, the world state updates. If not, it's logged as an illegal
  attempt and ignored.
- Over many turns, you watch how the rules shape the politicians'
  behaviour — does power concentrate? Does the legislature deadlock?
  Do emergencies linger forever?

It's a flight-simulator for governments.

## 2. Install it

```bash
git clone https://github.com/arianXdev/constitution-sim.git
cd constitution-sim
pip install -e ".[dev,llm]"
```

You now have a command called `constitution-sim`.

## 3. Plug in AI politicians (recommended)

The whole point of this project is to simulate politicians' behaviour
with **agentic AI**. Set an OpenAI key in your environment:

```bash
export OPENAI_API_KEY=sk-...
```

That's it. The CLI auto-detects the key and uses GPT-4o-mini agents by
default. (If you'd rather use Anthropic, set `ANTHROPIC_API_KEY` and the
CLI will pick that.)

If you have *no* key, the CLI falls back to a deterministic heuristic
agent — same outputs, just less surprising behaviour. Everything below
works in both modes.

## 4. Your first 30-second simulation

```bash
constitution-sim run \
  --constitution constitutions/simple_constitution.yaml \
  --turns 6 \
  --log /tmp/my_first_run.jsonl
```

It prints which agent type was picked, runs 6 turns, and writes one
JSON line per event to `/tmp/my_first_run.jsonl`. Open the file —
that's the *whole audit trail*: every decision, who made it, whether
it was legal, why.

## 5. See what happened

```bash
constitution-sim replay --log /tmp/my_first_run.jsonl --show-first 6
```

You'll see something like:

```
Replay of /tmp/my_first_run.jsonl
  Events:        6
  Legal:         6
  Illegal:       0
  Turns covered: 0..5
  By actor:
    agent_executive     3
    agent_legislature   3
  By action:
    ProposeLaw          3
    VoteLaw             3

First events:
  t  0 agent_executive          ProposeLaw         OK  Action is legal.
  t  1 agent_legislature        VoteLaw            OK  Action is legal.
  ...
```

## 6. Run the bigger game (multiple roles + shocks + plots)

```bash
constitution-sim run \
  --constitution constitutions/advanced_constitution.yaml \
  --scenario     constitutions/scenario.yaml \
  --turns 30 --runs 5 --seed 42 \
  --log         /tmp/cs/events.jsonl \
  --metrics-out /tmp/cs/metrics.csv \
  --plot-dir    /tmp/cs/plots
```

Five seeded runs of 30 turns each, with all five roles, scenario
shocks firing in the middle, and `.png` plots written to
`/tmp/cs/plots/` (one per institutional metric: `legitimacy.png`,
`power_concentration.png`, `corruption_proxy.png`, …).

## 7. The headline experiment: compare two constitutions

```bash
# Run A: balanced (advanced)
constitution-sim run --constitution constitutions/advanced_constitution.yaml \
  --scenario constitutions/scenario.yaml --turns 12 --runs 3 --seed 11 \
  --log /tmp/A/events.jsonl --metrics-out /tmp/A/metrics.csv \
  --plot-dir /tmp/A/plots

# Run B: power-grab (strong executive)
constitution-sim run --constitution constitutions/strong_executive_constitution.yaml \
  --scenario constitutions/scenario.yaml --turns 12 --runs 3 --seed 11 \
  --log /tmp/B/events.jsonl --metrics-out /tmp/B/metrics.csv \
  --plot-dir /tmp/B/plots

# Compare
constitution-sim compare --a /tmp/A/metrics.csv --b /tmp/B/metrics.csv
```

You'll see something like (numbers will differ slightly with LLM agents):

```
metric                       A_mean       B_mean        delta
----------------------------------------------------------------
power_concentration          0.4729       0.9167      +0.4437
num_active_laws              0.4667       1.6667      +1.2000
corruption_proxy             0.0000       0.1667      +0.1667
legitimacy                   0.3641       0.3199      -0.0442
```

That `+0.44` jump in `power_concentration` and the appearance of
`corruption_proxy` is the simulator detecting that the strong-executive
constitution lets one actor concentrate lawmaking and generates
illegal-action attempts as the judiciary tries (and fails) to push
back. That's the framework working.

## 8. Editing a constitution

Open `constitutions/simple_constitution.yaml`. The structure:

```yaml
name: "My Constitution"
version: "1.0"
description: "Make it your own."
allow_emergency_powers: true

initial_state:
  variables:
    public_trust: 0.5
    budget: 1000.0

roles:
  Executive:
    name: "Executive"
    permissions: ["ProposeLaw", "DeclareEmergency", "DoNothing"]
    goals: ["pass legislation", "stabilise during shocks"]
    utility_weights:
      public_trust: 1.0    # cares a lot about public trust
      state_capacity: 0.5  # cares somewhat about capacity
    observation_limits:
      see_pending_bills: true
      see_active_shocks: true

  Legislature:
    name: "Legislature"
    permissions: ["VoteLaw", "DoNothing"]
    ...

rules:
  - name: "Executive Proposal"
    description: "Executive proposes laws."
    allowed_actions: ["ProposeLaw"]
    applies_to_roles: ["Executive"]
  ...
```

Knobs you can turn:

- **`permissions`**: which typed actions a role is allowed to propose.
  Drop `VoteLaw` from a role and they can't vote.
- **`goals`** and **`utility_weights`**: shown to the LLM in its
  persona. They tell the AI agent what it cares about.
- **`observation_limits`**: hide things from a role. Set
  `see_pending_bills: false` for the Bureaucracy and they'll never see
  bills.
- **`initial_state.variables`**: starting trust, budget, capacity, etc.
- **`allow_emergency_powers`**: master switch. If false,
  `DeclareEmergency` actions get rejected by the rules engine.
- **`rules`**: structured rule statements. The MVP rules engine
  honours role permissions; the `rules` block documents intent and
  is reserved for future structured constraints.

After editing, validate:

```bash
constitution-sim validate --constitution constitutions/my_constitution.yaml
```

## 9. Editing a scenario

`constitutions/scenario.yaml` lists *shocks* — sudden events that nudge the
world's variables:

```yaml
shocks:
  - id: "shock_1"
    name: "Economic Crisis"
    description: "A sudden economic downturn slashes budget and trust."
    duration_turns: 3
    trigger_turn: 2            # fires deterministically on turn 2
    effects:
      public_trust: -0.2
      budget: -100.0

  - id: "shock_2"
    name: "Corruption Scandal"
    duration_turns: 2
    trigger_probability: 0.05  # fires with 5% chance every turn
    effects:
      public_trust: -0.3
```

`trigger_turn` is deterministic; `trigger_probability` is random
per-turn (seeded). `effects` are deltas applied to world variables on
the turn the shock fires.

## 10. Flags worth knowing

| Flag                 | What it does                                                    |
|----------------------|------------------------------------------------------------------|
| `--constitution`     | Path to a constitution YAML.                                    |
| `--scenario`         | (Optional) path to a scenario YAML.                             |
| `--turns N`          | Number of turns per run.                                        |
| `--runs N`           | Number of seeded runs (>1 = multi-run evaluation).              |
| `--seed N`           | Base seed (run i uses `seed + i`).                              |
| `--agent-type`       | `auto` (default), `openai`, `anthropic`, `heuristic`, `llm-mock`. |
| `--model NAME`       | Override the LLM model (`gpt-4o-mini`, `claude-sonnet-4-5`, …). |
| `--log PATH`         | Where to write the per-event JSONL log.                         |
| `--metrics-out PATH` | Where to write the per-turn metrics CSV.                        |
| `--plot-dir PATH`    | Where to write `.png` plots (multi-run only).                   |

## 11. What metrics tell you

Each row in the metrics CSV is one turn of one run. The interesting
columns:

| metric              | reading                                                       |
|---------------------|---------------------------------------------------------------|
| `power_concentration` | 0 = laws spread across actors; 1 = one actor authored all of them. |
| `deadlock_counter`    | consecutive turns of non-progress on pending bills.        |
| `trust_volatility`    | how jumpy public_trust is turn-to-turn.                    |
| `legitimacy`          | trust × (1 − illegal-action rate). High = stable system.   |
| `corruption_proxy`    | total illegal-action attempts. A canary for stress.        |
| `emergency_active`    | 1 if an emergency is currently active.                     |
| `emergency_turns`     | cumulative turns spent under emergency powers.             |

## 12. Five experiments to try this weekend

1. **The dictator test.** Use `constitutions/strong_executive_constitution.yaml`.
   Watch `power_concentration` climb above 0.9. Then in the YAML, add
   `StrikeDownLaw` back to the Judiciary's `permissions` — re-run and
   watch it drop.
2. **The deadlock test.** In `advanced_constitution.yaml`, raise the
   Legislature's `public_trust` utility weight to 5.0 (so it's wary of
   most proposals). Watch `deadlock_counter` climb.
3. **The fog-of-war test.** Set
   `observation_limits.see_active_laws: false` for the Executive.
   They'll keep proposing without knowing the legislative graveyard.
4. **The emergency-creep test.** Set
   `initial_state.variables.public_trust: 0.1` in the advanced
   constitution. Watch the Executive's AI persona declare an emergency
   to "save the country" — and the `emergency_turns` metric climb.
5. **The shock test.** Add new shocks to `scenario.yaml` with
   `trigger_probability: 0.15` and `effects: { public_trust: -0.3 }`.
   Run 30 turns × 5 seeds. See how legitimacy looks across runs.

## 13. Reading the event log directly

`events.jsonl` has one line per event. Each line is:

```json
{
  "turn": 4,
  "actor_id": "agent_executive",
  "action_type": "ProposeLaw",
  "action_data": {"law_id": "law_4", "content": "..."},
  "is_legal": true,
  "reason": "Action is legal.",
  "timestamp": "2025-..."
}
```

For an illegal attempt, `is_legal` is `false` and `reason` tells you
exactly why the rules engine rejected it.

## 14. Cheat sheet

```bash
# Validate
constitution-sim validate --constitution constitutions/advanced_constitution.yaml

# One quick AI-powered simulation (auto-picks LLM if a key is set)
constitution-sim run \
  --constitution constitutions/advanced_constitution.yaml \
  --scenario     constitutions/scenario.yaml \
  --turns 10 --log /tmp/quick.jsonl

# Force the heuristic agent (deterministic, no API needed)
constitution-sim run --agent-type heuristic ...

# Replay
constitution-sim replay --log /tmp/quick.jsonl --show-first 10

# Compare two metrics CSVs
constitution-sim compare --a /tmp/A.csv --b /tmp/B.csv

# Run the test suite
pytest -q
```

## 15. When things go wrong

- **LLM returns garbage / 401 / rate-limit.** The agent logs a warning
  and falls back to the deterministic heuristic policy for that turn.
  The simulator never crashes.
- **Constitution YAML invalid.** `constitution-sim validate` will tell
  you exactly which field is malformed (Pydantic error path).
- **No plots appear.** Plots are only written when `--runs > 1`. Single
  runs only write the JSONL log and (optionally) the metrics CSV.
- **Determinism mismatch.** Only `--agent-type heuristic` is byte-for-byte
  reproducible. LLM mode is reproducible up to provider variance (and
  `temperature=0.0` already helps).

Have fun stress-testing constitutions!
