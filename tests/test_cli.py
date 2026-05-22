"""Tests for the CLI subcommands."""

from constitution_sim.app.cli import _resolve_agent_type, main


def test_cli_validate_simple_ok(capsys):
    rc = main(["validate", "--constitution", "examples/simple_constitution.yaml"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Simple Constitution" in captured


def test_cli_validate_missing_file_errors(capsys, tmp_path):
    rc = main(["validate", "--constitution", str(tmp_path / "nope.yaml")])
    assert rc != 0


def test_cli_run_single(tmp_path):
    log = tmp_path / "events.jsonl"
    metrics = tmp_path / "metrics.csv"
    rc = main(
        [
            "run",
            "--constitution",
            "examples/simple_constitution.yaml",
            "--turns",
            "4",
            "--seed",
            "7",
            "--log",
            str(log),
            "--metrics-out",
            str(metrics),
            # Force heuristic so CI doesn't hit a paid API even when
            # OPENAI_API_KEY happens to be set in the runner env.
            "--agent-type",
            "heuristic",
        ]
    )
    assert rc == 0
    assert log.exists()
    assert metrics.exists()
    # Each turn -> 1 logged event (round-robin with 2 agents = 4 events).
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 4


def test_cli_backcompat_without_subcommand(tmp_path):
    log = tmp_path / "events.jsonl"
    rc = main(
        [
            "--constitution",
            "examples/simple_constitution.yaml",
            "--turns",
            "2",
            "--log",
            str(log),
            "--agent-type",
            "heuristic",
        ]
    )
    assert rc == 0
    assert log.exists()


def test_cli_replay(tmp_path, capsys):
    # First produce a log
    log = tmp_path / "events.jsonl"
    main(
        [
            "run",
            "--constitution",
            "examples/simple_constitution.yaml",
            "--turns",
            "3",
            "--log",
            str(log),
            "--agent-type",
            "heuristic",
        ]
    )
    # Then replay it
    rc = main(["replay", "--log", str(log)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Replay" in captured
    assert "Events:" in captured


def test_resolve_agent_type_auto_picks_openai(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _resolve_agent_type("auto") == "openai"


def test_resolve_agent_type_auto_picks_anthropic(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anth")
    assert _resolve_agent_type("auto") == "anthropic"


def test_resolve_agent_type_auto_falls_back_to_heuristic(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _resolve_agent_type("auto") == "heuristic"


def test_resolve_agent_type_pass_through():
    assert _resolve_agent_type("heuristic") == "heuristic"
    assert _resolve_agent_type("openai") == "openai"


def test_cli_compare(tmp_path, capsys):
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    main(
        [
            "run",
            "--constitution",
            "examples/advanced_constitution.yaml",
            "--scenario",
            "examples/scenario.yaml",
            "--turns",
            "5",
            "--runs",
            "2",
            "--seed",
            "1",
            "--log",
            str(log_a),
            "--metrics-out",
            str(csv_a),
            "--plot-dir",
            str(tmp_path / "plots_a"),
            "--agent-type",
            "heuristic",
        ]
    )
    main(
        [
            "run",
            "--constitution",
            "examples/strong_executive_constitution.yaml",
            "--scenario",
            "examples/scenario.yaml",
            "--turns",
            "5",
            "--runs",
            "2",
            "--seed",
            "1",
            "--log",
            str(log_b),
            "--metrics-out",
            str(csv_b),
            "--plot-dir",
            str(tmp_path / "plots_b"),
            "--agent-type",
            "heuristic",
        ]
    )
    rc = main(["compare", "--a", str(csv_a), "--b", str(csv_b)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "metric" in captured
    # The legitimacy metric should appear in the comparison table.
    assert "legitimacy" in captured
