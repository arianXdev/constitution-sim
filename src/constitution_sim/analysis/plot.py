"""Lightweight plotting helpers for evaluation results."""

from pathlib import Path

import pandas as pd

# Metrics worth plotting by default. The list intentionally over-includes;
# missing columns are silently skipped.
DEFAULT_METRICS = [
    "public_trust",
    "num_active_laws",
    "num_pending_bills",
    "power_concentration",
    "deadlock_counter",
    "trust_volatility",
    "legitimacy",
    "corruption_proxy",
    "emergency_turns",
    "state_capacity",
    "budget",
]


def plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean ± SD of each tracked metric over turns into `output_dir`.

    Silently no-ops if matplotlib/seaborn are missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # safe in headless environments
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed; skipping plots.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "turn" not in df.columns:
        print("No data to plot.")
        return

    for metric in DEFAULT_METRICS:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="turn", y=metric, errorbar="sd")
        plt.title(f"{metric} over Time (Mean ± 1 SD)")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png")
        plt.close()
