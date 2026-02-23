#!/usr/bin/env python3
"""Plot evaluation rewards from all experiments in the logs folder."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_eval_rewards(csv_path: Path) -> np.ndarray:
    """Load eval_rewards.csv - each row is an evaluation epoch."""
    rewards = []
    with open(csv_path) as f:
        for line in f:
            values = [float(x) for x in line.strip().split(",") if x]
            rewards.append(values)
    return np.array(rewards)


def get_newest_run(experiment_dir: Path) -> Path | None:
    """Get the newest run directory (sorted by timestamp name)."""
    runs = sorted(experiment_dir.iterdir())
    return runs[-1] if runs else None


def extract_label(experiment_name: str) -> tuple[str, str]:
    """Extract the label and env from experiment name like 'appo_CartPole-v1_abc123'.

    Returns:
        A tuple of (label, env_name) where label is like 'appo_abc123'.
    """
    parts = experiment_name.split("_")
    if len(parts) >= 3:
        exp_type = parts[0]  # e.g., 'appo' or 'ppo'
        env_name = parts[1]  # e.g., 'CartPole-v1'
        exp_hash = parts[-1]  # the hash
        return f"{exp_type}_{exp_hash}", env_name
    return experiment_name, "Unknown"


def main():
    logs_dir = Path(__file__).parent / "logs"

    plt.figure(figsize=(10, 6))

    env_names = set()

    for experiment_dir in sorted(logs_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        newest_run = get_newest_run(experiment_dir)
        if newest_run is None:
            continue

        eval_rewards_path = newest_run / "eval_rewards.csv"
        if not eval_rewards_path.exists():
            print(f"Skipping {experiment_dir.name}: no eval_rewards.csv")
            continue

        rewards = load_eval_rewards(eval_rewards_path)
        mean_rewards = rewards.mean(axis=1)
        std_rewards = rewards.std(axis=1)
        epochs = np.arange(1, len(mean_rewards) + 1)

        label, env_name = extract_label(experiment_dir.name)
        env_names.add(env_name)

        (line,) = plt.plot(epochs, mean_rewards, label=label)
        plt.fill_between(
            epochs,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            color=line.get_color(),
        )

    env_str = ", ".join(sorted(env_names)) if env_names else "Unknown"
    plt.xlabel("Epoch")
    plt.ylabel("Evaluation Reward")
    plt.title(f"Evaluation Rewards ({env_str})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(logs_dir.parent / "eval_rewards_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
