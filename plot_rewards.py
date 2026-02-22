"""
Script to plot evaluation rewards from multiple experiments.

This script processes all eval_rewards.csv files found in the logs directory,
computes statistics (mean, min, max) across episodes (axis=-1), and generates
plots with metadata from config.json files.
"""

import json
from pathlib import Path
from typing import (
    Any,
    Dict,
)

import matplotlib.pyplot as plt
import numpy as np


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_rewards(csv_path: Path) -> np.ndarray:
    """Load rewards from a CSV file."""
    return np.loadtxt(csv_path, delimiter=",")


def compute_statistics(
    rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, min, and max statistics across the last axis (episodes).

    Args:
        rewards: Array of shape (n_evaluations, n_episodes)

    Returns:
        Tuple of (mean, min, max) arrays of shape (n_evaluations,)
    """
    mean_rewards = np.mean(rewards, axis=-1)
    min_rewards = np.min(rewards, axis=-1)
    max_rewards = np.max(rewards, axis=-1)
    return mean_rewards, min_rewards, max_rewards


def format_config_text(config: Dict[str, Any]) -> str:
    """Format configuration dictionary into a readable text string."""
    lines = []
    for key, value in sorted(config.items()):
        # Format key to be more readable
        formatted_key = key.replace("_", " ").title()
        lines.append(f"{formatted_key}: {value}")
    return "\n".join(lines)


def create_plot(
    experiment_name: str,
    experiment_id: str,
    timestamp: str,
    mean_rewards: np.ndarray,
    min_rewards: np.ndarray,
    max_rewards: np.ndarray,
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create and save a plot of rewards with statistics.

    Args:
        experiment_name: Name of the experiment (subfolder name)
        experiment_id: Unique experiment hash/ID
        timestamp: Timestamp of the experiment run
        mean_rewards: Mean rewards across episodes
        min_rewards: Minimum rewards across episodes
        max_rewards: Maximum rewards across episodes
        config: Configuration dictionary
        output_path: Path where to save the plot
    """
    # 4K resolution: 3840x2160 at 200 DPI -> figsize=(19.2, 10.8)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    # Number of evaluation steps
    steps = np.arange(len(mean_rewards))

    # Plot mean rewards
    ax.plot(steps, mean_rewards, label="Mean", linewidth=2, color="blue")

    # Plot min and max as filled area
    ax.fill_between(
        steps, min_rewards, max_rewards, alpha=0.3, color="blue", label="Min-Max Range"
    )

    # Plot min and max lines
    ax.plot(
        steps, min_rewards, "--", label="Min", linewidth=1, color="darkblue", alpha=0.7
    )
    ax.plot(
        steps, max_rewards, "--", label="Max", linewidth=1, color="darkblue", alpha=0.7
    )

    # Labels and title
    ax.set_xlabel("Evaluation Step", fontsize=16)
    ax.set_ylabel("Reward", fontsize=16)
    ax.set_title(
        f"{experiment_name}\nID: {experiment_id}  |  Timestamp: {timestamp}",
        fontsize=18,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=13)

    # Add configuration as text box
    config_text = format_config_text(config)
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    ax.text(
        1.02,
        0.5,
        config_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox=props,
    )

    # Adjust layout to make room for the text box
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot at 4K resolution (3840x2160)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    """Main function to process all experiments and generate plots."""
    # Define paths
    logs_dir = Path(__file__).parent / "logs"
    output_dir = Path(__file__).parent

    if not logs_dir.exists():
        print(f"Error: Logs directory not found at {logs_dir}")
        return

    # Find all eval_rewards.csv files
    reward_files = list(logs_dir.glob("*/*/eval_rewards.csv"))

    if not reward_files:
        print("No eval_rewards.csv files found in the logs directory.")
        return

    print(f"Found {len(reward_files)} experiment(s) to process.")

    # Process each experiment
    for reward_file in reward_files:
        try:
            # Extract experiment name, ID, and timestamp from path
            timestamp = reward_file.parent.name
            full_experiment_name = reward_file.parent.parent.name

            # Parse experiment name and ID (format: algo_env_hash)
            parts = full_experiment_name.rsplit("_", 1)
            if len(parts) == 2:
                experiment_name = parts[0]
                experiment_id = parts[1]
            else:
                experiment_name = full_experiment_name
                experiment_id = "unknown"

            print(
                f"\nProcessing: {experiment_name} (ID: {experiment_id}) / {timestamp}"
            )

            # Load config
            config_file = reward_file.parent / "config.json"
            if not config_file.exists():
                print("  Warning: No config.json found, skipping...")
                continue
            config = load_config(config_file)

            # Load rewards
            rewards = load_rewards(reward_file)
            print(f"  Loaded rewards with shape: {rewards.shape}")

            # Compute statistics
            mean_rewards, min_rewards, max_rewards = compute_statistics(rewards)

            # Create output filename
            output_filename = f"{experiment_name}_{timestamp}_rewards.png"
            output_path = output_dir / output_filename

            # Create and save plot
            create_plot(
                experiment_name=experiment_name,
                experiment_id=experiment_id,
                timestamp=timestamp,
                mean_rewards=mean_rewards,
                min_rewards=min_rewards,
                max_rewards=max_rewards,
                config=config,
                output_path=output_path,
            )

        except Exception as e:
            print(f"  Error processing {reward_file}: {e}")
            continue

    print("\nDone! All plots have been generated.")


if __name__ == "__main__":
    main()
