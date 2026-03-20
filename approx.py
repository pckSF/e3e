from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt

from scs.data import TrajectoryData
from scs.utils import neighborhood_counts

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/cllander_trajectories/episodes_1000_maxlen_250.hdf5"
################################################################################


if __name__ == "__main__":
    data = TrajectoryData.load_hdf5(data_path)
    print(f"Loaded trajectory data from: {data_path}")
    sorted_actions = jnp.sort(jnp.tanh(data.actions), axis=0)

    action_1 = sorted_actions[:, 0]
    action_2 = sorted_actions[:, 1]

    nnc_actions_1 = neighborhood_counts(action_1, window_size=0.1) / action_1.shape[0]
    nnc_actions_2 = neighborhood_counts(action_2, window_size=0.1) / action_2.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(action_1, nnc_actions_1, color="steelblue")
    axes[0].set_title("Main Thruster")
    axes[0].set_xlabel("action value")
    axes[0].set_ylabel("neighborhood count (normalized)")
    axes[1].plot(action_2, nnc_actions_2, color="steelblue")
    axes[1].set_title("Lateral Thruster")
    axes[1].set_xlabel("action value")
    axes[1].set_ylabel("neighborhood count (normalized)")
    fig.tight_layout()
    plt.show()
