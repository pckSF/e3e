from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt

from scs.data import TrajectoryData
from scs.utils import neighborhood_counts

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/llander_trajectories/episodes_1000_maxlen_250.hdf5"
################################################################################


if __name__ == "__main__":
    data = TrajectoryData.load_hdf5(data_path)
    print(f"Loaded trajectory data from: {data_path}")
    observations = data.observations
    n_dims = observations.shape[1]
    sorted_obs = jnp.sort(observations, axis=0)

    feature_names = [
        "x position",
        "y position",
        "x velocity",
        "y velocity",
        "angle",
        "angular velocity",
        "left leg contact",
        "right leg contact",
    ]
    window_size = 0.1

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        counts = neighborhood_counts(sorted_obs[:, i], window_size)
        ax.plot(sorted_obs[:, i], counts)
        ax.set_title(feature_names[i])
        ax.set_xlabel("value")
        ax.set_ylabel("neighborhood count")

    fig.tight_layout()
    plt.show()
