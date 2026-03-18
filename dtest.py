from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.stats
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
        col = sorted_obs[:, i]

        counts = neighborhood_counts(col, window_size)
        ax1 = ax
        ax1.plot(col, counts, color="steelblue", label="neighborhood count")
        ax1.set_ylabel("neighborhood count", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")

        ax2 = ax1.twinx()
        kde = jax.scipy.stats.gaussian_kde(col)
        xs = jnp.linspace(col.min(), col.max(), 512)
        ax2.plot(xs, kde(xs), color="darkorange", label="KDE")
        ax2.set_ylabel("density", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

        ax1.set_title(feature_names[i])
        ax1.set_xlabel("value")

    fig.tight_layout()
    plt.show()
