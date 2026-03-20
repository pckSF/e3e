from __future__ import annotations

import jax.numpy as jnp

from scs.data import TrajectoryData

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/cllander_trajectories/episodes_1000_maxlen_250.hdf5"
################################################################################


if __name__ == "__main__":
    data = TrajectoryData.load_hdf5(data_path)
    print(f"Loaded trajectory data from: {data_path}")
    observations = data.observations
    n_dims = observations.shape[1]
    sorted_obs = jnp.sort(observations, axis=0)

    # y_pos = sorted_obs[:, 1]
    # nn_counts = neighborhood_counts(y_pos, window_size=0.1) / y_pos.shape[0]

    # plt.plot(y_pos, nn_counts)
    # plt.xlabel("y position")
    # plt.ylabel("normalized neighborhood count")
    # plt.title("Normalized Neighborhood Counts for y position")
    # plt.show()
