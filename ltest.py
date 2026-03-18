from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from scs.data import TrajectoryData
from scs.encodings import (
    approx_distance,
    approx_distance_weighted,
    estimated_normal_distribution,
    first_encoding,
)
from scs.utils import discretize, mean_distance

################################################################################
# Hyperparameters
################################################################################
data_path: str = "data/llander_trajectories/episodes_1000_maxlen_250.hdf5"
BINS: int = 50
EUCLIDEAN_SUBSAMPLE: int = 2000  # pairwise cost is O(N²·D); subsample to keep it fast
################################################################################

OBS_LABELS = [
    "x position",
    "y position",
    "x velocity",
    "y velocity",
    "angle",
    "angular velocity",
    "left leg contact",
    "right leg contact",
]

if __name__ == "__main__":
    data = TrajectoryData.load_hdf5(data_path)
    print(f"Loaded trajectory data from: {data_path}")
    observations = data.observations  # [T, 8]
    n_dims = observations.shape[1]

    # Euclidean baseline — computed once, overlaid on all bottom-row plots.
    # Subsample to keep the O(N²) pairwise computation tractable.
    key = jax.random.PRNGKey(0)
    idx = jax.random.choice(
        key, observations.shape[0], (EUCLIDEAN_SUBSAMPLE,), replace=False
    )
    obs_sub = observations[idx]  # [S, 8]
    # Mean Euclidean distance from every sample point to all other sample points.
    diffs = obs_sub[:, None, :] - obs_sub[None, :, :]  # [S, S, 8]
    pairwise = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  # [S, S]
    euclidean_dist_per_point = jnp.mean(pairwise, axis=1)  # [S]
    print(f"Euclidean baseline computed on {EUCLIDEAN_SUBSAMPLE} subsampled points.")

    fig, axes = plt.subplots(2, n_dims, figsize=(4 * n_dims, 8))
    fig.suptitle("Observation Distributions — First Encoding (Welford's)", fontsize=14)

    for d in range(n_dims):
        obs_d = observations[:, d]
        x_min, x_max = float(jnp.min(obs_d)), float(jnp.max(obs_d))
        x_range = (x_min, x_max)
        x_points = jnp.linspace(x_min, x_max, 1000)

        rolling_stats, _ = first_encoding(obs_d)
        density_fn, params = estimated_normal_distribution(obs_d)

        # Row 0: estimated density + data histogram + rolling mean markers
        ax0 = axes[0, d]
        bin_width = (x_max - x_min) / BINS
        bin_centers, counts = discretize(obs_d, bins=BINS, value_range=x_range)
        density = counts / (jnp.sum(counts) * bin_width)
        ax0.scatter(bin_centers, density, alpha=0.5, s=10, label="data")
        ax0.plot(
            x_points,
            density_fn(x_points),
            label=f"N({float(params['mean']):.2f}, {float(params['std']):.2f})",
        )
        ax0.axvline(
            float(rolling_stats.mean), linestyle="--", linewidth=1.5, label="mean"
        )
        ax0.axvline(
            float(rolling_stats.l_mean),
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label="l_mean",
        )
        ax0.axvline(
            float(rolling_stats.u_mean),
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label="u_mean",
        )
        ax0.set_title(OBS_LABELS[d])
        ax0.set_xlabel("value")
        ax0.set_ylabel("density")
        ax0.legend(fontsize=6)

        # Row 1: approx_distance and approx_distance_weighted vs true distance
        ax1 = axes[1, d]
        ad = approx_distance(x_points, rolling_stats)
        adw = approx_distance_weighted(x_points, rolling_stats)
        true_dist = mean_distance(x_points, obs_d)
        ax1.plot(
            x_points, true_dist, label="true distance (1D)", linewidth=2, color="black"
        )
        ax1.plot(x_points, ad, label="approx_distance", linestyle="--")
        ax1.plot(x_points, adw, label="approx_distance_weighted", linestyle=":")
        # Euclidean baseline: scatter each subsampled point's dim-d value vs its
        # mean full-space Euclidean distance to all other subsampled points.
        ax1.scatter(
            obs_sub[:, d],
            euclidean_dist_per_point,
            alpha=0.15,
            s=4,
            color="gray",
            label="euclidean (full-space)",
        )
        ax1.set_title(OBS_LABELS[d])
        ax1.set_xlabel("value")
        ax1.set_ylabel("distance")
        ax1.legend(fontsize=6)

    plt.tight_layout()
    plt.show()
