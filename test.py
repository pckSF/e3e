from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy import stats
import matplotlib.pyplot as plt

from scs.encodings import (
    approx_distance,
    approx_distance_weighted,
    estimated_normal_distribution,
    first_encoding,
)
from scs.gen_data import (
    get_bimodal_function,
    get_multimodal_function,
    get_normal_function,
)
from scs.utils import batch_means, discretize

if TYPE_CHECKING:
    from typing import Callable

################################################################################
# Hyperparameters
################################################################################
N_SAMPLES: int = 10_000
X_RANGE: tuple[int, int] = (-10, 10)
BINS: int = 50

DISTRIBUTIONS: dict[str, dict[str, Callable[[jax.Array], jax.Array]]] = {
    "normal": {
        "sampler": get_normal_function(mean=0.0, std=1.0),
        "true_density": lambda x: stats.norm.pdf(x, loc=0.0, scale=1.0),
    },
    "bimodal w=0.5": {
        "sampler": get_bimodal_function(
            mean1=-3.0, std1=1.0, mean2=3.0, std2=1.0, weight=0.5
        ),
        "true_density": lambda x: (
            0.5 * stats.norm.pdf(x, loc=-3.0, scale=1.0)
            + 0.5 * stats.norm.pdf(x, loc=3.0, scale=1.0)
        ),
    },
    "bimodal w=0.2": {
        "sampler": get_bimodal_function(
            mean1=-3.0, std1=1.0, mean2=3.0, std2=1.0, weight=0.2
        ),
        "true_density": lambda x: (
            0.2 * stats.norm.pdf(x, loc=-3.0, scale=1.0)
            + 0.8 * stats.norm.pdf(x, loc=3.0, scale=1.0)
        ),
    },
    "bimodal w=0.8": {
        "sampler": get_bimodal_function(
            mean1=-3.0, std1=1.0, mean2=3.0, std2=1.0, weight=0.8
        ),
        "true_density": lambda x: (
            0.8 * stats.norm.pdf(x, loc=-3.0, scale=1.0)
            + 0.2 * stats.norm.pdf(x, loc=3.0, scale=1.0)
        ),
    },
    "trimodal 1/3": {
        "sampler": get_multimodal_function(
            means=[-4.0, 0.0, 4.0],
            stds=[1.0, 1.0, 1.0],
            weights=[1 / 3, 1 / 3, 1 / 3],
        ),
        "true_density": lambda x: (
            (1 / 3) * stats.norm.pdf(x, loc=-4.0, scale=1.0)
            + (1 / 3) * stats.norm.pdf(x, loc=0.0, scale=1.0)
            + (1 / 3) * stats.norm.pdf(x, loc=4.0, scale=1.0)
        ),
    },
    "trimodal skewed": {
        "sampler": get_multimodal_function(
            means=[-4.0, 0.0, 4.0],
            stds=[1.0, 1.5, 0.8],
            weights=[0.5, 0.3, 0.2],
        ),
        "true_density": lambda x: (
            0.5 * stats.norm.pdf(x, loc=-4.0, scale=1.0)
            + 0.3 * stats.norm.pdf(x, loc=0.0, scale=1.5)
            + 0.2 * stats.norm.pdf(x, loc=4.0, scale=0.8)
        ),
    },
}
################################################################################


def mean_distance(x: jax.Array, data: jax.Array) -> jax.Array:
    """Compute the mean absolute distance from each query point to a dataset.

    For every element in ``x``, calculates the average absolute difference
    against all elements in ``data``, producing one distance value per query
    point.
    """
    if not isinstance(x, jax.Array):
        x = jnp.array([x])
    distances = jnp.abs(x[:, jnp.newaxis] - data[jnp.newaxis, :])
    return jnp.mean(distances, axis=-1)


def plot_distribution(
    functions: dict[str, Callable[[jax.Array], jax.Array]],
    x_range: tuple[float, float],
    data: jax.Array | None = None,
    bins: int = 50,
    num_points: int = 1000,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    x = jnp.linspace(x_range[0], x_range[1], num_points)
    for label, func in functions.items():
        y = func(x)
        ax.plot(x, y, label=label)
    if data is not None:
        bin_width = (x_range[1] - x_range[0]) / bins
        bin_centers, counts = discretize(data, bins=bins, value_range=x_range)
        density = counts / (jnp.sum(counts) * bin_width)
        ax.scatter(bin_centers, density, alpha=0.7, label="Discretized Bins")
    ax.legend()
    ax.set_title("Estimated Normal Distributions")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if show:
        plt.show()
    return ax


def _add_rolling_markers(ax: plt.Axes, rolling_stats) -> None:
    ax.axvline(
        x=float(rolling_stats.mean),
        linestyle="--",
        linewidth=1.5,
        label="Rolling Mean",
    )
    trans = ax.get_xaxis_transform()
    for xval, char, lbl in [
        (float(rolling_stats.l_mean), "(", "l_mean"),
        (float(rolling_stats.u_mean), ")", "u_mean"),
    ]:
        ax.axvline(x=xval, linestyle=":", linewidth=1, alpha=0.6)
        ax.text(
            xval,
            -0.06,
            char,
            transform=trans,
            ha="center",
            va="top",
            fontsize=14,
            clip_on=False,
        )
        ax.text(
            xval,
            -0.12,
            lbl,
            transform=trans,
            ha="center",
            va="top",
            fontsize=7,
            clip_on=False,
            color="gray",
        )


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_dist = len(DISTRIBUTIONS)
    points = jnp.linspace(X_RANGE[0], X_RANGE[1], 21)

    fig, axes = plt.subplots(1, n_dist, figsize=(6 * n_dist, 5))
    if n_dist == 1:
        axes = [axes]

    for col, (name, config) in enumerate(DISTRIBUTIONS.items()):
        data_key, key = jax.random.split(key)
        data = config["sampler"](data_key, n=N_SAMPLES)

        normal_density_fn, params = estimated_normal_distribution(data)
        print(f"[{name}] Estimated parameters: {params}")

        rolling_stats, _ = first_encoding(data)
        m, u_m, l_m, u_count, l_count = batch_means(data)
        print(
            f"[{name}] Rolling stats — mean: {rolling_stats.mean:.4f}, "
            f"l_mean: {rolling_stats.l_mean:.4f} (n={int(rolling_stats.l_count)}), "
            f"u_mean: {rolling_stats.u_mean:.4f} (n={int(rolling_stats.u_count)})"
        )
        print(
            f"[{name}] Batch stats   — mean: {m:.4f}, "
            f"l_mean: {l_m:.4f} (n={l_count}), "
            f"u_mean: {u_m:.4f} (n={u_count})"
        )
        print(
            f"[{name}] Δ mean: {(float(rolling_stats.mean) - m):.4f}, "
            f"Δ l_mean: {(float(rolling_stats.l_mean) - l_m):.4f}, "
            f"Δ u_mean: {(float(rolling_stats.u_mean) - u_m):.4f}, "
            f"Δ l_count: {(int(rolling_stats.l_count) - l_count)}, "
            f"Δ u_count: {(int(rolling_stats.u_count) - u_count)}"
        )

        density_fns: dict[str, Callable[[jax.Array], jax.Array]] = {
            "Estimated Normal": normal_density_fn
        }
        if config["true_density"] is not None:
            density_fns["True Density"] = config["true_density"]

        ax = axes[col]
        plot_distribution(density_fns, x_range=X_RANGE, data=data, bins=BINS, ax=ax)
        ax.set_title(name)
        _add_rolling_markers(ax, rolling_stats)
        ax.legend(fontsize=7)

        # Distance metrics
        mean_dist = mean_distance(points, data)
        ad = approx_distance(points, rolling_stats)
        adw = approx_distance_weighted(points, rolling_stats)
        mae_ad = float(jnp.mean(jnp.abs(mean_dist - ad)))
        mae_adw = float(jnp.mean(jnp.abs(mean_dist - adw)))
        print(f"[{name}] MAE approx_distance:          {mae_ad:.4f}")
        print(f"[{name}] MAE approx_distance_weighted: {mae_adw:.4f}")

    plt.tight_layout()
    plt.show()
