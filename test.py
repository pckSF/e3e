from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy import stats
import matplotlib.pyplot as plt

from scs.encodings import approx_distance, estimated_normal_distribution, first_encoding
from scs.gen_data import get_bimodal_function, get_normal_function
from scs.utils import discretize

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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_dist = len(DISTRIBUTIONS)
    fig, axes = plt.subplots(1, n_dist, figsize=(6 * n_dist, 5))
    if n_dist == 1:
        axes = [axes]

    for ax, (name, config) in zip(axes, DISTRIBUTIONS.items()):
        data_key, key = jax.random.split(key)
        data = config["sampler"](data_key, n=N_SAMPLES)

        normal_density_fn, params = estimated_normal_distribution(data)
        print(f"[{name}] Estimated parameters: {params}")

        rolling_stats, _ = first_encoding(data)
        print(
            f"[{name}] Rolling stats — mean: {rolling_stats.mean:.4f}, "
            f"l_mean: {rolling_stats.l_mean:.4f}, u_mean: {rolling_stats.u_mean:.4f}, "
            f"count l_mean: {rolling_stats.l_count}, "
            f"count u_mean: {rolling_stats.u_count}"
        )

        density_fns: dict[str, Callable[[jax.Array], jax.Array]] = {
            "Estimated Normal": normal_density_fn
        }
        if config["true_density"] is not None:
            density_fns["True Density"] = config["true_density"]

        plot_distribution(density_fns, x_range=X_RANGE, data=data, bins=BINS, ax=ax)
        ax.set_title(name)

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

        ax.legend()

        points = jnp.linspace(X_RANGE[0], X_RANGE[1], 21)
        mean_dist = mean_distance(points, data)
        approx_dist = approx_distance(points, rolling_stats)
        mae = jnp.mean(jnp.abs(mean_dist - approx_dist))
        print(f"[{name}] Mean distance vs Approx distance: {mean_dist / approx_dist}")
        print(f"[{name}] MAE(mean_dist, approx_dist): {mae:.4f}")

    plt.tight_layout()
    plt.show()
