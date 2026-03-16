from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy import stats
import matplotlib.pyplot as plt

from scs.encodings import approx_distance, first_encoding

if TYPE_CHECKING:
    from typing import Callable


def get_normal_function(
    mean: float, std: float
) -> Callable[[jax.Array, int], jax.Array]:
    def random_normal(key: jax.Array, n=1000) -> jax.Array:
        return mean + std * jax.random.normal(key, shape=(n,))

    return random_normal


def estimated_normal_distribution(
    data: jax.Array,
) -> tuple[Callable[[jax.Array], jax.Array], dict[str, float]]:
    mean = jnp.mean(data)
    std = jnp.std(data)

    def density(x: jax.Array) -> jax.Array:
        return stats.norm.pdf(x, loc=mean, scale=std)

    return density, {"mean": mean, "std": std}


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


def discretize(
    x: jax.Array, bins: int, value_range: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    """Discretizes x into bins over value_range, returning (bin_centers, counts)."""
    bin_edges = jnp.linspace(value_range[0], value_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    indices = jnp.clip(jnp.digitize(x, bin_edges) - 1, 0, bins - 1)
    counts = jnp.bincount(indices, length=bins)
    return bin_centers, counts


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
    data_key, key = jax.random.split(key)
    normal_fn = get_normal_function(mean=0.0, std=1.0)
    true_density_fn = lambda x: stats.norm.pdf(x, loc=0.0, scale=1.0)
    data = normal_fn(data_key, n=10000)

    normal_density_fn, params = estimated_normal_distribution(data)
    print(f"Estimated parameters: {params}")

    rolling_stats, stats_hist = first_encoding(data)
    print(
        f"Rolling stats — mean: {rolling_stats.mean:.4f}, "
        f"l_mean: {rolling_stats.l_mean:.4f}, u_mean: {rolling_stats.u_mean:.4f}"
    )

    ax = plot_distribution(
        functions={
            "True Normal": true_density_fn,
            "Estimated Normal": normal_density_fn,
        },
        x_range=(-10, 10),
        data=data,
    )

    ax.axvline(
        x=float(rolling_stats.mean), linestyle="--", linewidth=1.5, label="Rolling Mean"
    )

    trans = ax.get_xaxis_transform()
    for xval, char, label in [
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
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=7,
            clip_on=False,
            color="gray",
        )

    ax.legend()
    plt.show()

    points = jnp.linspace(-10, 10, 21)
    mean_dist = mean_distance(points, data)
    approx_dist = approx_distance(points, rolling_stats)
    print(f"Mean distance: {mean_dist}\nApprox distance: {approx_dist}")
    print(f"Mean distance vs Approx distance: {mean_dist / approx_dist}")
