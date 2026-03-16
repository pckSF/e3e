from __future__ import annotations

from typing import TYPE_CHECKING

from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy import stats
import matplotlib.pyplot as plt

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


def discretize(
    x: jax.Array, bins: int, value_range: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    """Discretizes x into bins over value_range, returning (bin_centers, counts)."""
    bin_edges = jnp.linspace(value_range[0], value_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    indices = jnp.clip(jnp.digitize(x, bin_edges) - 1, 0, bins - 1)
    counts = jnp.bincount(indices, length=bins)
    return bin_centers, counts


@struct.dataclass
class RunningStats:
    mean: jax.Array
    std: jax.Array
    u_mean: jax.Array
    l_mean: jax.Array
    count: int
    u_count: int
    l_count: int


def first_encoding(x: jax.Array) -> jax.Array:
    stats = RunningStats(
        mean=jnp.zeros(()),
        std=jnp.zeros(()),
        u_mean=jnp.zeros(()),
        l_mean=jnp.zeros(()),
        count=0,
        u_count=0,
        l_count=0,
    )

    def _update_approx(
        carry: RunningStats, new_value: jax.Array
    ) -> tuple[RunningStats, RunningStats]:
        cented_value = new_value - carry.mean
        new_mean = carry.mean + cented_value / (carry.count + 1)
        new_std = jnp.sqrt(
            (carry.std + cented_value * (new_value - new_mean)) / (carry.count + 1e-8)
        )

        value_above_mean = new_value > carry.mean
        new_u_mean = carry.u_mean + value_above_mean * (new_value - carry.u_mean) / (
            carry.u_count + 1
        )
        new_u_count = jnp.asarray(carry.u_count + value_above_mean, dtype=jnp.float32)
        new_l_mean = carry.l_mean + (1.0 - value_above_mean) * (
            new_value - carry.l_mean
        ) / (carry.l_count + 1)
        new_l_count = carry.l_count + (1.0 - value_above_mean)

        new_stats = RunningStats(
            mean=new_mean,
            std=new_std,
            u_mean=new_u_mean,
            l_mean=new_l_mean,
            count=carry.count + 1,
            u_count=new_u_count,
            l_count=new_l_count,
        )
        return new_stats, new_stats

    return jax.lax.scan(_update_approx, stats, x)


def plot_distribution(
    functions: dict[str, Callable[[jax.Array], jax.Array]],
    x_range: tuple[float, float],
    data: jax.Array | None = None,
    bins: int = 50,
    num_points: int = 1000,
) -> None:
    x = jnp.linspace(x_range[0], x_range[1], num_points)
    for label, func in functions.items():
        y = func(x)
        plt.plot(x, y, label=label)
    if data is not None:
        bin_width = (x_range[1] - x_range[0]) / bins
        bin_centers, counts = discretize(data, bins=bins, value_range=x_range)
        density = counts / (jnp.sum(counts) * bin_width)
        plt.scatter(bin_centers, density, alpha=0.7, label="Discretized Bins")
    plt.legend()
    plt.title("Estimated Normal Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    data_key, key = jax.random.split(key)
    normal_fn = get_normal_function(mean=0.0, std=1.0)
    true_density_fn = lambda x: stats.norm.pdf(x, loc=0.0, scale=1.0)
    data = normal_fn(data_key, n=10000)

    normal_density_fn, params = estimated_normal_distribution(data)
    print(f"Estimated parameters: {params}")

    plot_distribution(
        functions={
            "True Normal": true_density_fn,
            "Estimated Normal": normal_density_fn,
        },
        x_range=(-10, 10),
        data=data,
    )

    stats, stats_hist = first_encoding(data)
