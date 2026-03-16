from __future__ import annotations

from typing import TYPE_CHECKING

from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy import stats

if TYPE_CHECKING:
    from typing import Callable


@struct.dataclass
class RunningStats:
    mean: jax.Array
    std: jax.Array
    variance: jax.Array
    u_mean: jax.Array
    l_mean: jax.Array
    count: int
    u_count: int
    l_count: int


def approx_distance(x: jax.Array, stats: RunningStats) -> jax.Array:
    """Estimate the distance of ``x`` from the distribution described by ``stats``.

    Computes the absolute distance from ``x`` to each of three reference
    points — the overall running mean, the upper conditional mean (mean of
    values above the running mean), and the lower conditional mean (mean of
    values below the running mean) — then returns their average as a single
    scalar approximation of how far ``x`` sits from the centre of the
    observed distribution.
    """
    distance_mean = jnp.abs(x - stats.mean)
    distance_u = jnp.abs(x - stats.u_mean)
    distance_l = jnp.abs(x - stats.l_mean)
    return (distance_mean + distance_u + distance_l) / 3.0


def approx_distance_weighted(x: jax.Array, stats: RunningStats) -> jax.Array:
    u_weight = stats.u_count / stats.count
    l_weight = stats.l_count / stats.count
    distance_u = jnp.abs(x - stats.u_mean) * u_weight
    distance_l = jnp.abs(x - stats.l_mean) * l_weight
    return distance_u + distance_l


def first_encoding(x: jax.Array) -> jax.Array:
    """Compute running statistics over a 1-D sequence via Welford's algorithm.

    Scans over each element of ``x`` in order, maintaining a ``RunningStats``
    carry that accumulates the running mean, variance, standard deviation, and
    upper/lower conditional means (i.e. the mean of values observed above and
    below the current running mean, respectively).  All intermediate states are
    stacked and returned alongside the final state.

    Args:
        x: A 1-D JAX array of scalar values to encode.

    Returns:
        A tuple ``(final_stats, all_stats)`` where ``final_stats`` is the
        ``RunningStats`` after processing the full sequence and ``all_stats``
        is a ``RunningStats`` whose leaves are arrays of shape ``(len(x),)``
        holding the running statistics recorded after each step.
    """
    stats = RunningStats(
        mean=jnp.zeros(()),
        std=jnp.zeros(()),
        variance=jnp.zeros(()),
        u_mean=jnp.zeros(()),
        l_mean=jnp.zeros(()),
        count=0,
        u_count=0,
        l_count=0,
    )

    def _update_approx(
        carry: RunningStats, new_value: jax.Array
    ) -> tuple[RunningStats, RunningStats]:
        new_count = carry.count + 1
        delta = new_value - carry.mean
        new_mean = carry.mean + delta / new_count
        new_variance = carry.variance + delta * (new_value - new_mean)
        new_std = jnp.sqrt(new_variance / jnp.maximum(new_count - 1, 1))

        above = (new_value > carry.mean).astype(jnp.float32)
        below = 1.0 - above
        new_u_count = carry.u_count + above
        new_l_count = carry.l_count + below
        new_u_mean = carry.u_mean + above * (new_value - carry.u_mean) / jnp.maximum(
            new_u_count, 1
        )
        new_l_mean = carry.l_mean + below * (new_value - carry.l_mean) / jnp.maximum(
            new_l_count, 1
        )

        new_stats = RunningStats(
            mean=new_mean,
            std=new_std,
            variance=new_variance,
            u_mean=new_u_mean,
            l_mean=new_l_mean,
            count=new_count,
            u_count=new_u_count,
            l_count=new_l_count,
        )
        return new_stats, new_stats

    return jax.lax.scan(_update_approx, stats, x)


def estimated_normal_distribution(
    data: jax.Array,
) -> tuple[Callable[[jax.Array], jax.Array], dict[str, float]]:
    mean = jnp.mean(data)
    std = jnp.std(data)

    def density(x: jax.Array) -> jax.Array:
        return stats.norm.pdf(x, loc=mean, scale=std)

    return density, {"mean": mean, "std": std}
