from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp


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


def first_encoding(x: jax.Array) -> jax.Array:
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
