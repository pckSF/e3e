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
        cented_value = new_value - carry.mean
        new_mean = carry.mean + cented_value / (carry.count + 1)
        new_variance = carry.variance + cented_value * (new_value - new_mean)
        new_std = jnp.sqrt(new_variance / (carry.count + 1e-8))

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
            variance=new_variance,
            u_mean=new_u_mean,
            l_mean=new_l_mean,
            count=carry.count + 1,
            u_count=new_u_count,
            l_count=new_l_count,
        )
        return new_stats, new_stats

    return jax.lax.scan(_update_approx, stats, x)
