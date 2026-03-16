from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from typing import Callable


def get_normal_function(
    mean: float, std: float
) -> Callable[[jax.Array, int], jax.Array]:
    def random_normal(key: jax.Array, n=1000) -> jax.Array:
        return mean + std * jax.random.normal(key, shape=(n,))

    return random_normal


def get_bimodal_function(
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    weight: float = 0.5,
) -> Callable[[jax.Array, int], jax.Array]:
    """Return a sampler for a two-component Gaussian mixture distribution.

    Each call draws ``n`` samples by independently assigning each sample to
    one of two normal components: component 1 is chosen with probability
    ``weight`` and component 2 with probability ``1 - weight``.

    Args:
        mean1: Mean of the first Gaussian component.
        std1: Standard deviation of the first Gaussian component.
        mean2: Mean of the second Gaussian component.
        std2: Standard deviation of the second Gaussian component.
        weight: Mixing weight for the first component in ``[0, 1]``.
            Defaults to ``0.5`` (equal mixture).

    Returns:
        A callable ``(key, n=1000) -> jax.Array`` that draws ``n`` samples
        from the mixture using the supplied JAX PRNG key.
    """

    def random_bimodal(key: jax.Array, n: int = 1000) -> jax.Array:
        k1, k2, k3 = jax.random.split(key, 3)
        mask = jax.random.uniform(k1, shape=(n,)) < weight
        s1 = mean1 + std1 * jax.random.normal(k2, shape=(n,))
        s2 = mean2 + std2 * jax.random.normal(k3, shape=(n,))
        return jnp.where(mask, s1, s2)

    return random_bimodal
