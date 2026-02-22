from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def get_train_batch_indices(
    n_batches: int,
    batch_size: int,
    max_index: int,
    resample: bool,
    key: jax.Array,
) -> jax.Array:
    """Generates random indices for training batches.

    Without resampling ``n_batches * batch_size`` must be less or equal than the
    ``max_index`` since no sample is allowed to repeat across the set of batches.

    With resampling, each batch is sampled independently, allowing for samples to
    reappear in multiple batches.

    Note:
        Outdated; use ``jax.random.permutation`` instead!
    """
    print(
        "Outdated function 'get_train_batch_indices' called in scs/utils.py! "
        "Use jax.random.permutation in combination with reshape instead."
    )
    indices = jnp.arange(max_index)
    if not resample:
        return jax.random.choice(key, indices, (n_batches, batch_size), replace=False)
    else:
        keys = jax.random.split(key, n_batches)
        return jax.lax.map(
            partial(jax.random.choice, a=indices, shape=(batch_size,), replace=False),
            keys,
        )


def compare_observations(
    observation1: jax.Array | np.ndarray,
    observation2: jax.Array | np.ndarray,
    mask: jax.Array | np.ndarray | None = None,
) -> bool:
    """Compares two observation arrays for equality."""
    if mask is not None:
        observation1, observation2 = observation1[mask], observation2[mask]
    return bool(jnp.array_equal(observation1, observation2))


def observations_healthcheck(
    observation1: jax.Array | np.ndarray,
    observation2: jax.Array | np.ndarray,
    mask: jax.Array | np.ndarray,
) -> None:
    if not compare_observations(observation1, observation2, mask):
        raise ValueError(
            f"Relevant observation arrays do not match!\n"
            f"Observation 1: {observation1[mask]}\n"
            f"Observation 2: {observation2[mask]}"
        )
