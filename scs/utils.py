from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np

    from scs.data import TrajectoryData


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


@jax.jit
def stack_trajectories(data: list[TrajectoryData]) -> TrajectoryData:
    """Stacks a list of Data containers along a new leading axis.

    This mirrors the way a ``jax.lax.scan`` would stack returns across iterations.
    """
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *data)


@jax.jit
def concatenate_losses(data: list[TrajectoryData]) -> TrajectoryData:
    """Concatenates a list of Data containers along the leading axis."""
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *data)


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


def discretize(
    x: jax.Array, bins: int, value_range: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    """Discretizes x into bins over value_range, returning (bin_centers, counts)."""
    bin_edges = jnp.linspace(value_range[0], value_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    indices = jnp.clip(jnp.digitize(x, bin_edges) - 1, 0, bins - 1)
    counts = jnp.bincount(indices, length=bins)
    return bin_centers, counts


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


def batch_means(
    data: jax.Array,
) -> tuple[float, float, float, int, int]:
    """Compute exact mean, upper conditional mean, and lower conditional mean.

    Splits ``data`` at its own mean: values strictly above the mean contribute
    to ``u_mean`` and the rest to ``l_mean``.  Also returns the element counts
    for each partition so results can be directly compared against the counts
    tracked by ``RunningStats``.

    Args:
        data: 1-D JAX array of observed values.

    Returns:
        A tuple ``(mean, u_mean, l_mean, u_count, l_count)``.
    """
    m = jnp.mean(data)
    above = data > m
    u_count = int(jnp.sum(above))
    l_count = int(data.shape[0]) - u_count
    u_m = float(jnp.mean(data[above]))
    l_m = float(jnp.mean(data[~above]))
    return float(m), u_m, l_m, u_count, l_count


def neighborhood_counts(data: jax.Array, window_size: float) -> jax.Array:
    """Count how many points fall within a symmetric window around each point.

    Args:
        data: 1-D JAX array of values sorted in ascending order.
        window_size: Half-width of the window, i.e. the neighbourhood of ``x``
            is ``(x - window_size, x + window_size)``.

    Returns:
        1-D integer array of the same length as ``data``, where element ``i``
        is the number of points in ``data`` within ``window_size`` of
        ``data[i]``.
    """

    def _count(x: jax.Array) -> int:
        l_idx, r_idx = jnp.searchsorted(
            data, jnp.array([x - window_size, x + window_size])
        )
        return r_idx - l_idx

    return jax.vmap(_count)(data)
