from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flax import (
    struct,
)
import h5py
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from scs.ppo.agent_config import PPOConfig


@struct.dataclass
class TrajectoryData:
    """JAX-friendly PyTree data container.

    Fields that are scanned over axis 0:

    - ``observations``:         ``[T, N, obs_dim]``
    - ``actions``:              ``[T, N, action_dim]``
    - ``action_log_densities``: ``[T, N, action_dim]``
    - ``rewards``:              ``[T, N]``
    - ``next_observations``:    ``[T, N, obs_dim]``
    - ``terminals``:            ``[T, N]``
    - ``truncations``:          ``[T, N]``
    """

    observations: jax.Array
    actions: jax.Array
    action_log_densities: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    truncations: jax.Array

    @classmethod
    def load_hdf5(cls, path: str) -> TrajectoryData:
        """Loads TrajectoryData from an HDF5 file."""
        with h5py.File(path, "r") as f:
            return cls(
                observations=jnp.asarray(f["observations"]),
                actions=jnp.asarray(f["actions"]),
                action_log_densities=jnp.asarray(f["action_log_densities"]),
                rewards=jnp.asarray(f["rewards"]),
                next_observations=jnp.asarray(f["next_observations"]),
                terminals=jnp.asarray(f["terminals"]),
                truncations=jnp.asarray(f["truncations"]),
            )

    def save_hdf5(self, path: str, compression: str = "gzip") -> None:
        """Saves TrajectoryData to an HDF5 file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for name in (
                "observations",
                "actions",
                "action_log_densities",
                "rewards",
                "next_observations",
                "terminals",
                "truncations",
            ):
                f.create_dataset(
                    name,
                    data=np.asarray(getattr(self, name)),
                    compression=compression,
                )


def get_batch_from_trajectory(
    data: TrajectoryData, batch_indices: jax.Array
) -> TrajectoryData:
    """Returns batch(es) of data based on the provided indices."""
    return jax.tree.map(lambda leaf: leaf[batch_indices], data)


def separate_trajectory_rollouts(
    data: TrajectoryData,
    config: PPOConfig,
) -> TrajectoryData:
    """Reshapes trajectory data into ``[n_rollouts, n_steps, ...]`` format.

    For training we want ``n_batches * batch_size`` samples of length
    ``n_rollout_steps``. To avoid calling the rollout function multiple times,
    we instead run a longer rollout of length ``required_rollout_steps`` and then
    reshape the data accordingly.

    Since the rollout returns data of shape ``[required_rollout_steps, n_actors, ...]``,
    the first step is to swap the first two axes such that the reshape operation can
    split the collected timesteps across actors and not the actors across timesteps.
    """

    def _reshape_leaf(leaf: jax.Array) -> jax.Array:
        feature_shape = leaf.shape[2:]
        leaf = jnp.swapaxes(leaf, 0, 1)
        return leaf.reshape((-1, config.n_rollout_steps) + feature_shape)

    return jax.tree.map(_reshape_leaf, data)
