from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from flax import (
    struct,
)
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from scs.ppo.agent_config import PPOConfig


@struct.dataclass
class TrajectoryData:
    """JAX-friendly PyTree data container.

    Fields that are scanned over axis 0:

    - ``observations``:         ``[T, N, obs_dim]``
    - ``actions``:              ``[T, N]`` (discrete action indices)
    - ``policy_logits``:        ``[T, N, n_actions]``
    - ``rewards``:              ``[T, N]``
    - ``next_observations``:    ``[T, N, obs_dim]``
    - ``terminals``:            ``[T, N]``
    - ``truncated``:            ``[T, N]``
    """

    observations: jax.Array
    actions: jax.Array
    policy_logits: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    truncated: jax.Array

    @classmethod
    def load(cls, path: str) -> TrajectoryData:
        """Loads TrajectoryData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves TrajectoryData to a pickle file."""
        data_dict = {
            "observations": self.observations,
            "actions": self.actions,
            "policy_logits": self.policy_logits,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
            "truncated": self.truncated,
        }
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)


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
