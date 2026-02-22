from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp

from scs.data import TrajectoryData


@struct.dataclass
class BufferState:
    """Container for the replay buffer's state.

    Fields that are JAX PyTrees:
    - ``observations``:       ``[max_size, obs_dim]``
    - ``actions``:            ``[max_size, action_dim]``
    - ``rewards``:            ``[max_size]``
    - ``next_observations``:  ``[max_size, obs_dim]``
    - ``terminals``:          ``[max_size]``

    - ``ptr``:                 Current pointer index in the buffer.
    - ``max_size``:            Maximum size of the buffer.

    """

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminals: jax.Array
    ptr: int
    max_size: int = struct.field(pytree_node=False)


def initialize_simple_buffer(
    max_size: int,
    obs_dim: int,
    action_dim: int,
) -> BufferState:
    """Creates a new replay buffer with zero-initialized arrays."""
    return BufferState(
        observations=jnp.zeros((max_size, obs_dim)),
        actions=jnp.zeros((max_size, action_dim)),
        rewards=jnp.zeros((max_size,)),
        next_observations=jnp.zeros((max_size, obs_dim)),
        terminals=jnp.zeros((max_size,)),
        ptr=0,
        max_size=max_size,
    )


def add_batch(
    buffer_state: BufferState,
    observations: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_observations: jax.Array,
    terminals: jax.Array,
) -> BufferState:
    """Adds a batch of transitions to the replay buffer.

    Info:
        Ensure max_size is divisibale by batch_size in config setup
        to enable simple and clean insertion.
    """
    batch_size = observations.shape[0]
    insert_at = buffer_state.ptr
    observations = jax.lax.dynamic_update_slice_in_dim(
        buffer_state.observations, observations, insert_at, axis=0
    )
    actions = jax.lax.dynamic_update_slice_in_dim(
        buffer_state.actions, actions, insert_at, axis=0
    )
    rewards = jax.lax.dynamic_update_slice_in_dim(
        buffer_state.rewards, rewards, insert_at, axis=0
    )
    next_observations = jax.lax.dynamic_update_slice_in_dim(
        buffer_state.next_observations, next_observations, insert_at, axis=0
    )
    terminals = jax.lax.dynamic_update_slice_in_dim(
        buffer_state.terminals, terminals, insert_at, axis=0
    )
    new_ptr = (insert_at + batch_size) % buffer_state.max_size
    return buffer_state.replace(  # type: ignore[attr-defined]
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        ptr=new_ptr,
    )


def uniform_sample_batches(
    buffer_state: BufferState,
    batch_size: int,
    n_batches: int,
    key: jax.Array,
) -> TrajectoryData:
    """Samples a batch of transitions uniformly from the replay buffer."""
    batch_indices = jax.random.randint(
        key, (n_batches, batch_size), 0, buffer_state.max_size
    )
    return TrajectoryData(
        observations=buffer_state.observations[batch_indices],
        actions=buffer_state.actions[batch_indices],
        action_log_densities=jnp.zeros((n_batches, batch_size)),  # Placeholder
        rewards=buffer_state.rewards[batch_indices],
        next_observations=buffer_state.next_observations[batch_indices],
        terminals=buffer_state.terminals[batch_indices],
    )
