from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from scs.data import TrajectoryData
from scs.rl_computations import on_distribution_action_normal_log_density
from scs.utils import stack_trajectories

if TYPE_CHECKING:
    from scs.env_wrapper import JNPWrapper
    from scs.nn_modules import NNTrainingState
    from scs.ppo.agent_config import PPOConfig
    from scs.ppo.models import PPOModel


def collect_trajectories(
    train_state: NNTrainingState,
    observation: jax.Array,
    env: JNPWrapper,
    config: PPOConfig,
    key: jax.Array,
) -> tuple[TrajectoryData, jax.Array]:
    """Collects a fixed-length trajectory batch for every parallel actor.

    Args:
        train_state: Current training state backing the policy evaluation.
        env_state: Vectorized environment state prior to the rollout.
        env: Vectorized environment wrapper containing step and reset functions.
        config: PPO configuration determining horizon and actor count.
        key: PRNG key consumed for sampling actions during the rollout.

    Returns:
        Trajectory data capturing the ``config.n_actor_steps`` horizon for
        each actor and the updated environment state for the next collection pass.
    """
    model: PPOModel = nnx.merge(train_state.model_def, train_state.model_state)
    keys = jax.random.split(key, num=config.required_rollout_steps)

    def _transition(
        observation: jax.Array,
        action_key: jax.Array,
    ) -> tuple[jax.Array, TrajectoryData]:
        """Performs one vectorized transition and captures trajectory data."""
        a_mean, a_log_std = model.get_policy(observation)
        a_std = jnp.exp(a_log_std)
        normal_sample = jax.random.normal(action_key, shape=a_mean.shape)
        action = a_mean + a_std * normal_sample
        action_log_density = on_distribution_action_normal_log_density(
            normal_sample, a_log_std
        )
        next_observation, reward, terminated, truncated = env.step(jnp.tanh(action))
        timestep = TrajectoryData(
            observation,
            action,
            action_log_density,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        reset_mask = jnp.logical_or(terminated, truncated)
        if jnp.any(reset_mask):
            next_observation = env.reset(options={"reset_mask": np.asarray(reset_mask)})
        return next_observation, timestep

    trajectory = []
    for k in keys:
        observation, data = _transition(observation, k)
        trajectory.append(data)

    return (stack_trajectories(trajectory), observation)
