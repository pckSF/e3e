from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp

from scs.data import TrajectoryData
from scs.rl_computations import on_distribution_action_normal_log_density

if TYPE_CHECKING:
    from mujoco_playground import State
    from scs.env_wrapper import EnvTrainingWrapper
    from scs.nn_modules import NNTrainingState
    from scs.ppo.agent_config import PPOConfig
    from scs.ppo.models import PPOModel


def collect_trajectories(
    train_state: NNTrainingState,
    env_state: State,
    env_def: EnvTrainingWrapper,
    config: PPOConfig,
    key: jax.Array,
) -> tuple[TrajectoryData, State]:
    """Collects a fixed-length trajectory batch for every parallel actor.

    Caches an initial state for each environment that can be used during the
    rollout. Randomness across initial states is ensured by a large number of
    parallel rollouts as well as across multiple rollouts.

    The ``EnvTrainingWrapper`` automatically resets the environments of the batch
    that have terminated, using the cached initial state for that environment.

    Args:
        train_state: Current training state backing the policy evaluation.
        env_state: Vectorized environment state prior to the rollout.
        env_def: Vectorized environment wrapper containing step and reset functions.
        config: PPO configuration determining horizon and actor count.
        key: PRNG key consumed for sampling actions during the rollout.

    Returns:
        Trajectory data capturing the ``config.n_actor_steps`` horizon for
        each actor and the updated environment state for the next collection pass.
    """
    model: PPOModel = nnx.merge(train_state.model_def, train_state.model_state)
    keys = jax.random.split(key, num=config.required_rollout_steps + 1)
    cached_initial_state = env_def.get_initial_state(
        jax.random.split(keys[0], num=config.n_actors)
    )

    def _scan_transition(
        env_state: State,
        action_key: jax.Array,
    ) -> tuple[State, TrajectoryData]:
        """Performs one vectorized transition and captures trajectory data."""
        state_obs = env_state.obs
        a_mean, a_log_std = model.get_policy(state_obs)
        a_std = jnp.exp(a_log_std)
        normal_sample = jax.random.normal(action_key, shape=a_mean.shape)
        action = a_mean + a_std * normal_sample
        # Reuse normal_sample and a_log_std to compute log density
        action_log_density = on_distribution_action_normal_log_density(
            normal_sample, a_log_std
        )
        next_env_state = env_def.step(env_state, jnp.tanh(action))
        env_state = env_def.conditional_reset(next_env_state, cached_initial_state)
        # Downstream "done"->"terminal" to better distinguish it from "truncated".
        return env_state, TrajectoryData(
            state_obs,
            action,
            action_log_density,
            next_env_state.reward,
            next_env_state.obs,
            next_env_state.done,
            next_env_state.info["truncated"],
        )

    env_state, trajectory = jax.lax.scan(
        _scan_transition,
        env_state,
        keys[1:],
    )
    return trajectory, env_state
