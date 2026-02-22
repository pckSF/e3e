from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
)

from flax import nnx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from mujoco_playground import State
    from scs.env_wrapper import EnvTrainingWrapper
    from scs.nn_modules import (
        NNTrainingState,
        NNTrainingStateSoftTarget,
    )
    from scs.ppo.agent_config import PPOConfig
    from scs.ppo.models import PPOModel
    from scs.sac.agent_config import SACConfig
    from scs.sac.models import SACPolicy


@partial(jax.jit, static_argnums=(1, 2))
def evaluation_trajectory(
    train_state: NNTrainingState | NNTrainingStateSoftTarget,
    env_def: EnvTrainingWrapper,
    config: PPOConfig | SACConfig,
    key: jax.Array,
) -> jax.Array:
    """Runs the agent for a full evaluation trajectory in parallel environments.

    Args:
        train_state: Training state containing the policy parameters to test.
        env_def: Environment wrapped in vectorization wrapper.
        config: PPO or SAC configuration specifying the number of actors.
        key: PRNG key used for reset and action sampling.

    Returns:
        Reward totals per parallel environment with shape
        ``(config.n_actors,)``.
    """
    model: PPOModel | SACPolicy = nnx.merge(
        train_state.model_def, train_state.model_state
    )

    def _scan_eval_transition(
        carry: tuple[State, jax.Array],
        key: jax.Array,
    ) -> tuple[tuple[State, jax.Array], jax.Array]:
        """Performs a transition during evaluation; rewards are masked after done."""
        env_state, terminated = carry
        a_mean, a_log_std = model.get_policy(env_state.obs)
        action = a_mean + jnp.exp(a_log_std) * jax.random.normal(
            key, shape=a_mean.shape
        )
        next_env_state = env_def.step(env_state, jnp.tanh(action))
        step_reward = next_env_state.reward * jnp.logical_not(terminated)
        terminated = jnp.logical_or(terminated, next_env_state.done)
        return (next_env_state, terminated), step_reward

    keys = jax.random.split(key, num=config.n_actors + 1)
    key, reset_keys = keys[0], keys[1:]
    env_state = env_def.reset(reset_keys)
    keys = jax.random.split(key, num=env_def.config.episode_length)
    (_env_state, _terminated), rewards = jax.lax.scan(
        _scan_eval_transition,
        (env_state, jnp.zeros((config.n_actors,), dtype=bool)),
        keys,
    )
    return jnp.sum(rewards, axis=0)
