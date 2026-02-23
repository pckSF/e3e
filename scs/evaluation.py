from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from scs.appo.agent_config import APPOConfig
    from scs.appo.models import APPOModel
    from scs.env_wrapper import JNPWrapper
    from scs.ppo.agent_config import PPOConfig
    from scs.ppo.models import PPOModel


def evaluation_trajectory(
    model: PPOModel | APPOModel,
    env: JNPWrapper,
    config: PPOConfig | APPOConfig,
    key: jax.Array,
    episode_length: int = 1000,
) -> jax.Array:
    """Runs the agent for a full evaluation trajectory in parallel environments.

    Args:
        model: PPO model containing the policy parameters to test.
        env: Environment wrapped in vectorization wrapper.
        config: PPO configuration specifying the number of actors.
        key: PRNG key used for reset and action sampling.
        episode_length: Number of steps to run in the evaluation trajectory.

    Returns:
        Reward totals per parallel environment with shape
        ``(config.n_actors,)``.
    """

    def _eval_transition(
        observation: jax.Array,
        terminated: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Performs a transition during evaluation; rewards are masked after done."""
        policy_logits = model.get_policy_logits(observation)
        action = jax.random.categorical(key, policy_logits)
        next_observation, reward, termination, truncation = env.step(action)
        step_reward = reward * jnp.logical_not(terminated)
        done = jnp.logical_or(termination, truncation)
        terminated = jnp.logical_or(terminated, done)
        if jnp.any(done):
            next_observation = env.reset(options={"reset_mask": np.asarray(done)})
        return next_observation, terminated, step_reward

    observation = env.reset()
    terminated = jnp.zeros((config.n_actors,), dtype=jnp.bool_)
    keys = jax.random.split(key, num=episode_length)
    eval_rewards = []
    for k in keys:
        observation, terminated, rewards = _eval_transition(
            observation, terminated, key=k
        )
        eval_rewards.append(rewards)
    return jnp.sum(jnp.stack(eval_rewards), axis=0)
