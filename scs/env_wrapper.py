"""Wrapper for vectorized Gymnasium environments.

This module provides a thin wrapper around Gymnasium's vectorized environments
that converts observations and rewards to JAX arrays. The wrapper uses `Any` typing
for the underlying environment because Gymnasium's type hierarchy is inconsistent:
`gym.make_vec()` returns `VectorEnv` but useful attributes like `envs`, `get_attr`,
and `call` are only defined on concrete subclasses like `SyncVectorEnv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from gymnasium.core import RenderFrame


class JNPWrapper:
    """Wrapper for vectorized Gymnasium environments that converts to JAX arrays."""

    def __init__(self, env: Any) -> None:
        self._env = env

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Batched observation shape: (num_envs, obs_dim)."""
        return self._env.observation_space.shape

    @property
    def n_observation_features(self) -> int:
        """Number of observation features (single env)."""
        return int(self._env.envs[0].observation_space.shape[0])

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Batched action shape."""
        return self._env.action_space.shape

    @property
    def n_action_features(self) -> int:
        """Number of action features (single env)."""
        action_space = self._env.envs[0].action_space
        if hasattr(action_space, "n"):  # Discrete
            return int(action_space.n)
        return int(action_space.shape[0])  # Box/continuous

    @property
    def n_actions(self) -> int:
        """Alias for n_action_features for backward compatibility."""
        return self.n_action_features

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Returns a tuple of np random seeds for the wrapped envs."""
        return self._env.get_attr("np_random")

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Returns a tuple of the numpy random number generators for the wrapped
        envs.
        """
        return self._env.get_attr("np_random")

    def step(
        self, action: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        next_obs, reward, terminated, truncated, _info = self._env.step(
            np.asarray(action)
        )
        return (
            jnp.asarray(next_obs),
            jnp.asarray(reward),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
        )

    def reset(
        self,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> jax.Array:
        obs, _info = self._env.reset(seed=seed, options=options)
        return jnp.asarray(obs)

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return self._env.call(name, *args, **kwargs)

    def render(self) -> tuple[RenderFrame, ...] | None:
        return self._env.render()

    def get_attr(self, name: str) -> tuple[Any, ...]:
        return self._env.get_attr(name)

    def set_attr(self, name: str, values: list[Any] | tuple[Any, ...]) -> None:
        self._env.set_attr(name, values)

    def close_extras(self, **kwargs: Any) -> None:
        self._env.close_extras(**kwargs)
