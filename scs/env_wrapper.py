from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import struct
import jax
import jax.numpy as jnp

from mujoco_playground import MjxEnv

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    import mujoco
    from mujoco import mjx

    from mujoco_playground import State
    from scs.ppo.agent_config import PPOConfig
    from scs.sac.agent_config import SACConfig


@struct.dataclass
class NormalizeObservationState:
    count: int
    mean: jax.Array
    std: jax.Array
    variance_sum: jax.Array


def create_normalizer(n_features: int) -> NormalizeObservationState:
    """Create a new observation normalizer with default initial values."""
    return NormalizeObservationState(
        count=0,
        mean=jnp.zeros((n_features,)),
        std=jnp.ones((n_features,)),
        variance_sum=jnp.zeros((n_features,)),
    )


def update_normalizer(
    normalizer: NormalizeObservationState,
    observation: jax.Array,
    max_std_value: float,
    min_std_value: float,
) -> NormalizeObservationState:
    """Updates the normalizer state with Welford's algorithm and computes std.

    Note:
        Maybe use exponential moving average to improve numerical stability?
    """
    new_count = normalizer.count + observation.shape[0]
    delta_obs_mean = observation - normalizer.mean
    new_mean = normalizer.mean + jnp.sum(delta_obs_mean, axis=0) / new_count
    delta_obs_new_mean = observation - new_mean
    new_variance_sum = normalizer.variance_sum + jnp.sum(
        delta_obs_mean * delta_obs_new_mean, axis=0
    )
    std = jnp.sqrt(new_variance_sum / jnp.maximum(new_count - 1, 1))
    std = jnp.clip(std, a_min=min_std_value, a_max=max_std_value)
    return NormalizeObservationState(
        count=new_count,
        mean=new_mean,
        std=std,
        variance_sum=new_variance_sum,
    )


def normalize(
    normalizer: NormalizeObservationState,
    env_state: State,
    max_abs_value: float,
) -> State:
    """Apply z-score normalization to the observations in the given state.

    Returns:
        A new state with normalized (and optionally clipped) observations.
    """
    normalized_obs = (env_state.obs - normalizer.mean) / normalizer.std
    if max_abs_value > 0.0:
        normalized_obs = jnp.clip(normalized_obs, -max_abs_value, max_abs_value)
    return env_state.replace(obs=normalized_obs)


class EnvTrainingWrapper(MjxEnv):
    """Wraps an environment definition for vectorized execution, episode
    truncation, and normalization.

    Applies ``jax.vmap`` to ``reset`` and ``step`` for batched execution, tracks
    episode steps, and triggers resets when the episode length limit is reached.
    Can also apply running z-score normalization to observations if the respective
    flag is set.
    """

    def __init__(
        self,
        env_def: MjxEnv,
        env_config: ConfigDict,
        agent_config: PPOConfig | SACConfig,
    ) -> None:
        """Initialize the training wrapper.

        Args:
            env_def: The underlying MJX environment to wrap.
            env_config: Configuration containing ``episode_length`` and other
                environment parameters.
            agent_config: Configuration containing normalization parameters.
        """
        self._env_def = env_def
        self.config = env_config
        self._normalize_observations = agent_config.normalize_observations
        self.max_std_value = agent_config.max_std_value
        self.min_std_value = agent_config.min_std_value
        self.max_obs_abs_value = agent_config.max_obs_abs_value

    def reset(self, key: jax.Array) -> State:
        """Resets the environment for a batch of keys and initializes step counters.

        Optionally initializes observation normalizer.
        """
        env_state = jax.vmap(self._env_def.reset)(key)
        env_state.info["step"] = jnp.zeros(key.shape[0], dtype=jnp.uint32)
        env_state.info["truncated"] = jnp.zeros(key.shape[0], dtype=bool)
        if self._normalize_observations:
            env_state.info["obs_normalizer"] = create_normalizer(
                env_state.obs.shape[-1]
            )
        return env_state

    def get_initial_state(self, key: jax.Array) -> State:
        """Returns an initial state for each environment in the batch.

        Removes the environment state statistics stored in the the ``.info`` dict
        that are not being reset upon environment resets. These are metrics that
        are traced across all environments and not batch specific. For example,
        the observation normalizer state is preserved across resets, whereas the
        step counter is reset to zero.
        """
        env_state = self.reset(key)
        if self._normalize_observations:
            del env_state.info["obs_normalizer"]
        return env_state

    def conditional_reset(self, env_state: State, initial_state: State) -> State:
        """Resets terminated or truncated episodes to the provided initial state."""
        reset_mask = jnp.logical_or(env_state.done, env_state.info["truncated"])

        def _do_reset(current_env_state: State) -> State:
            """Applies the reset to the environment state.

            Optionally ensures that the observation normalizer state is preserved
            across resets. The ``normalizer state`` can safely be removed from the
            ``current_env_state`` since the ``initial_state`` does not contain it.

            Note:
                This version with the ``dict`` operations seems to be as fast as
                using a shape check to skip non-batched leaves in ``_masked_replace``
                with (``if reset_mask.shape[0] != current_leaf.shape[0]``) and
                mapping ``_masked_replace`` across all leafs.
            """
            if self._normalize_observations:
                obs_normalizer_state = current_env_state.info.pop("obs_normalizer")

            def _masked_replace(
                current_leaf: jax.Array, reset_leaf: jax.Array
            ) -> State:
                """Selectively replaces values in a leaf of the state tree.

                The ``reset_mask`` is reshaped to match the dimensions of the leaf
                array to correctly broadcast the mask across its dimensions.
                """
                leaf_mask = reset_mask.reshape(
                    reset_mask.shape + (1,) * (current_leaf.ndim - reset_mask.ndim)
                )
                return jnp.where(leaf_mask, reset_leaf, current_leaf)

            current_env_state = jax.tree.map(
                _masked_replace, current_env_state, initial_state
            )
            if self._normalize_observations:
                current_env_state.info["obs_normalizer"] = obs_normalizer_state

            return current_env_state

        return jax.lax.cond(
            jnp.any(reset_mask),
            _do_reset,
            lambda current_env_state: current_env_state,
            env_state,
        )

    def step(self, env_state: State, action: jax.Array) -> State:
        """Steps the environment with a batch of actions.

        Keeps track of episode step counts and sets truncation flags when the
        episode length limit is reached.

        Optionally normalizes observations and updates the normalizer statistics.
        """
        if self._normalize_observations:
            obs_normalizer = env_state.info.pop("obs_normalizer")
        env_state = jax.vmap(self._env_def.step)(env_state, action)
        env_state.info["step"] += 1
        env_state.info["truncated"] = (
            env_state.info["step"] >= self.config.episode_length
        )
        if self._normalize_observations:
            current_observations = env_state.obs
            env_state = normalize(obs_normalizer, env_state, self.max_obs_abs_value)
            env_state.info["obs_normalizer"] = update_normalizer(
                obs_normalizer,
                current_observations,
                self.max_std_value,
                self.min_std_value,
            )
        return env_state

    @property
    def observation_size(self) -> int:
        """Return the dimensionality of the observation space."""
        return self._env_def.observation_size

    @property
    def action_size(self) -> int:
        """Return the dimensionality of the action space."""
        return self._env_def.action_size

    @property
    def unwrapped(self) -> MjxEnv:
        """Return the original unwrapped environment."""
        return self._env_def

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Returns the underlying MuJoCo model."""
        return self._env_def.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Returns the underlying MJX model."""
        return self._env_def.mjx_model

    @property
    def xml_path(self) -> str:
        """Return the path to the environment's MuJoCo XML model file."""
        return self._env_def.xml_path
