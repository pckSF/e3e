from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from flax import nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from scs.data import TrajectoryData
from scs.nn_modules import get_activation_function
from scs.ppo.models import PPOModel

if TYPE_CHECKING:
    from scs.data_logging import DataLogger
    from scs.env_wrapper import JNPWrapper


def load_model(path: str) -> PPOModel:
    model_path = Path(path)
    if not model_path.is_dir():
        raise ValueError("Model path must be a directory containing model checkpoints.")
    run_dir = model_path.parent
    config = json.loads((run_dir / "config.json").read_text())

    env = gym.make(config["env_name"])
    n_obs = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    env.close()

    abstract_model = nnx.eval_shape(
        lambda: PPOModel(
            input_features=n_obs,
            action_shape=n_actions,
            value_hidden_sizes=tuple(config["value_hidden_sizes"]),
            policy_hidden_sizes=tuple(config["policy_hidden_sizes"]),
            use_layernorm=config["layernorm"],
            activation=get_activation_function(config["activation"]),
            rngs=nnx.Rngs(0),
        )
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(model_path, abstract_state)

    return nnx.merge(graphdef, restored_state)


def collect_data(
    model: PPOModel,
    env: JNPWrapper,
    n_episodes: int,
    max_length: int,
    key: jax.Array,
    logger: DataLogger,
) -> None:
    episode_keys = jax.random.split(key, num=n_episodes)

    def _transition(
        observation: jax.Array,
        action_key: jax.Array,
    ) -> tuple[jax.Array, TrajectoryData]:
        """Performs one vectorized transition and captures trajectory data."""
        policy_logits = model.get_policy_logits(observation)
        action = jax.random.categorical(action_key, policy_logits)
        next_observation, reward, terminated, truncated = env.step(action)
        timestep = TrajectoryData(
            observation,
            action,
            policy_logits,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        reset_mask = jnp.logical_or(terminated, truncated)
        if jnp.any(reset_mask):
            next_observation = env.reset(options={"reset_mask": np.asarray(reset_mask)})
        return next_observation, timestep

    for ek in tqdm(episode_keys, total=n_episodes, desc="Collecting episodes"):
        step_keys = jax.random.split(ek, num=max_length)
        obs = env.reset()
        for sk in step_keys:
            obs, timestep = _transition(obs, sk)
            logger.save_csv_rows_async("observations", timestep.observations)
            logger.save_csv_rows_async("actions", timestep.actions)
            logger.save_csv_rows_async("rewards", timestep.rewards)
            logger.save_csv_rows_async("next_observations", timestep.next_observations)
            logger.save_csv_rows_async("terminals", timestep.terminals)
            logger.save_csv_rows_async("truncated", timestep.truncated)
