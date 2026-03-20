from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from flax import nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from scs.data import TrajectoryData
from scs.nn_modules import get_activation_function
from scs.ppo.models import PPOModel
from scs.rl_computations import on_distribution_action_normal_log_density

if TYPE_CHECKING:
    from types import TracebackType

    from scs.env_wrapper import JNPWrapper

# Fields written to the HDF5 file, in canonical order.
_TRAJECTORY_FIELDS: tuple[str, ...] = (
    "observations",
    "actions",
    "action_log_densities",
    "rewards",
    "next_observations",
    "terminals",
    "truncations",
)


class TrajectoryWriter:
    """Incrementally writes trajectory data to an HDF5 file.

    Datasets are created on the first call to ``add_episode`` and extended
    on subsequent calls, so memory usage stays constant regardless of total
    dataset size.

    Args:
        path: Destination HDF5 file; parent directories are created if needed.
        metadata: Key-value pairs stored as HDF5 root attributes.
        compression: HDF5 compression filter applied to every dataset.

    Example::

        with TrajectoryWriter("data.hdf5") as writer:
            writer.add_episode(episode_arrays)
    """

    def __init__(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        compression: str = "gzip",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._metadata = metadata or {}
        self._file: h5py.File | None = None
        self._total_steps = 0

    def __enter__(self) -> Self:
        self._file = h5py.File(self._path, "w")
        for key, value in self._metadata.items():
            self._file.attrs[key] = value
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.attrs["total_timesteps"] = self._total_steps
            self._file.close()
            self._file = None

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        """Append one episode's worth of timesteps to the HDF5 datasets.

        Args:
            episode: Mapping from field name to array of shape
                ``[episode_length, ...]``.

        Raises:
            AssertionError: If called outside of a ``with`` block.
        """
        assert self._file is not None, "Writer must be used as a context manager."
        n_steps = episode["observations"].shape[0]

        for name in _TRAJECTORY_FIELDS:
            arr = episode[name]
            if name not in self._file:
                maxshape = (None,) + arr.shape[1:]
                chunks = (min(n_steps, 1024),) + arr.shape[1:]
                self._file.create_dataset(
                    name,
                    data=arr,
                    maxshape=maxshape,
                    chunks=chunks,
                    compression=self._compression,
                )
            else:
                ds = self._file[name]
                old_len = ds.shape[0]
                ds.resize(old_len + n_steps, axis=0)
                ds[old_len:] = arr

        self._total_steps += n_steps


def load_model(path: str, env: JNPWrapper) -> PPOModel:
    model_path = Path(path)
    if not model_path.is_dir():
        raise ValueError("Model path must be a directory containing model checkpoints.")
    run_dir = model_path.parent
    config = json.loads((run_dir / "config.json").read_text())

    abstract_model = nnx.eval_shape(
        lambda: PPOModel(
            input_features=env.observation_shape[0],
            action_shape=env.action_shape[0],
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
    writer: TrajectoryWriter,
) -> None:
    episode_keys = jax.random.split(key, num=n_episodes)

    def _transition(
        observation: jax.Array,
        action_key: jax.Array,
    ) -> tuple[jax.Array, TrajectoryData, jax.Array]:
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
        return next_observation, timestep, reset_mask

    for ek in tqdm(episode_keys, total=n_episodes, desc="Collecting episodes"):
        step_keys = jax.random.split(ek, num=max_length)
        obs = env.reset()

        # Buffer one episode in numpy arrays
        ep_data: dict[str, list[np.ndarray]] = {f: [] for f in _TRAJECTORY_FIELDS}
        for sk in step_keys:
            obs, timestep, reset_mask = _transition(obs, sk)
            ep_data["observations"].append(np.asarray(timestep.observations))
            ep_data["actions"].append(np.asarray(timestep.actions))
            ep_data["action_log_densities"].append(
                np.asarray(timestep.action_log_densities)
            )
            ep_data["rewards"].append(np.asarray(timestep.rewards))
            ep_data["next_observations"].append(np.asarray(timestep.next_observations))
            ep_data["terminals"].append(np.asarray(timestep.terminals))
            ep_data["truncations"].append(np.asarray(timestep.truncations))
            if np.any(reset_mask):
                break

        # Stack episode and write to HDF5 in one resize call
        writer.add_episode({k: np.stack(v) for k, v in ep_data.items()})
