from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import jax

from scs.collect_data import TrajectoryWriter, collect_data, load_model
from scs.env_wrapper import JNPWrapper

################################################################################
# Settings
################################################################################
checkpoint_path: str = (
    "/home/pcksf/projects/e3e/logs/ppo_LunarLander-v3"
    "_a275b94b4a0f84073ded1111e174b93f/20260317_145825/checkpoint_00015"
)
episodes: int = 1000
max_length: int = 250
data_dir: str = "data/llander_trajectories"
seed: int = 0
################################################################################

if __name__ == "__main__":
    model = load_model(checkpoint_path)
    key = jax.random.PRNGKey(seed)
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Collecting trajectory data with seed: {seed}")
    env = JNPWrapper(gym.make("LunarLander-v3"))

    out_path = Path(data_dir) / f"episodes_{episodes}_maxlen_{max_length}.hdf5"
    metadata = {
        "env_name": "LunarLander-v3",
        "checkpoint_path": checkpoint_path,
        "n_episodes": episodes,
        "max_length": max_length,
        "seed": seed,
    }
    with TrajectoryWriter(out_path, metadata=metadata) as writer:
        collect_data(model, env, episodes, max_length, key, writer)
