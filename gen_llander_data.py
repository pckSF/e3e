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
    "_f5632d7f77300b6c8cc26036f8365a44/20260319_172631/checkpoint_00027"
)
episodes: int = 1000
max_length: int = 250
data_dir: str = "data/cllander_trajectories"
seed: int = 0
################################################################################

if __name__ == "__main__":
    env = JNPWrapper(gym.make("LunarLander-v3", continuous=True))
    model = load_model(checkpoint_path, env)
    key = jax.random.PRNGKey(seed)
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Collecting trajectory data with seed: {seed}")

    out_path = Path(data_dir) / f"episodes_{episodes}_maxlen_{max_length}.hdf5"
    metadata = {
        "env_name": "LunarLander-v3-continuous",
        "checkpoint_path": checkpoint_path,
        "n_episodes": episodes,
        "max_length": max_length,
        "seed": seed,
    }
    with TrajectoryWriter(out_path, metadata=metadata) as writer:
        collect_data(model, env, episodes, max_length, key, writer)
