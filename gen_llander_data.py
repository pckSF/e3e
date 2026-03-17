from __future__ import annotations

import gymnasium as gym
import jax

from scs.collect_data import collect_data, load_model
from scs.env_wrapper import JNPWrapper

################################################################################
# Settings
################################################################################
checkpoint_path: str = (
    "/home/pcksf/projects/e3e/logs/ppo_LunarLander-v3"
    "_a275b94b4a0f84073ded1111e174b93f/20260317_145825/checkpoint_00010"
)
data_path: str = "data/llander_trajectories.pkl"
seed: int = 0
################################################################################

if __name__ == "__main__":
    model = load_model(checkpoint_path)
    key = jax.random.PRNGKey(seed)
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Collecting trajectory data with seed: {seed}")
    env = JNPWrapper(gym.make("LunarLander-v3"))
    data = collect_data(model, env, 10000, 250, key)
    data.save(data_path)
    print(f"Saved trajectory data to: {data_path}")
