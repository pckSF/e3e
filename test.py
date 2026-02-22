from __future__ import annotations

import time

from flax import nnx
import jax

from mujoco_playground import registry
from scs.buffer import initialize_simple_buffer, uniform_sample_batches
from scs.configuration import get_config
from scs.env_wrapper import EnvTrainingWrapper
from scs.sac.agent_config import create_sac_config
from scs.sac.rollouts import warmup_buffer

# Setup config
base_config, config_hash = get_config(
    "scs/configs/sac/sac_walkerwalk.json", create_sac_config, with_hash=True
)
initial_seed = base_config.seed

# Setup logging

seed = initial_seed
agent_config = base_config

# Setup RNG
rngs = nnx.Rngs(
    seed,
    config=seed + 1,
    training=seed + 2,
    sample=seed + 3,
    evaluation=seed + 4,
)

# Setup Environment
env_def = EnvTrainingWrapper(
    registry.load(agent_config.env_name),
    registry.get_default_config(agent_config.env_name),
    agent_config,
)

buffer_state = initialize_simple_buffer(
    agent_config.replay_buffer_size,
    env_def.observation_size,
    env_def.action_size,
)
env_state = env_def.reset(jax.random.split(rngs.sample(), num=agent_config.n_actors))

# Create the model
start_time = time.time()
buffer_state = warmup_buffer(
    env_state,
    buffer_state,
    env_def,
    agent_config,
    rngs.sample(),
)
end_time = time.time()
print(f"Warmup time: {end_time - start_time:.2f} seconds")

# Free cache for next iteration
jax.clear_caches()

batch = uniform_sample_batches(
    buffer_state,
    100,
    5,
    rngs.sample(),
)
