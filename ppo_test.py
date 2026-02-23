from __future__ import annotations

import argparse
from pathlib import Path
import time

from flax import nnx
import gymnasium as gym
import jax

from scs.configuration import get_config
from scs.data_logging import DataLogger
from scs.env_wrapper import JNPWrapper
from scs.ppo import train_agent
from scs.ppo.agent_config import create_ppo_config
from scs.ppo.models import make_ppo_train_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="scs/configs/ppo/ppo_llander.json",
        help="Path to the PPO configuration JSON file.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of different random seeds to run.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_file)
    if config_path.suffix.lower() != ".json" or not config_path.is_file():
        parser.error(
            f"config_file must be a path to an existing JSON file: {args.config_file}"
        )

    if args.n_seeds < 1:
        parser.error(f"n_seeds must be a positive integer, got: {args.n_seeds}")

    return args


def main() -> None:
    args = parse_args()

    # Setup config
    base_config, config_hash = get_config(
        args.config_file, create_ppo_config, with_hash=True
    )
    initial_seed = base_config.seed
    log_dir = f"logs/ppo_{base_config.env_name}_{config_hash}"

    for i in range(args.n_seeds):
        # Setup logging
        logger = DataLogger(log_dir)
        logger.log_info(f"Experiment configuration hash: {config_hash}")

        seed = initial_seed + i
        agent_config = get_config(args.config_file, create_ppo_config, seed=seed)

        # Setup RNG
        rngs = nnx.Rngs(
            seed,
            config=seed + 1,
            training=seed + 2,
            sample=seed + 3,
            evaluation=seed + 4,
        )
        logger.log_info(f"Starting training with seed: {seed}", print_message=True)

        # Setup Environment
        env = JNPWrapper(
            gym.make_vec(
                "CartPole-v1", num_envs=agent_config.n_actors, vectorization_mode="sync"
            ),
        )

        # Create the model
        train_state = make_ppo_train_state(
            env,
            agent_config,
            rngs,
        )

        start_time = time.time()
        train_state, losses, eval_rewards, kl_estimates = train_agent(
            train_state=train_state,
            env=env,
            config=agent_config,
            data_logger=logger,
            max_training_loops=agent_config.max_training_loops,
            rngs=rngs,
        )
        end_time = time.time()
        logger.log_info(f"Completed training in {end_time - start_time:.2f} seconds")

        # Free cache for next iteration
        jax.clear_caches()


if __name__ == "__main__":
    main()
