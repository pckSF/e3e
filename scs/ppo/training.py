from __future__ import annotations

from functools import partial
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scs.data import (
    TrajectoryData,
    separate_trajectory_rollouts,
)
from scs.evaluation import evaluation_trajectory
from scs.ppo.agent import (
    PPOLossComponents,
    update_step,
)
from scs.ppo.rollouts import collect_trajectories

if TYPE_CHECKING:
    from flax import nnx

    from mujoco_playground import State
    from scs.data_logging import DataLogger
    from scs.env_wrapper import EnvTrainingWrapper
    from scs.nn_modules import NNTrainingState
    from scs.ppo.agent_config import PPOConfig


def updates_on_rollout(
    train_state: NNTrainingState,
    trajectories: TrajectoryData,
    config: PPOConfig,
    key: jax.Array,
) -> tuple[NNTrainingState, jax.Array, PPOLossComponents]:
    """Performs multiple update passes on one set of collected trajectories.

    Args:
        train_state: The neural network model's training state container.
        trajectories: Trajectories of experience collected from the environment.
        config: The agent's configuration.
        key: A JAX random key for generating batch indices.

    Returns:
        A tuple containing the updated training state, the loss values for
        each training step, and the loss components.
    """

    def _update_on_rollout(
        train_state: NNTrainingState,
        key: jax.Array,
    ) -> tuple[NNTrainingState, tuple[jax.Array, PPOLossComponents]]:
        """Performs one update pass based on a sampled batch split of the data."""
        # Obtain batch indices via permutation of index array
        batch_indices = jax.random.permutation(
            key, config.batch_size * config.n_batches
        ).reshape((config.n_batches, config.batch_size))
        train_state, (loss, loss_components) = jax.lax.scan(
            partial(
                update_step,
                trajectories=trajectories,
                config=config,
            ),
            train_state,
            batch_indices,
        )
        return train_state, (loss, loss_components)

    keys = jax.random.split(key, num=config.n_epochs_per_rollout)
    train_state, (losses, loss_components) = jax.lax.scan(
        _update_on_rollout,
        train_state,
        keys,
    )
    return train_state, losses, loss_components


@partial(jax.jit, static_argnums=(2, 3, 4), donate_argnums=(0, 1))
def training_epoch(
    train_state: NNTrainingState,
    env_state: State,
    env_def: EnvTrainingWrapper,
    config: PPOConfig,
    n_loops: int,
    key: jax.Array,
) -> tuple[NNTrainingState, State, jax.Array, PPOLossComponents]:
    """Executes multiple training loops as a single JIT-compiled epoch.

    Each training loop collects a rollout segment of length
    ``config.n_actor_steps`` across ``config.n_actors`` parallel environments,
    then performs multiple PPO update passes on the collected trajectory data.

    Args:
        train_state: Current training state containing model parameters.
        env_state: Vectorized environment state at the start of the epoch.
        env_def: Environment wrapper providing step and reset functions.
        config: PPO configuration with rollout and optimization settings.
        n_loops: Number of training loops to execute within this epoch.
        key: PRNG key split internally for each training loop.

    Returns:
        A tuple containing the updated training state, final environment state,
        loss values reshaped to ``(n_loops * n_epochs_per_rollout, n_batches)``,
        and loss components with the same batch structure.
    """
    keys = jax.random.split(key, num=n_loops)

    def _training_loop(
        carry: tuple[NNTrainingState, State],
        key: jax.Array,
    ) -> tuple[tuple[NNTrainingState, State], tuple[jax.Array, PPOLossComponents]]:
        """Collects trajectories and performs PPO updates."""
        train_state, env_state = carry
        rollout_key, update_key = jax.random.split(key, num=2)
        trajectories, env_state = collect_trajectories(
            train_state,
            env_state,
            env_def,
            config,
            rollout_key,
        )
        trajectories = separate_trajectory_rollouts(trajectories, config)
        train_state, loss, loss_components = updates_on_rollout(
            train_state,
            trajectories,
            config,
            update_key,
        )
        return (train_state, env_state), (loss, loss_components)

    (train_state, env_state), (losses, loss_components) = jax.lax.scan(
        _training_loop,
        (train_state, env_state),
        keys,
    )
    return (
        train_state,
        env_state,
        losses.reshape((-1, config.n_batches)),
        jax.tree.map(
            lambda leaf: leaf.reshape((-1, config.n_batches)), loss_components
        ),
    )


def train_agent(
    train_state: NNTrainingState,
    env_def: EnvTrainingWrapper,
    config: PPOConfig,
    data_logger: DataLogger,
    max_training_loops: int,
    rngs: nnx.Rngs,
) -> tuple[NNTrainingState, jax.Array, jax.Array, jax.Array]:
    """Trains a PPO agent over multiple training loop iterations.

    Checkpoints and CSV logs are written through ``data_logger``.

    Args:
        train_state: Initial training state containing model definition, parameters,
            and the corresponding optimizer.
        env_def: Environment wrapped in vectorization wrapper.
        config: PPO configuration with training hyperparameters.
        data_logger: Logger responsible for persisting checkpoints and metrics.
        max_training_loops: Number of outer training loops to perform.
        rngs: Container of PRNG streams used during training and evaluation.

    Returns:
        Final training state along with histories of mean loss, evaluation reward,
        and KL divergence estimates stored as JAX arrays.
    """
    data_logger.log_info(f"Training PPO agent on environment: {config.env_name}")
    data_logger.store_metadata("config", config.to_dict())
    loss_history: list[float] = []
    kl_estimate_history: list[float] = []
    eval_history: list[float] = []
    n_epochs = max_training_loops // config.evaluation_frequency
    data_logger.log_info(
        f"Training {n_epochs} epochs with {config.evaluation_frequency} "
        f"training loops each; based on evaluation frequency of "
        f"{config.evaluation_frequency} and {max_training_loops} max training loops",
        print_message=True,
    )
    n_env_steps = 0

    env_state = env_def.reset(jax.random.split(rngs.training(), num=config.n_actors))
    progress_bar: tqdm = tqdm(range(n_epochs), desc="Training Epochs")
    for epoch in progress_bar:
        start_time = time.time()
        train_state, env_state, losses, loss_components = training_epoch(
            train_state,
            env_state,
            env_def,
            config,
            config.evaluation_frequency,
            rngs.training(),
        )
        end_time = time.time()
        data_logger.log_info(
            f"Iteration {epoch + 1}/{n_epochs} "
            f"completed in {end_time - start_time:.2f} seconds"
        )
        n_env_steps += config.env_steps_per_rollout * config.evaluation_frequency
        data_logger.save_csv_rows_async("losses", losses)
        data_logger.save_csv_rows_async("ppo_value", loss_components.ppo_value)
        data_logger.save_csv_rows_async("value_loss", loss_components.value_loss)
        data_logger.save_csv_rows_async("entropy", loss_components.entropy)
        data_logger.save_csv_rows_async("kl_estimate", loss_components.kl_estimate)
        loss_history.append(float(np.mean(losses)))
        kl_estimate_history.append(float(np.mean(loss_components.kl_estimate)))
        data_logger.save_checkpoint(
            filename="checkpoint",
            data=train_state.model_state,
        )
        eval_rewards = evaluation_trajectory(
            train_state,
            env_def,
            config,
            rngs.evaluation(),
        )
        data_logger.save_csv_rows_async("eval_rewards", eval_rewards)
        eval_history.append(float(np.mean(eval_rewards)))
        progress_bar.set_postfix(
            {
                "loss": f"{loss_history[-1]:.4f}",
                "eval reward": f"{eval_history[-1]:.2f}",
                "kl estimate": f"{kl_estimate_history[-1]:.4f}",
                "env steps": f"{n_env_steps}",
            }
        )
    data_logger.wait_until_finished()
    return (
        train_state,
        jnp.array(loss_history),
        jnp.array(eval_history),
        jnp.array(kl_estimate_history),
    )
