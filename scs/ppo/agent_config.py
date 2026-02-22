from __future__ import annotations

from typing import Protocol, cast

from scs.configuration import make_config


class PPOConfig(Protocol):
    """Defines the structure of a PPO configuration object.

    Attributes:
        env_name: The name of the Mujoco-playground environment to train on.
        lr_policyvalue: The learning rate for the shared policy-value optimizer.
        lr_schedule_policyvalue: The learning rate schedule type (``'constant'``,
            ``'linear'``, or ``'exponential'``).
        lr_end_value_policyvalue: The end value for the learning rate schedule.
        lr_decay_policyvalue: The decay rate for the learning rate schedule.
        optimizer_policyvalue: The optimizer to use for training (``'adam'`` or
            ``'sgd'``).
        value_hidden_sizes: The sizes of the hidden layers for the value network.
        policy_hidden_sizes: The sizes of the hidden layers for the policy network.
        activation: The activation function to use (``'relu'``, ``'swish'``,
            ``'tanh'``, or ``'gelu'``).
        layernorm: Whether to use layer normalization in the networks.
        discount_factor: The discount factor for future rewards (gamma).
        clip_parameter: The clipping parameter for the PPO surrogate objective.
        entropy_coefficient: The coefficient for the entropy term in the loss.
        gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
        n_actors: The number of parallel actors collecting experience.
        n_rollout_steps: The number of steps each actor collects per rollout.
        n_batches: The number of batches to split the rollout into.
        batch_size: The number of samples per batch for training.
        required_rollout_steps: The total steps each actor must collect per rollout
            to satisfy batch requirements. Computed as:
            ``n_rollout_steps * ((batch_size * n_batches) // n_actors)``.
        n_epochs_per_rollout: The number of epochs trained on each set of rollout
            data.
        value_loss_coefficient: Coefficient for the value loss term.
        evaluation_frequency: Frequency of running policy evaluations (in training
            loops).
        normalize_advantages: Whether to normalize advantages.
        normalize_observations: Whether to normalize observations.
        max_std_value: Upper bound for the running standard deviation estimate used
            in observation normalization, no clipping when set to zero.
        min_std_value: Lower bound for the running standard deviation estimate to
            prevent division by near-zero values.
        max_obs_abs_value: If positive, clip normalized observations to this absolute
            value. No clipping when zero.
        reward_scaling: The scaling factor applied to rewards.
        max_training_loops: The maximum number of training loops to perform.
        env_steps_per_rollout: Total environment steps collected per rollout across
            all actors. Computed as: ``required_rollout_steps * n_actors``.
        seed: The random seed for reproducibility.
    """

    env_name: str
    lr_policyvalue: float
    lr_schedule_policyvalue: str
    lr_end_value_policyvalue: float
    lr_decay_policyvalue: float
    optimizer_policyvalue: str
    value_hidden_sizes: tuple[int, ...]
    policy_hidden_sizes: tuple[int, ...]
    activation: str
    layernorm: bool
    discount_factor: float
    clip_parameter: float
    entropy_coefficient: float
    gae_lambda: float
    n_actors: int
    n_rollout_steps: int
    n_batches: int
    batch_size: int
    required_rollout_steps: int
    n_epochs_per_rollout: int
    value_loss_coefficient: float
    evaluation_frequency: int
    normalize_advantages: bool
    normalize_observations: bool
    max_std_value: float
    min_std_value: float
    max_obs_abs_value: float
    reward_scaling: float
    max_training_loops: int
    env_steps_per_rollout: int
    seed: int

    def to_dict(self) -> dict: ...


def create_ppo_config(
    env_name: str,
    lr_policyvalue: float,
    lr_schedule_policyvalue: str,
    lr_end_value_policyvalue: float,
    lr_decay_policyvalue: float,
    optimizer_policyvalue: str,
    value_hidden_sizes: tuple[int, ...],
    policy_hidden_sizes: tuple[int, ...],
    activation: str,
    layernorm: bool,
    discount_factor: float,
    clip_parameter: float,
    entropy_coefficient: float,
    gae_lambda: float,
    n_actors: int,
    n_rollout_steps: int,
    n_batches: int,
    batch_size: int,
    n_epochs_per_rollout: int,
    value_loss_coefficient: float,
    evaluation_frequency: int,
    normalize_advantages: bool,
    normalize_observations: bool,
    max_std_value: float,
    min_std_value: float,
    max_obs_abs_value: float,
    reward_scaling: float,
    max_training_loops: int,
    seed: int,
) -> PPOConfig:
    """Generates the configuration for the PPO agent.

    This function provides a base configuration with sensible defaults and
    serves as a template to manually create sets of PPO parameters. See
    ``PPOConfig`` for a description of all configuration attributes.

    The function validates input constraints and computes derived attributes:

    - **required_rollout_steps**: Calculated as
      ``n_rollout_steps * ((batch_size * n_batches) // n_actors)`` to ensure enough
      data is collected to fill all training batches.
    - **env_steps_per_rollout**: Calculated as ``required_rollout_steps * n_actors``,
      representing total environment interactions per rollout.

    Example:
        To obtain data for 32 batches of size 1024 with 2048 actors with a desired
        rollout length of 30, we run the 2048 actors for
        ``((1024 * 32) // 2048) * 30 = 16 * 30 = 480`` steps and then split the
        resulting set of ``2048 * 480`` steps into 32 batches of 1024 samples of
        length 30.

    Returns:
        A ``PPOConfig`` object containing the PPO hyperparameters.

    Raises:
        ValueError: If batch/rollout constraints are violated or invalid
            optimizer/schedule values are provided.
    """
    if (batch_size * n_batches) % n_actors != 0:
        raise ValueError(
            f"Total training samples (batch_size * n_batches = "
            f"{batch_size} * {n_batches} = {batch_size * n_batches}) "
            f"be evenly divisible by n_actors ({n_actors}). "
            f"Current remainder: {(batch_size * n_batches) % n_actors}."
        )
    if max_training_loops % evaluation_frequency != 0:
        raise ValueError(
            f"Max training loops must be evenly divisible by evaluation frequency; "
            f"received max training loops {max_training_loops} and evaluation "
            f"frequency {evaluation_frequency}."
        )
    if optimizer_policyvalue.lower() not in {"adam", "sgd"}:
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_policyvalue}"
        )
    if lr_schedule_policyvalue.lower() not in {"constant", "linear", "exponential"}:
        raise ValueError(
            f"Unsupported learning rate schedule, "
            f"expected 'constant', 'linear', or 'exponential'; "
            f"received {lr_schedule_policyvalue}"
        )
    if activation.lower() not in {"relu", "swish", "tanh", "gelu"}:
        raise ValueError(
            f"Unsupported activation function, expected 'relu', 'swish', 'tanh' or "
            f"'gelu'; received: {activation}"
        )

    required_rollout_steps = n_rollout_steps * ((batch_size * n_batches) // n_actors)
    env_steps_per_rollout = required_rollout_steps * n_actors

    config = make_config(
        {
            "env_name": env_name,
            "lr_policyvalue": lr_policyvalue,
            "lr_schedule_policyvalue": lr_schedule_policyvalue.lower(),
            "lr_end_value_policyvalue": lr_end_value_policyvalue,
            "lr_decay_policyvalue": lr_decay_policyvalue,
            "optimizer_policyvalue": optimizer_policyvalue.lower(),
            "value_hidden_sizes": value_hidden_sizes,
            "policy_hidden_sizes": policy_hidden_sizes,
            "activation": activation.lower(),
            "layernorm": layernorm,
            "discount_factor": discount_factor,
            "clip_parameter": clip_parameter,
            "entropy_coefficient": entropy_coefficient,
            "gae_lambda": gae_lambda,
            "n_actors": n_actors,
            "n_rollout_steps": n_rollout_steps,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "required_rollout_steps": required_rollout_steps,
            "n_epochs_per_rollout": n_epochs_per_rollout,
            "value_loss_coefficient": value_loss_coefficient,
            "evaluation_frequency": evaluation_frequency,
            "normalize_advantages": normalize_advantages,
            "normalize_observations": normalize_observations,
            "max_std_value": max_std_value,
            "min_std_value": min_std_value,
            "max_obs_abs_value": max_obs_abs_value,
            "reward_scaling": reward_scaling,
            "max_training_loops": max_training_loops,
            "env_steps_per_rollout": env_steps_per_rollout,
            "seed": seed,
        }
    )
    return cast("PPOConfig", config)
