from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from flax import (
    nnx,
    struct,
)
import jax
import jax.numpy as jnp

from scs.data import (
    TrajectoryData,
    get_batch_from_trajectory,
)
from scs.rl_computations import (
    calculate_gae,
)

if TYPE_CHECKING:
    from scs.nn_modules import NNTrainingState
    from scs.ppo.agent_config import PPOConfig
    from scs.ppo.models import PPOModel


@struct.dataclass
class PPOLossComponents:
    ppo_value: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    kl_estimate: jax.Array


def loss_fn(
    model: PPOModel,
    batch: TrajectoryData,
    config: PPOConfig,
) -> tuple[jax.Array, PPOLossComponents]:
    """Computes the PPO loss for a batch of trajectory data.

    This function calculates the combined loss for the actor-critic model, which
    includes the policy loss (PPO's clipped objective), the value function loss,
    and an entropy bonus to encourage exploration.

    Since the optax optimizers perform gradient descent, we return the negative
    of the total loss. Reminder, the loss is composed of a:

    - ``ppo_value``, which are the weighted advantages that we want to maximize.
    - ``value_loss``, which we want to minimize.
    - ``entropy``, which we want to maximize.

    Args:
        model: The actor-critic model being trained.
        batch: A batch of trajectory data from rollouts.
        config: The agent's configuration.

    Returns:
        The total PPO loss for the batch.
    """
    policy_logits, values = model(batch.observations)
    next_values = model.get_values(batch.next_observations)
    values = jnp.squeeze(values)
    next_values = jnp.squeeze(jax.lax.stop_gradient(next_values))

    advantages = calculate_gae(
        batch.rewards * config.reward_scaling,
        jax.lax.stop_gradient(values),
        next_values,
        batch.terminals,
        batch.truncated,
        config.discount_factor,
        config.gae_lambda,
    )
    returns = advantages + jax.lax.stop_gradient(values)
    value_loss = jnp.mean((returns - values) ** 2)

    if config.normalize_advantages:
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Discrete action entropy: H = -sum(p * log(p))
    policy = jax.nn.softmax(policy_logits)
    log_policy = jax.nn.log_softmax(policy_logits)
    entropy = jnp.mean(-jnp.sum(policy * log_policy, axis=-1))

    old_log_policy = jax.nn.log_softmax(batch.policy_logits)

    actions = batch.actions[..., jnp.newaxis]
    actions_log_probs = jnp.take_along_axis(log_policy, actions).squeeze()
    old_actions_log_probs = jnp.take_along_axis(old_log_policy, actions).squeeze()
    log_ratios = actions_log_probs - old_actions_log_probs

    density_ratios = jnp.exp(log_ratios)

    policy_gradient_value = density_ratios * advantages
    clipped_pg_value = (
        jax.lax.clamp(
            1.0 - config.clip_parameter, density_ratios, 1.0 + config.clip_parameter
        )
        * advantages
    )
    ppo_value = jnp.minimum(policy_gradient_value, clipped_pg_value).mean()

    kl_estimate = ((density_ratios - 1) - log_ratios).mean()
    return -(
        ppo_value
        - config.value_loss_coefficient * value_loss
        + config.entropy_coefficient * entropy
    ), PPOLossComponents(ppo_value, value_loss, entropy, kl_estimate)


def update_step(
    train_state: NNTrainingState,
    batch_indices: jax.Array,
    trajectories: TrajectoryData,
    config: PPOConfig,
) -> tuple[NNTrainingState, tuple[jax.Array, PPOLossComponents]]:
    """Performs a single training step on a batch of data.

    This function is designed to be used with ``jax.lax.scan`` to iterate over
    a set of batch indices.

    Args:
        train_state: The current training state, the carry in a scan.
        batch_indices: The indices for the data batch to be processed.
        trajectories: The full stacked trajectories of all agent data for the rollout.
        config: The agent's configuration.

    Returns:
        A tuple containing the updated training state and the loss for the batch.
    """
    model = nnx.merge(train_state.model_def, train_state.model_state)
    batch = get_batch_from_trajectory(trajectories, batch_indices)
    grad_loss_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, loss_components), grads = grad_loss_fn(model, batch, config)
    return train_state.apply_gradients(grads), (loss, loss_components)
