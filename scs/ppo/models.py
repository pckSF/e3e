from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from flax import nnx

from scs.nn_modules import (
    NNTrainingState,
    construct_mlp,
    get_activation_function,
    get_optimizer,
)

if TYPE_CHECKING:
    import jax

    from scs.env_wrapper import JNPWrapper
    from scs.ppo.agent_config import PPOConfig


def make_ppo_train_state(
    env: JNPWrapper,
    agent_config: PPOConfig,
    rngs: nnx.Rngs,
) -> NNTrainingState:
    """Creates training state for for the PPO policy-value network."""
    model = PPOModel(
        input_features=env.n_observation_features,
        action_shape=env.n_actions,
        value_hidden_sizes=agent_config.value_hidden_sizes,
        policy_hidden_sizes=agent_config.policy_hidden_sizes,
        activation=get_activation_function(agent_config.activation),
        rngs=rngs,
    )
    train_state = NNTrainingState.create(
        model_def=nnx.graphdef(model),
        model_state=nnx.state(model, nnx.Param),
        optimizer=get_optimizer(agent_config, model),
    )
    return train_state


class PPOModel(nnx.Module):
    """Actor-critic network with separate policy and value heads.

    The network uses two independent MLP branches: one outputs logits for a
    categorical policy over discrete actions, and the other outputs a scalar
    state-value estimate.
    """

    def __init__(
        self,
        input_features: int,
        action_shape: int,
        value_hidden_sizes: tuple[int, ...],
        policy_hidden_sizes: tuple[int, ...],
        rngs: nnx.Rngs,
        use_layernorm: bool = True,
        activation: Callable[[jax.Array], jax.Array] = nnx.relu,
        kernel_initializer: nnx.Initializer = nnx.initializers.orthogonal(),
    ) -> None:
        """Initializes the policy-value network.

        Args:
            input_features: Dimension of the observation space.
            action_shape: Dimension of the action space.
            value_hidden_sizes: Hidden layer sizes for the value MLP.
            policy_hidden_sizes: Hidden layer sizes for the policy MLP.
            rngs: Flax NNX random number generators.
            use_layernorm: Whether to apply layer normalization after each hidden
                layer.
            activation: Activation function applied after each hidden layer.
            kernel_initializer: Initializer for layer weights.
        """
        # Setup value-network layers
        self.value_mlp = construct_mlp(
            input_features=input_features,
            hidden_layers_sizes=value_hidden_sizes,
            use_layernorm=use_layernorm,
            kernel_initializer=kernel_initializer,
            activation=activation,
            rngs=rngs,
        )
        self.value = nnx.Linear(
            in_features=value_hidden_sizes[-1],
            out_features=1,
            kernel_init=kernel_initializer,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # Setup policy-network layers
        self.policy_mlp = construct_mlp(
            input_features=input_features,
            hidden_layers_sizes=policy_hidden_sizes,
            use_layernorm=use_layernorm,
            kernel_initializer=kernel_initializer,
            activation=activation,
            rngs=rngs,
        )
        self.policy_logits = nnx.Linear(
            in_features=policy_hidden_sizes[-1],
            out_features=action_shape,
            kernel_init=kernel_initializer,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def get_policy_logits(self, observation: jax.Array) -> jax.Array:
        """Calls just the policy branch of the network."""
        p = self.policy_mlp(observation)
        return self.policy_logits(p)

    def get_values(self, observation: jax.Array) -> jax.Array:
        """Calls just the value branch of the network."""
        v = self.value_mlp(observation)
        return self.value(v)

    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Computes policy logits and value estimates for observations.

        Returns:
            A tuple containing:

            - The policy logits (unnormalized log probabilities over actions).
            - The value-function estimates.
        """
        v = self.value_mlp(observation)
        p = self.policy_mlp(observation)
        return (
            self.policy_logits(p),
            self.value(v),
        )
