from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from flax import (
    nnx,
    struct,
)
import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    from scs.appo.agent_config import APPOConfig
    from scs.ppo.agent_config import PPOConfig


MODEL_CONFIG_POSTFIX: dict[str, str] = {
    "PPOModel": "policyvalue",
    "APPOModel": "policyvalue",
    "SACPolicy": "policy",
    "SACQvalue": "qvalue",
}


def get_optimizer(
    config: PPOConfig | APPOConfig, model: nnx.Module
) -> optax.GradientTransformation:
    """Creates an optimizer based on the configuration and model class name.

    Args:
        config: The configuration object (``PPOConfig`` or ``SACConfig``).
        model: The model instance whose class name is used to extract the postfix.
            The class name is converted to lowercase to get the postfix (e.g.,
            ``PPOModel`` -> ``"policyvalue"``, ``Policy`` -> ``"policy"``).

    Returns:
        An ``optax`` gradient transformation (optimizer with learning rate schedule).
    """
    # Extract postfix from model class name using the mapping dictionary
    model_name = model.__class__.__name__
    model_postfix = MODEL_CONFIG_POSTFIX[model_name]

    optimizer_name = getattr(config, f"optimizer_{model_postfix}")
    lr_schedule_type = getattr(config, f"lr_schedule_{model_postfix}")
    lr_init = getattr(config, f"lr_{model_postfix}")
    lr_end = getattr(config, f"lr_end_value_{model_postfix}")
    lr_decay = getattr(config, f"lr_decay_{model_postfix}")

    if optimizer_name == "adam":
        optimizer = optax.adam
    elif optimizer_name == "sgd":
        optimizer = optax.sgd
    else:
        raise ValueError(
            f"Unsupported optimizer, expected 'adam' or 'sgd'; "
            f"received: {optimizer_name}"
        )

    n_update_steps = config.n_batches * config.max_training_loops
    # PPOConfig has n_epochs_per_rollout, SACConfig does not
    if hasattr(config, "n_epochs_per_rollout"):
        n_update_steps *= config.n_epochs_per_rollout

    if lr_schedule_type == "constant":
        lr_schedule = optax.constant_schedule(value=lr_init)
    elif lr_schedule_type == "linear":
        lr_schedule = optax.linear_schedule(
            init_value=lr_init,
            end_value=lr_end,
            transition_steps=n_update_steps,
        )
    elif lr_schedule_type == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=lr_init,
            transition_steps=n_update_steps,
            decay_rate=lr_decay,
            end_value=lr_end,
        )
    else:
        raise ValueError(
            f"Unsupported learning rate schedule, expected 'constant', 'linear' or "
            f"'exponential'; received {lr_schedule_type}"
        )
    return optimizer(learning_rate=lr_schedule)


def get_activation_function(activation_name: str) -> Callable[[jax.Array], jax.Array]:
    """Returns an activation function based on the configuration."""
    activation_name = activation_name.lower()
    if activation_name == "relu":
        return nnx.relu
    elif activation_name == "swish":
        return nnx.swish
    elif activation_name == "tanh":
        return jax.nn.tanh
    elif activation_name == "gelu":
        return nnx.gelu
    else:
        raise ValueError(
            f"Unsupported activation function, expected 'relu', 'swish', 'tanh' or "
            f"'gelu'; received: {activation_name}"
        )


class NNTrainingState(struct.PyTreeNode):
    """Training state container for a Neural Network that can be passed through
    JAX transformations.

    Attributes:
        model_def: The static graph definition of the neural network.
        model_state: The dynamic state of the model, including its parameters.
        optimizer: The ``optax`` optimizer used for gradient updates.
        optimizer_state: The current state of the optimizer.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState

    def apply_gradients(self, grads: nnx.State) -> NNTrainingState:
        """Applies gradients to the model parameters.

        Args:
            grads: The gradients to be applied.

        Returns:
            A new ``NNTrainingState`` with updated model and optimizer states.
        """
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        model_state = optax.apply_updates(self.model_state, updates)
        return self.replace(
            model_state=model_state,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
    ) -> NNTrainingState:
        """Creates a new ``NNTrainingState`` instance.

        Args:
            model_def: The static graph definition of the neural network.
            model_state: The initial state of the model.
            optimizer: The ``optax`` optimizer to use.

        Returns:
            A new ``NNTrainingState`` instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


class NNTrainingStateSoftTarget(struct.PyTreeNode):
    """Training state with a soft-updating target network that can be passed through
    JAX transformations.

    Attributes:
        model_def: The static graph definition of the neural network.
        model_state: The dynamic state of the model, including its parameters.
        target_model_state: The state of the target network.
        optimizer: The ``optax`` optimizer used for gradient updates.
        optimizer_state: The current state of the optimizer.
        tau: The interpolation factor for the soft update.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    target_model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState
    tau: float = struct.field(pytree_node=False)

    def apply_gradients(self, grads: nnx.State) -> NNTrainingStateSoftTarget:
        """Applies gradients and performs a soft update on the target network.

        Args:
            grads: The gradients to be applied to the main model.

        Returns:
            A new ``NNTrainingStateSoftTarget`` with updated model, target
            model, and optimizer states.
        """
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        model_state = optax.apply_updates(self.model_state, updates)

        target_model_state = jax.tree.map(
            lambda tp, p: self.tau * p + (1 - self.tau) * tp,
            self.target_model_state,
            model_state,
        )
        return self.replace(
            model_state=model_state,
            optimizer_state=optimizer_state,
            target_model_state=target_model_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        optimizer: optax.GradientTransformation,
        tau: float,
    ) -> NNTrainingStateSoftTarget:
        """Creates a new ``NNTrainingStateSoftTarget`` instance.

        Note:
            It is important to explicitly copy the model parameters for the target
            network to avoid issues when donating the training state to JAX
            transformations where a referenced model state causes an error due to
            the donation attempting to donate the same object twice.

        Args:
            model_def: The static graph definition of the neural network.
            model_state: The initial state of the model.
            optimizer: The ``optax`` optimizer to use.
            tau: The interpolation factor for the soft update.

        Returns:
            A new ``NNTrainingStateSoftTarget`` instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            target_model_state=jax.tree.map(jnp.copy, model_state),
            tau=tau,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


@nnx.jit(static_argnums=(2,))
def soft_update_target_model(
    model: nnx.Module,
    model_target: nnx.Module,
    tau: float,
) -> nnx.Module:
    """Performs a soft update on the parameters of a target model.

    Args:
        model: The source model (e.g., the online network).
        model_target: The target model to be updated.
        tau: The interpolation factor for the soft update.

    Returns:
        A new target model with updated parameters.
    """
    model_params = nnx.state(model)
    graph_def, target_params, batch_stats = nnx.split(  # type: ignore[misc]
        model_target, nnx.Param, nnx.BatchStat
    )
    updated_params = jax.tree.map(
        lambda tp, p: tau * p + (1 - tau) * tp,
        target_params,
        model_params,
    )
    return nnx.merge(graph_def, updated_params, batch_stats)


def construct_mlp(
    input_features: int,
    hidden_layers_sizes: tuple[int, ...],
    use_layernorm: bool,
    kernel_initializer: nnx.Initializer,
    activation: Callable[[jax.Array], jax.Array],
    rngs: nnx.Rngs,
) -> nnx.Sequential:
    """Builds a feedforward MLP with optional layer normalization.

    Constructs a sequence of ``Linear -> [LayerNorm] -> Activation`` blocks.
    Allows to call any type of head on the output of the MLP.

    Args:
        input_features: Dimensionality of the input tensor.
        hidden_layers_sizes: Sizes of each hidden layer in order.
        use_layernorm: If ``True``, insert ``LayerNorm`` after each linear layer.
        kernel_initializer: Initializer for the linear layer weights.
        activation: Activation function applied after each (optionally normalized)
            linear layer.
        rngs: Flax NNX random number generators.

    Returns:
        The MLP in the form of an ``nnx.Sequential`` module.
    """
    layers: list[Callable[..., Any]] = []
    for i, hidden_size in enumerate((input_features,) + hidden_layers_sizes[:-1]):
        linear_layer = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_layers_sizes[i],
            kernel_init=kernel_initializer,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        layers.append(linear_layer)
        if use_layernorm:
            layernorm_layer = nnx.LayerNorm(
                num_features=hidden_layers_sizes[i],
                rngs=rngs,
            )
            layers.append(layernorm_layer)
        layers.append(activation)
    return nnx.Sequential(*layers)
