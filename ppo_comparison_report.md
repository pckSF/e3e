# PPO Implementation Comparison: SCS vs Brax

This report compares the SCS PPO implementation (`scs/ppo/`) against the Brax PPO
implementation (`brax/training/agents/ppo/`) as run via `train_jax_ppo.py`.

Each difference is categorized by its primary impact area:
1. **Convergence / Final Performance**
2. **Speed / Efficiency**

---

## 1. Convergence / Final Performance

### 1.1 CRITICAL — GAE Scans Over the Wrong Axis

**SCS** (`rl_computations.py` → `gae_from_td_residuals`):

```python
_, gae = jax.lax.scan(
    _get_gae_value,
    jnp.zeros_like(td_residuals[-1]),
    (td_residuals, terminals),
    reverse=True,
)
```

After `separate_trajectory_rollouts`, trajectories are reshaped to
`[n_batches * batch_size,  n_rollout_steps, ...]`.  After batch selection inside
`loss_fn`, the shapes entering GAE are:

| Array | Shape |
|---|---|
| `td_residuals` | `[batch_size, n_rollout_steps]` |
| `terminals` | `[batch_size, n_rollout_steps]` |

`jax.lax.scan` iterates over **axis 0** (`batch_size = 1024`).
The carry has shape `[n_rollout_steps]` (= 30).

This means the scan **accumulates GAE across different trajectory segments**
instead of across time steps within each segment.  The correct behavior (used by
Brax) is:

**Brax** (`losses.py` → `compute_gae`):

```python
# Put time first
data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
# ...
jax.lax.scan(
    compute_vs_minus_v_xs,
    (lambda_, acc),            # acc shape: [batch_size]
    (truncation_mask, deltas, termination),  # scan axis 0 = time
    length=int(truncation_mask.shape[0]),
    reverse=True,
)
```

Brax swaps axes so that **time is axis 0** (length 30) and the scan carry is
`[batch_size]` (1024).  Each scan step processes one timestep across all batch
elements in parallel.

**Effect:**  With `gae_lambda = 0.95`, each SCS batch element's advantage
estimate is contaminated by TD residuals from *unrelated* trajectory segments.
The one-step TD residual component is correct, but multi-step bootstrapping is
corrupted.  This significantly degrades advantage quality, slowing convergence
and reducing final performance.

**Fix:**  Transpose `td_residuals` and `terminals` to
`[n_rollout_steps, batch_size]` before scanning, and set the initial carry to
`jnp.zeros_like(td_residuals[:, -1])` (i.e. shape `[batch_size]`).  Or
equivalently, `jax.vmap` the single-sequence GAE over the batch axis.

---

### 1.2 No Truncation / Termination Distinction

**Brax** (`losses.py`):

```python
truncation = data.extras['state_extras']['truncation']
termination = (1 - data.discount) * (1 - truncation)
```

Brax explicitly separates *truncation* (episode ended due to time limit) from
*termination* (episode ended due to a true terminal condition).  When a truncation
occurs, the TD residual is masked out (`deltas *= truncation_mask`), but the GAE
carry is **not** reset to zero—meaning the value function can still bootstrap
through truncated boundaries.

**SCS** (`rollouts.py`):

```python
reset_mask = next_env_state.done
```

`done` is set to `True` for **both** truncation and termination.  In `calculate_gae`:

```python
td_residuals = rewards + gamma * next_values * (1.0 - terminals) - values
```

When `terminals = 1` at a truncated step, the bootstrap term
`gamma * next_values` is zeroed out.  This treats the truncated state as if the
episode truly ended (discounted return = 0 beyond this point), biasing value
estimates downward for episodes that didn't finish naturally.

**Effect:**  For environments with enforced episode length limits (which is
almost all MuJoCo Playground environments), the value function underestimates
the return of long-lived states.  This can significantly hurt performance on
tasks where the agent would benefit from understanding that the future continues
beyond the episode cutoff.

---

### 1.3 Effective Value Loss Coefficient Is 2× Larger

**Brax:**

```python
v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient  # 0.5 × 0.5 = 0.25
total_loss = policy_loss + v_loss + entropy_loss
```

**SCS:**

```python
value_loss = jnp.mean((returns - values) ** 2)        # no 0.5 factor
loss = -(ppo_value - value_loss_coefficient * value_loss + entropy * entropy_coefficient)
# effective value term = value_loss_coefficient × mean(error²)  = 0.5 × mean(error²)
```

The missing `0.5` factor doubles the effective value-loss gradient.

| | Effective value loss multiplier | Value loss gradient scale |
|---|---|---|
| **Brax** | `0.5 × vf_coefficient = 0.25` | `0.5 × vf_coefficient × 2 × v_error = 0.5 × v_error` |
| **SCS** | `value_loss_coefficient = 0.5` | `value_loss_coefficient × 2 × v_error = 1.0 × v_error` |

**Effect:**  The value function gets 2× the gradient relative to the policy
loss.  This can make value learning faster but also increases the risk of
overfitting the value function to noisy targets, potentially destabilizing
training.  This imbalance shifts the effective loss weighting away from the
standard PPO formulation.

---

### 1.4 Policy Standard Deviation Parameterization

**Brax** (`distribution.py` — `NormalTanhDistribution`, the default):

```python
scale = (jax.nn.softplus(raw_scale) + self._min_std) * self._var_scale
# min_std = 0.001, var_scale = 1.0
```

The std is guaranteed to be ≥ `min_std`.  The `softplus` function provides
smooth gradients near zero.

**SCS** (`models.py` / `rollouts.py`):

```python
self.policy_log_std = nnx.Linear(...)   # outputs unconstrained log_std
a_std = jnp.exp(a_log_std)              # std = exp(log_std), range (0, ∞)
```

There is no lower bound on std.  As `log_std → -∞`, `std → 0`, which can cause:
- Premature convergence to a near-deterministic policy
- Numerical instability in log-probability computation
- Very large importance ratios when recomputing log-probs with a changed mean

**Effect:**  Without a minimum std, the policy can collapse to near-deterministic
behavior early in training, reducing exploration and potentially converging to
poor local optima.  Brax's `softplus + min_std` is a safer parameterization.

---

### 1.5 Entropy Computation (No Tanh Log-Det-Jacobian Correction)

**Brax** (`distribution.py`):

```python
def entropy(self, parameters, seed):
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))
    # ...
```

The entropy accounts for the tanh squashing.  Since tanh compresses the tails,
the true entropy of the post-tanh action distribution is **lower** than the
raw Gaussian entropy.  The correction term
$\log |\det J_{\tanh}(x)| = 2(\log 2 - x - \text{softplus}(-2x))$
is negative for most $x$, reducing the entropy estimate.

**SCS** (`agent.py`):

```python
entropy = jnp.sum(a_log_stds + 0.5 * (jnp.log(2 * jnp.pi) + 1), axis=-1).mean()
```

This is the exact differential entropy of the *raw Gaussian*, ignoring the tanh
transform entirely.

**Effect on PPO:**  The importance ratio $\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}$
is **not affected** (the tanh Jacobian cancels in the ratio since both
numerator and denominator are evaluated at the same raw action).  However, the
entropy bonus is inflated in SCS relative to the true post-tanh entropy.  With the
same `entropy_coefficient`, SCS applies a *stronger* exploration incentive than
Brax.  This may delay convergence to high-reward deterministic behaviors, or it
may help avoid premature convergence—the net effect is task-dependent.

---

### 1.6 Weight Initialization

**Brax** (`networks.py`):  Default kernel initializer is `lecun_uniform`.

**SCS** (`models.py`):  Default kernel initializer is `orthogonal`.

**Effect:**  Orthogonal initialization preserves gradient norms across layers,
which can improve training stability in deep networks.  `lecun_uniform` is
variance-preserving under the assumption of linear activations.  Both are
reasonable choices; the difference may meaningfully affect early training
dynamics and convergence speed, especially in deeper value networks (5 layers).

---

### 1.7 Layer Normalization Differences

Both implementations support layer normalization, but they differ in **ordering**
and **default usage**.

**Brax** (`networks.py` — `MLP`):  `Linear → Activation → LayerNorm`
(post-activation norm).  **Disabled by default.**

```python
hidden = linen.Dense(hidden_size, ...)(hidden)
if i != len(self.layer_sizes) - 1 or self.activate_final:
    hidden = self.activation(hidden)
    if self.layer_norm:
        hidden = linen.LayerNorm()(hidden)
```

**SCS** (`nn_modules.py` — `construct_mlp`):  `Linear → LayerNorm → Activation`
(pre-activation norm).  **Enabled in the WalkerWalk config.**

```python
layers.append(Linear(...))
if use_layernorm:
    layers.append(LayerNorm(...))
layers.append(activation)
```

**Effect:**  Pre-activation normalization (SCS) is generally considered more
stable for training, as it normalizes inputs to the activation function.  The
fact that SCS enables it while Brax doesn't means SCS has additional per-layer
computation but potentially more stable gradient flow.  This also changes the
effective representation capacity of the network.

---

### 1.8 No Gradient Clipping

**Brax** (`train.py`):

```python
if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
```

Gradient clipping is available and used in some configs (e.g., vision PPO uses
`max_grad_norm=1.0`).

**SCS:**  No gradient clipping mechanism exists.  The optimizer is constructed
directly without any gradient transformation.

**Effect:**  Without gradient clipping, large gradient spikes (from outlier
transitions or early training instability) can cause parameter updates that
destabilize the policy.  The impact depends on the environment and
hyperparameters; stable environments with well-tuned learning rates may not
need it.

---

### 1.9 Environment Resets During Training (`num_resets_per_eval`) — UNSURE

**Brax:**  With `num_resets_per_eval = 10` (dm_control_suite default), all
environments are fully reset to new random initial states after each training
epoch.  This provides fresh starting conditions, increasing exploration
diversity.

**SCS:**  No full environment reset during training.  However,
`collect_trajectories` creates **fresh random initial states for each rollout**
via `env.get_initial_state(jax.random.split(keys[0], ...))`.  Episodes that
terminate during a rollout are reset to these fresh states.

**Question:**  Does SCS's per-rollout fresh initial state provide equivalent
exploration diversity to Brax's periodic full environment reset?  The answer
depends on how frequently episodes terminate naturally vs. being truncated by
the time limit.  If most episodes run to the time limit and rarely terminate
early, SCS environments would rarely use the fresh initial states, potentially
reducing state diversity compared to Brax's periodic full resets.

---

## 2. Speed / Efficiency

### 2.1 Redundant Value Function Forward Passes (~2× Value Network Cost)

**Brax** (`losses.py` — `compute_ppo_loss`):

```python
baseline = value_apply(normalizer_params, params.value, data.observation)  # [T, B]
terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)  # [B]
```

Values for timesteps $s_0, \ldots, s_{T-1}$ are computed once.
$V(s_{t+1})$ is obtained by slicing: `values_t_plus_1 = concat(values[1:], bootstrap_value)`.
Only the last next-observation needs a separate forward pass.

**Total value evaluations per minibatch:**  $B \times T + B = B \times (T + 1)$

**SCS** (`agent.py` — `loss_fn`):

```python
a_means, a_log_stds, values = model(batch.observations)      # [batch_size, T, obs_dim]
next_values = model.get_values(batch.next_observations)        # [batch_size, T, obs_dim]
```

Both `observations` ($s_0, \ldots, s_{T-1}$) and `next_observations`
($s_1, \ldots, s_T$) are passed through the value network independently.
The overlap ($s_1, \ldots, s_{T-1}$) is computed twice.

**Total value evaluations per minibatch:**  $2 \times B \times T$

With $T = 30$:

| | Value evals per minibatch |
|---|---|
| **Brax** | $31 \times B$ |
| **SCS** | $60 \times B$ |

**Effect:**  SCS performs ~1.94× the value-network forward pass computation per
gradient update.  Over 512 gradient updates per rollout, this is significant.
The fix is to compute values once for all observations and use index-shifting +
a single bootstrap evaluation, as Brax does.

Additionally, SCS calls `model(batch.observations)` which runs **both** the
policy and value MLPs for all observations, rather than calling them
independently.  This means even the policy MLP processes the observations,
although only the policy output is used in the loss alongside the value output.
This is equivalent to Brax's approach (both run on the same data) but worth
noting.

---

### 2.2 No Multi-Device Parallelism (pmap)

**Brax:**  Uses `jax.pmap` for multi-device training:

```python
training_epoch = jax.pmap(
    training_epoch,
    axis_name=_PMAP_AXIS_NAME,
    donate_argnums=(0, 1),
)
```

Gradients are synchronized with `jax.lax.pmean`.  Environment batches are split
across devices.

**SCS:**  Uses only `jax.jit`:

```python
@partial(jax.jit, static_argnums=(2, 3, 4), donate_argnums=(0, 1))
def training_epoch(...)
```

**Effect:**  On multi-GPU setups, Brax can distribute the workload across
devices for near-linear speedup. SCS is limited to a single device. On a single
GPU, there is no difference.

---

### 2.3 Flax Linen (Brax) vs Flax NNX (SCS)

**Brax** uses Flax Linen's functional API, where parameters are passed as
separate pytrees to a pure `apply` function.

**SCS** uses Flax NNX's object-oriented API, requiring `nnx.merge()` to
reconstruct the model from `(GraphDef, State)` on every `update_step` call
inside `jax.lax.scan`:

```python
def update_step(train_state, batch_indices, trajectories, config):
    model = nnx.merge(train_state.model_def, train_state.model_state)
    # ...
    grad_loss_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, loss_components), grads = grad_loss_fn(model, batch, config)
```

**Effect:**  Under JIT, the `nnx.merge` and `nnx.value_and_grad` calls compile
down to the same XLA operations as the Linen functional approach.  There should
be **no runtime overhead** after compilation.  However, the tracing/compilation
time might be slightly longer with NNX due to the additional abstractions.  This
is a one-time cost per JIT compilation and unlikely to matter in practice.

---

### 2.4 Observation Normalization Placement (When Enabled) — UNSURE

**Brax:**  Normalization is a preprocessing step inside the network's `apply`
function.  The normalizer parameters are updated once per training step (after
collecting data, before SGD).  All observations (including those collected in
the past) are re-normalized with the latest statistics during loss computation.

**SCS:**  Normalization happens inside `env_wrapper.step()`.  Observations are
normalized at collection time and the normalized values are stored in the
trajectory.  During loss computation, the already-normalized observations are
used without re-normalization.

**Effect:**  Brax's approach ensures consistency: all observations in a batch are
normalized with the same statistics.  SCS's approach means observations within a
single rollout are normalized with slightly different running statistics (updated
on each step).  In practice, the normalizer converges quickly and the difference
is minor.

**Additional concern (SCS):**  When an environment is conditionally reset during
a rollout, the reset observation comes from `get_initial_state`, which calls
`reset` but does **not** apply observation normalization.  This means the first
post-reset observation seen by the policy is **unnormalized**, while subsequent
observations are normalized.  This inconsistency could cause policy instability if
observation normalization is enabled.

**Note:**  The current WalkerWalk config has `normalize_observations: false`,
so this difference has no effect with current settings.

---

## Summary Table

| # | Difference | Area | Impact | Confidence |
|---|---|---|---|---|
| 1.1 | GAE scans over batch axis instead of time axis | Convergence | **Critical** — corrupts multi-step advantage estimates | High |
| 1.2 | No truncation vs termination distinction | Convergence | **High** — biases value estimates in time-limited episodes | High |
| 1.3 | Value loss coefficient effectively 2× | Convergence | **Medium** — shifts loss balance, may cause value overfitting | High |
| 1.4 | `exp(log_std)` vs `softplus + min_std` | Convergence | **Medium** — risk of std collapse to zero | High |
| 1.5 | Entropy ignores tanh Jacobian | Convergence | **Low–Medium** — inflated entropy bonus | High |
| 1.6 | Orthogonal vs lecun_uniform init | Convergence | **Low–Medium** — affects early training dynamics | High |
| 1.7 | Pre-activation vs post-activation LayerNorm (and enabled vs disabled) | Convergence | **Low–Medium** — changes effective representation | High |
| 1.8 | No gradient clipping option | Convergence | **Low** — may matter in unstable regimes | High |
| 1.9 | No periodic full environment reset | Convergence | **Low** — uncertain impact on exploration | Unsure |
| 2.1 | 2× redundant value network forward passes | Speed | **Medium** — ~2× value net FLOPS per update | High |
| 2.2 | No pmap multi-device support | Speed | **High on multi-GPU, None on single-GPU** | High |
| 2.3 | NNX merge/split vs Linen functional API | Speed | **None at runtime** — compiled away by XLA | Medium |
| 2.4 | Observation normalization at env vs network level | Speed/Correctness | **None currently** (normalization disabled) | Medium |
