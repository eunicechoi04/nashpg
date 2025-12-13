# Alpha Warmup Schedule Summary

## What is Alpha Warmup?

**Alpha (α)** is the magnetic regularization coefficient in Nash Policy Gradient that controls how much the policy is penalized for deviating from a reference policy. Instead of using a **fixed** α throughout training, warmup schedules **gradually decrease** α from a high initial value to a target value over the first few training iterations.

### Mathematical Formula

```
Loss = PPO_Loss + Entropy_Loss + α(t) × KL(policy || reference)
                                  ↑
                           Warmup schedule controls this
```

### Why Use Warmup?

| Benefit | Description |
|---------|-------------|
| **Early Stability** | High α keeps policy close to reference, preventing wild changes |
| **Better Exploration** | Gradual decrease allows controlled policy deviation |
| **Improved Convergence** | Smooth transition from conservative to aggressive updates |

## The 4 Schedule Types

### 1. Linear Schedule
Decreases α at a constant rate:

```python
α(t) = α_high - (α_high - α_target) × (t / T_warmup)
```

**Trajectory:** Steady, predictable decrease
**Use case:** Baseline warmup approach

### 2. Exponential (λ=0.3)
Moderate exponential decay:

```python
α(t) = α_target + (α_high - α_target) × exp(-0.3 × t)
```

**Trajectory:** Fast initial drop, then gradual
**Use case:** Quick stabilization with slow fine-tuning

### 3. Exponential (λ=0.5)
Fast exponential decay:

```python
α(t) = α_target + (α_high - α_target) × exp(-0.5 × t)
```

**Trajectory:** Very fast initial drop
**Use case:** Rapid transition to low regularization

### 4. Cosine Schedule
Smooth cosine annealing:

```python
α(t) = α_target + 0.5 × (α_high - α_target) × (1 + cos(π × t / T_warmup))
```

**Trajectory:** S-curve decrease
**Use case:** Smooth transitions (popular in deep learning)

## Configuration

All experiments use these shared parameters:

```yaml
alpha_high: 0.4              # Initial α (2× baseline)
alpha_target: 0.2            # Final α (baseline mag_coef)
T_warmup: 10                 # Warmup duration (outer loops)
num_inner_update: 1000       # Gradient steps per outer loop
num_outer_update: 35         # Total training iterations
```

**Total training:** 35,000 gradient steps
**Warmup period:** First 10,000 steps (t=0 to t=9)
**Post-warmup:** Remaining 25,000 steps (t=10 to t=34) at α=0.2

## Quick Start

### Run All 4 Experiments

```bash
./scripts/run_leduc_schedules_pipeline.sh
```

**What it does:**
1. Trains all 4 schedules in parallel (~2.1 min each)
2. Computes exploitability as each completes (~15 min each)
3. Shows live progress bars and timing
4. Total time: ~17 minutes

### Run Individual Experiments

**Training:**
```bash
./scripts/train_leduc_schedule_linear.sh       # Linear
./scripts/train_leduc_schedule_exp03.sh        # Exponential (λ=0.3)
./scripts/train_leduc_schedule_exp05.sh        # Exponential (λ=0.5)
./scripts/train_leduc_schedule_cosine.sh       # Cosine
```

**Exploitability:**
```bash
./scripts/compute_exploit_leduc_schedule_linear.sh
./scripts/compute_exploit_leduc_schedule_exp03.sh
./scripts/compute_exploit_leduc_schedule_exp05.sh
./scripts/compute_exploit_leduc_schedule_cosine.sh
```

## Output Files

After running the pipeline:

```
logs/leduc_poker/nash_pg/
├── schedule_linear.json           # Linear results + alpha_t trajectory
├── schedule_exp_lambda03.json     # Exp (λ=0.3) results + alpha_t
├── schedule_exp_lambda05.json     # Exp (λ=0.5) results + alpha_t
└── schedule_cosine.json           # Cosine results + alpha_t

checkpoints/leduc_poker/nash_pg/
├── schedule_linear/
├── schedule_exp_lambda03/
├── schedule_exp_lambda05/
└── schedule_cosine/

runs/leduc_poker/nash_pg/          # TensorBoard logs
├── schedule_linear/
├── schedule_exp_lambda03/
├── schedule_exp_lambda05/
└── schedule_cosine/
```

## Analyzing Results

### 1. Verify Schedule Trajectories

Check that α(t) follows expected mathematical formula:

```bash
# Extract alpha_t values
cat logs/leduc_poker/nash_pg/schedule_linear.json | \
  jq '.train[] | select(.alpha_t) | {step: .step, alpha_t: .alpha_t}'
```

**Expected pattern:**
- t=0: α ≈ 0.4 (all schedules start here)
- t=5: α varies by schedule type
- t=10: α ≈ 0.2 (all schedules converge)
- t>10: α = 0.2 (fixed)

### 2. Compare Final Exploitability

```bash
# View final exploitability for each schedule
for schedule in linear exp_lambda03 exp_lambda05 cosine; do
  echo "=== $schedule ==="
  cat logs/leduc_poker/nash_pg/schedule_${schedule}.json | jq '.eval[-1]'
done
```

Lower exploitability = better Nash equilibrium approximation

### 3. Visualize in TensorBoard

```bash
tensorboard --logdir=runs/leduc_poker/nash_pg
```

**Key plots to compare:**
- `alpha_t` - Should match mathematical formulas
- `exploitability` - Convergence over time
- `actor_loss` - Training stability
- `mag_kl` - Policy deviation from reference

## Research Questions

This pipeline helps answer:

### 1. Does warmup improve convergence?
Compare final exploitability:
- **Schedules** (with warmup): schedule_linear, schedule_exp03, schedule_exp05, schedule_cosine
- **Baseline** (no warmup): Run `./scripts/train_leduc_baseline.sh` with fixed α=0.2

### 2. Which schedule type is best?
Compare across 4 schedule types:
- Convergence speed (steps to reach target exploitability)
- Final performance (exploitability at t=35)
- Training stability (loss variance during warmup)

### 3. Does decay rate matter?
Compare exponential schedules:
- **Fast decay** (λ=0.5): Reaches α_target quickly
- **Slow decay** (λ=0.3): Gradual transition

### 4. Training stability during warmup?
Monitor loss variance:
- **Warmup period** (t < 10): High α should stabilize training
- **Post-warmup** (t ≥ 10): Fixed α, should maintain stability

## Implementation Details

### Where Alpha is Used

**Training loop** (`train/nash_pg.py`):
```python
# Compute current alpha_t based on schedule
learner_state.alpha_t = compute_alpha_t(
    t=outer_loop_iteration,
    schedule_type=config.algorithm.alpha_schedule_type,
    alpha_high=config.algorithm.alpha_high,
    alpha_target=config.algorithm.alpha_target,
    T_warmup=config.algorithm.T_warmup,
    lambda_exp=config.algorithm.lambda_exp
)

# Use alpha_t in loss computation
Loss = PPO_Loss + Entropy_Loss + learner_state.alpha_t × KL_divergence
```

### Schedule Computation

**Function** (`train/nash_pg.py:39-78`):
```python
def compute_alpha_t(t, schedule_type, alpha_high, alpha_target, T_warmup, lambda_exp):
    if schedule_type == "none":
        return alpha_target

    if t >= T_warmup:  # After warmup, use target
        return alpha_target

    # During warmup (t < T_warmup)
    if schedule_type == "linear":
        return alpha_high - (alpha_high - alpha_target) * (t / T_warmup)

    elif schedule_type == "exponential":
        return alpha_target + (alpha_high - alpha_target) * jnp.exp(-lambda_exp * t)

    elif schedule_type == "cosine":
        return alpha_target + 0.5 * (alpha_high - alpha_target) * \
               (1 + jnp.cos(jnp.pi * t / T_warmup))
```

### Logged Metrics

Each training run logs `alpha_t` at every outer loop iteration, allowing you to:
- Verify schedule is working correctly
- Plot α trajectory over time
- Correlate α changes with loss/exploitability

## Visual Comparison

```
α
0.4 ├─────────────────────────────────────────
    │ ╲  All start at 0.4
    │  ╲
    │   ╲╲╲   Linear:    ──────  (constant rate)
0.35│    ╲ ╲╲  Exp(0.3):  ······  (moderate decay)
    │     ╲  ╲ Exp(0.5):  ──────  (fast decay)
0.3 │      ╲  ╲╲Cosine:   ─·─·─  (smooth S-curve)
    │       ╲╲  ╲
    │         ╲╲ ╲╲
0.25│          ╲╲  ╲
    │            ╲╲╲╲
0.2 │─────────────╲╲╲╲──────────  All converge to 0.2
    └──────────────────────────────► t (outer loops)
    0    2    4    6    8    10
         └─── Warmup Period ───┘
```

## Comparison Table

| Schedule Type | Decay Rate | α at t=5 | α at t=10 | Best For |
|--------------|------------|----------|-----------|----------|
| Linear | Constant | 0.3 | 0.2 | Predictable, steady warmup |
| Exp (λ=0.3) | Moderate | ~0.24 | ~0.21 | Quick stabilization + slow tuning |
| Exp (λ=0.5) | Fast | ~0.22 | ~0.20 | Rapid transition |
| Cosine | S-curve | ~0.27 | 0.2 | Smooth, gradual transitions |

## Next Steps

1. **Run experiments:** `./scripts/run_leduc_schedules_pipeline.sh`
2. **Verify schedules:** Check `alpha_t` in JSON logs matches formulas
3. **Compare results:** Use TensorBoard to visualize all 4 schedules
4. **Analyze stability:** Check loss variance during warmup (t<10) vs post-warmup (t≥10)
5. **Test on baseline:** Compare with no-warmup baseline to assess benefit

## Key Takeaways

✅ **Warmup schedules** gradually reduce regularization over training
✅ **4 schedule types** test different decay patterns
✅ **10 outer loops** warmup period, then fixed α=0.2
✅ **35k total steps** with parallel GPU execution (~17 min)
✅ **Alpha tracking** logged at every iteration for analysis

For more details, see `SCHEDULE_EXPERIMENTS.md`.
