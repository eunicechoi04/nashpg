# Alpha Warmup Schedule Experiments for Leduc Poker

## Overview

This pipeline tests **4 different alpha (regularization) warmup schedules** for Nash PG on Leduc Poker. The warmup schedule gradually adjusts the magnetic regularization coefficient (α) during training, potentially leading to better convergence.

## What is Alpha Warmup?

Instead of using a **fixed** magnetic coefficient (e.g., α = 0.2), warmup schedules **gradually decrease** α from a high initial value to a target value over the first few outer loops:

```
α(t) starts high (0.4) → smoothly decreases → reaches target (0.2)
```

This can help:
- ✅ **Stabilize early training** (high regularization prevents wild policy changes)
- ✅ **Improve exploration** (initially stays closer to reference policy)
- ✅ **Better convergence** (gradually allows more policy deviation)

## The 4 Schedule Types

### 1. **Linear Schedule**
Decreases α linearly over T_warmup iterations:

```
α(t) = α_high - (α_high - α_target) × (t / T_warmup)

Example: 0.4 → 0.35 → 0.3 → 0.25 → 0.2 (over 10 outer loops)
```

- **Properties**: Constant rate of decrease
- **Best for**: Predictable, steady warmup

### 2. **Exponential Schedule (λ=0.3)**
Exponentially decays from α_high to α_target:

```
α(t) = α_target + (α_high - α_target) × exp(-λ × t)

With λ=0.3: Moderate decay rate
```

- **Properties**: Fast initial decrease, then slows down
- **Best for**: Quick initial stabilization with gradual fine-tuning

### 3. **Exponential Schedule (λ=0.5)**
Same formula, but faster decay:

```
α(t) = α_target + (α_high - α_target) × exp(-λ × t)

With λ=0.5: Faster decay rate
```

- **Properties**: Very fast initial decrease
- **Best for**: Rapid transition to low regularization

### 4. **Cosine Schedule**
Cosine annealing from α_high to α_target:

```
α(t) = α_target + 0.5 × (α_high - α_target) × (1 + cos(π × t / T_warmup))
```

- **Properties**: Smooth S-curve decrease
- **Best for**: Gradual, smooth transitions (popular in deep learning)

## Visual Comparison

```
α
0.4 ├─────────────────────────────────────────
    │ ╲ All schedules start here
    │  ╲ ╲╲
0.35│   ╲  ╲╲╲    Linear: ─────
    │    ╲    ╲╲   Exp(0.3): ····
0.3 │     ╲     ╲  Exp(0.5): ----
    │      ╲    ╲╲ Cosine: ─·─·─
    │       ╲  ╲  ╲╲
0.25│        ╲╲    ╲
    │          ╲╲╲  ╲
0.2 │─────────────╲╲╲╲────────────── All converge here
    └──────────────────────────────► t (outer loops)
    0    2    4    6    8    10
         (Warmup Period)
```

## Configuration Details

All experiments use:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha_high` | 0.4 | Initial regularization (2× baseline) |
| `alpha_target` | 0.2 | Final regularization (baseline mag_coef) |
| `T_warmup` | 10 | Warmup duration (outer loop iterations) |
| `num_inner_update` | 1000 | Gradient steps per outer loop |
| `num_outer_update` | 50 | Total outer loops (warmup ends at t=10) |
| **Total training** | 50,000 steps | Same as baseline experiments |

### Lambda Values for Exponential Schedules

| Lambda | Decay Rate | α at t=10 | Description |
|--------|------------|-----------|-------------|
| 0.3 | Moderate | ~0.21 | Almost at target |
| 0.5 | Fast | ~0.203 | Reaches target quickly |

## Quick Start

### Run All 4 Experiments in Parallel

```bash
./scripts/run_leduc_schedules_pipeline.sh
```

This will:
1. ⚡ Train all 4 schedules in parallel (~3 min)
2. 🔍 Compute exploitability for each as training completes (~15 min each)
3. 📊 Show real-time progress with timing
4. ✅ Generate results for all 4 schedules

**Expected total time:** ~18 minutes (with parallel overlapping)

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

## Output Locations

After running the pipeline:

```
logs/leduc_poker/nash_pg/
├── schedule_linear.json           # Linear schedule results
├── schedule_exp_lambda03.json     # Exponential (λ=0.3) results
├── schedule_exp_lambda05.json     # Exponential (λ=0.5) results
└── schedule_cosine.json           # Cosine schedule results

checkpoints/leduc_poker/nash_pg/
├── schedule_linear/checkpoint_*
├── schedule_exp_lambda03/checkpoint_*
├── schedule_exp_lambda05/checkpoint_*
└── schedule_cosine/checkpoint_*

runs/leduc_poker/nash_pg/          # TensorBoard logs
├── schedule_linear/
├── schedule_exp_lambda03/
├── schedule_exp_lambda05/
└── schedule_cosine/
```

## Expected Pipeline Output

```bash
==========================================
Leduc Poker Schedule Experiments Pipeline
==========================================

Testing 4 different alpha warmup schedules:
  1. Linear
  2. Exponential (λ=0.3)
  3. Exponential (λ=0.5)
  4. Cosine

Pipeline stages:
  1. Training (4 schedules in parallel) - ETA: 3m 0s each
  2. Exploitability (starts as each training completes) - ETA: 15m 0s each

Expected total time: ~18 minutes (with overlapping exploits)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: Starting Training Jobs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[TRAIN] Starting: Linear Schedule (ETA: 3m 0s)
[TRAIN] Linear Schedule → PID 12345
[TRAIN] Starting: Exponential (λ=0.3) (ETA: 3m 0s)
...

========================================
Current Status:
========================================
Linear Schedule
  Train: ⚙ (1m 23s / ~3m 0s) | Exploit: ⏳ (waiting)
Exponential (λ=0.3)
  Train: ⚙ (1m 20s / ~3m 0s) | Exploit: ⏳ (waiting)
...
```

## Analyzing Results

### View Alpha Schedule Trajectory

Each JSON log file contains an `alpha_t` metric tracking the regularization coefficient over time:

```bash
# Extract alpha_t values
cat logs/leduc_poker/nash_pg/schedule_linear.json | jq '.train[] | select(.alpha_t) | {step: .step, alpha_t: .alpha_t}'
```

### Compare Exploitability

```bash
# View final exploitability for each schedule
cat logs/leduc_poker/nash_pg/schedule_linear.json | jq '.eval[-1]'
cat logs/leduc_poker/nash_pg/schedule_exp_lambda03.json | jq '.eval[-1]'
cat logs/leduc_poker/nash_pg/schedule_exp_lambda05.json | jq '.eval[-1]'
cat logs/leduc_poker/nash_pg/schedule_cosine.json | jq '.eval[-1]'
```

### TensorBoard Visualization

```bash
tensorboard --logdir=runs/leduc_poker/nash_pg

# Then compare:
# - alpha_t curves (should match mathematical formulas)
# - Training loss convergence
# - Exploitability over time
```

## Performance & GPU Usage

### Parallel Execution (4 experiments)

| Resource | Per Job | 4 Parallel | T4 Capacity | Utilization |
|----------|---------|------------|-------------|-------------|
| GPU Memory | ~2 GB | ~8 GB | 15 GB | 53% |
| GPU Compute | ~20% | ~80% | 100% | 80% |
| CPU Cores | 2 | 8 | 8 | 100% |

**Result:** All 4 schedules train simultaneously with good GPU utilization and no slowdown.

### Timeline

```
Time:     0min          3min              18min
          ├─────────────┼─────────────────┤
Training: [All 4 run simultaneously]
Exploit:               [Overlapping as trainings finish]

Wall Clock: ~18 minutes
```

## Research Questions

This pipeline helps answer:

1. **Does warmup help convergence?**
   - Compare final exploitability: Schedule vs No Schedule (baseline)

2. **Which schedule is best?**
   - Linear vs Exponential vs Cosine
   - Compare convergence speed and final performance

3. **Does decay rate matter?**
   - Exponential λ=0.3 vs λ=0.5
   - Fast vs slow transitions

4. **Training stability?**
   - Monitor loss variance during warmup period (t < 10)
   - Compare with post-warmup period (t ≥ 10)

## Comparison with Baseline

To compare with standard Nash PG (no warmup schedule):

```bash
# Run baseline (from EMA experiments)
./scripts/train_leduc_baseline.sh
./scripts/compute_exploit_leduc_baseline.sh

# Then compare:
# - Baseline: Fixed α = 0.2 throughout
# - Schedules: α starts at 0.4, ends at 0.2
```

## Troubleshooting

### Check if schedules are working

```bash
# Verify alpha_t is changing in logs
tail -f logs/train_leduc_schedule_linear.log | grep -i alpha

# Should show decreasing values during first 10 outer loops
```

### View individual experiment logs

```bash
tail -f logs/train_leduc_schedule_linear.log
tail -f logs/exploit_leduc_schedule_linear.log
```

### Clean up previous runs

```bash
rm -rf checkpoints/leduc_poker/nash_pg/schedule_*
rm -rf logs/leduc_poker/nash_pg/schedule_*.json
rm -rf runs/leduc_poker/nash_pg/schedule_*
```

## Script Files

```
Training Scripts:
  train_leduc_schedule_linear.sh
  train_leduc_schedule_exp03.sh
  train_leduc_schedule_exp05.sh
  train_leduc_schedule_cosine.sh

Exploitability Scripts:
  compute_exploit_leduc_schedule_linear.sh
  compute_exploit_leduc_schedule_exp03.sh
  compute_exploit_leduc_schedule_exp05.sh
  compute_exploit_leduc_schedule_cosine.sh

Master Pipeline:
  run_leduc_schedules_pipeline.sh  ← Run this for everything
```

## Next Steps

After running experiments:

1. **Plot alpha_t trajectories** to verify schedules work correctly
2. **Compare exploitability curves** across all 4 schedules
3. **Analyze convergence speed** during and after warmup
4. **Check training stability** (loss variance in early vs late training)
5. **Compare with baseline** (no warmup) to assess benefit

Happy experimenting! 🚀
