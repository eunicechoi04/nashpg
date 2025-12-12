# EMA (Exponential Moving Average) Implementation

## Overview

This implementation adds support for using Exponential Moving Average (EMA) to update the reference network (magnetic agent) in the Nash Policy Gradient algorithm, as an alternative to the standard periodic cloning approach.

## Mechanism

### Standard Approach (Baseline)
- The reference network (`mag_agent`) is updated by cloning the current agent every outer loop iteration (default: every 1000 steps)
- Formula: `mag_agent = clone(agent)` (complete parameter copy)

### EMA Approach
- The reference network is updated gradually after every gradient step (or every N steps)
- Formula: `ref_params = (1 - tau) * ref_params + tau * current_params`
- Parameters:
  - `tau`: Controls the update rate (smaller = slower update, more stable reference)
  - `ema_update_freq`: How often to apply the EMA update (default: 1 = every step)

## Implementation Details

### Configuration Parameters

Added to `conf/algorithm/nash_pg.yaml`:

```yaml
use_ema: false              # Enable/disable EMA updates
ema_tau: 0.001             # EMA coefficient (tau value)
ema_update_freq: 1         # Update frequency (1 = every step)
```

### Code Changes

**File: `train/nash_pg.py`**

1. **New function `apply_ema_update()`**:
   - Applies EMA update to the reference network
   - Returns a new agent with updated parameters
   - Uses JAX tree operations for efficient parameter updates

2. **Modified `LearnerState` dataclass**:
   - Added `ema_step_count` field to track update steps

3. **Modified `single_training_step()` function**:
   - After PPO update, applies EMA update if enabled
   - Increments step counter and checks update frequency

4. **Modified outer training loop**:
   - Only performs cloning when EMA is disabled
   - EMA updates happen during gradient steps instead

## Training Scripts

Four training scripts have been created in `scripts/`:

### 1. Baseline (No EMA)
**File**: `train_kuhn_baseline.sh`
```bash
./scripts/train_kuhn_baseline.sh
```
- Uses standard periodic cloning (every 1000 steps)
- Run name: `kuhn_poker/nash_pg/baseline_no_ema`

### 2. EMA with tau=0.001
**File**: `train_kuhn_ema_0001.sh`
```bash
./scripts/train_kuhn_ema_0001.sh
```
- Very slow EMA update (conservative)
- Run name: `kuhn_poker/nash_pg/ema_tau_0001`

### 3. EMA with tau=0.005
**File**: `train_kuhn_ema_0005.sh`
```bash
./scripts/train_kuhn_ema_0005.sh
```
- Medium EMA update rate
- Run name: `kuhn_poker/nash_pg/ema_tau_0005`

### 4. EMA with tau=0.02
**File**: `train_kuhn_ema_002.sh`
```bash
./scripts/train_kuhn_ema_002.sh
```
- Faster EMA update (more aggressive)
- Run name: `kuhn_poker/nash_pg/ema_tau_002`

## Running All Experiments

### Training

A master script is provided to run all training experiments sequentially:

```bash
./scripts/run_all_kuhn_ema_experiments.sh
```

This will run all four configurations (baseline + 3 EMA variants) one after another.

### Computing Exploitability

After training completes, you **must** compute exploitability for each trained model to properly evaluate performance. Individual exploitability computation scripts are provided for each experiment:

```bash
./scripts/compute_exploit_baseline.sh      # Baseline
./scripts/compute_exploit_ema_0001.sh      # tau=0.001
./scripts/compute_exploit_ema_0005.sh      # tau=0.005
./scripts/compute_exploit_ema_002.sh       # tau=0.02
```

Or run all exploitability computations at once:

```bash
./scripts/compute_all_exploits_ema.sh
```

**Important Notes:**
- Exploitability computation uses the **RL solver method** (trains an exploiting agent)
- This is the correct method for Nash PG experiments
- Each exploitability computation will take significant time as it trains an RL agent
- The exploitability values will be injected into the respective JSON log files
- Results will be saved in the `eval` section of each log file

## Expected Behavior

### Tau Values Comparison

- **tau = 0.001**: Very conservative updates
  - Reference network changes very slowly
  - More stable but less responsive to agent improvements
  - After 1000 steps: ~63.2% of parameters are from new agent

- **tau = 0.005**: Moderate updates
  - Balanced between stability and responsiveness
  - After 1000 steps: ~99.3% of parameters are from new agent

- **tau = 0.02**: Aggressive updates
  - Reference network tracks agent closely
  - More responsive but potentially less stable
  - After 1000 steps: ~100% of parameters are from new agent

### Training Configuration

All experiments use:
- 1000 inner updates per outer loop
- 50 outer loops (50,000 total gradient steps)
- Same seed (100) for reproducibility
- Same hyperparameters (lr=3e-4, mag_coef=0.2, etc.)

## Complete Workflow

### Step 1: Train All Models
```bash
./scripts/run_all_kuhn_ema_experiments.sh
```

This will create:
- **Checkpoints**: `checkpoints/kuhn_poker/nash_pg/[run_name]/checkpoint_*`
- **TensorBoard logs**: `runs/kuhn_poker/nash_pg/[run_name]/*`
- **JSON logs**: `logs/kuhn_poker/nash_pg/[run_name].json`

Where `[run_name]` is one of:
- `baseline_no_ema`
- `ema_tau_0001`
- `ema_tau_0005`
- `ema_tau_002`

### Step 2: Compute Exploitability
```bash
./scripts/compute_all_exploits_ema.sh
```

This will:
- Load each checkpoint from the training runs
- Train an exploiting agent against each checkpoint (using RL)
- Compute exploitability values
- Inject results into the JSON log files

### Step 3: Analyze Results

After both steps complete, each JSON log file will contain:
- **Training metrics**: `train` section (actor_loss, ppo_loss, entropy, mag_kl, etc.)
- **Rollout metrics**: `rollout` section (episode length, returns)
- **Exploitability**: `eval` section (exploitability values over training)

Access results via:
```bash
# View JSON logs
cat logs/kuhn_poker/nash_pg/baseline_no_ema.json
cat logs/kuhn_poker/nash_pg/ema_tau_0001.json
cat logs/kuhn_poker/nash_pg/ema_tau_0005.json
cat logs/kuhn_poker/nash_pg/ema_tau_002.json

# View TensorBoard
tensorboard --logdir=runs/kuhn_poker/nash_pg
```

## Verification

To verify the implementation is working:

1. Check that training runs without errors
2. Monitor the `mag_kl` metric in logs (should be different between EMA and baseline)
3. For EMA runs, the reference network should update continuously
4. For baseline, reference network updates only at outer loop boundaries
5. Check that exploitability values are present in the JSON logs after computation

## Technical Notes

- EMA updates use JAX tree operations for efficiency
- Compatible with JIT compilation via `nnx.jit`
- Step counter ensures correct update frequency
- Works with all agent types (KuhnPoker, MLP, etc.)

## Future Extensions

Possible enhancements:
- Adaptive tau (change tau during training)
- Different update frequencies for exploration
- Per-layer tau values
- Momentum-based updates
