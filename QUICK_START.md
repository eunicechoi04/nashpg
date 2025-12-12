# Quick Start Guide: EMA Experiments for Kuhn Poker

## Overview

This guide shows how to run the complete EMA (Exponential Moving Average) experiments for Kuhn Poker, comparing baseline Nash PG against three different EMA tau values.

## Prerequisites

✅ CUDA-enabled JAX installed (already configured for T4 GPU)
✅ All dependencies installed via `uv sync`

## Quick Run - Complete Pipeline

### Option 1: Run Everything Sequentially

```bash
# Step 1: Train all 4 configurations (baseline + 3 EMA variants)
./scripts/run_all_kuhn_ema_experiments.sh

# Step 2: Compute exploitability for all trained models
./scripts/compute_all_exploits_ema.sh
```

### Option 2: Run Individual Experiments

**Training:**
```bash
./scripts/train_kuhn_baseline.sh       # No EMA (baseline)
./scripts/train_kuhn_ema_0001.sh       # EMA tau=0.001
./scripts/train_kuhn_ema_0005.sh       # EMA tau=0.005
./scripts/train_kuhn_ema_002.sh        # EMA tau=0.02
```

**Exploitability Computation:**
```bash
./scripts/compute_exploit_baseline.sh       # No EMA (baseline)
./scripts/compute_exploit_ema_0001.sh       # EMA tau=0.001
./scripts/compute_exploit_ema_0005.sh       # EMA tau=0.005
./scripts/compute_exploit_ema_002.sh        # EMA tau=0.02
```

## What Gets Created

### During Training:
```
checkpoints/kuhn_poker/nash_pg/
├── baseline_no_ema/checkpoint_*
├── ema_tau_0001/checkpoint_*
├── ema_tau_0005/checkpoint_*
└── ema_tau_002/checkpoint_*

logs/kuhn_poker/nash_pg/
├── baseline_no_ema.json
├── ema_tau_0001.json
├── ema_tau_0005.json
└── ema_tau_002.json

runs/kuhn_poker/nash_pg/
├── baseline_no_ema/
├── ema_tau_0001/
├── ema_tau_0005/
└── ema_tau_002/
```

### After Exploitability Computation:
Each JSON log file will have an additional `eval` section containing exploitability measurements.

## View Results

```bash
# View exploitability in JSON logs
cat logs/kuhn_poker/nash_pg/baseline_no_ema.json | jq '.eval'
cat logs/kuhn_poker/nash_pg/ema_tau_0001.json | jq '.eval'
cat logs/kuhn_poker/nash_pg/ema_tau_0005.json | jq '.eval'
cat logs/kuhn_poker/nash_pg/ema_tau_002.json | jq '.eval'

# Launch TensorBoard to visualize training
tensorboard --logdir=runs/kuhn_poker/nash_pg
# Then open http://localhost:6006 in your browser
```

## Configuration Details

| Experiment | EMA Enabled | Tau Value | Description |
|------------|-------------|-----------|-------------|
| Baseline | No | N/A | Standard periodic cloning (every 1000 steps) |
| EMA 0.001 | Yes | 0.001 | Very slow, conservative updates |
| EMA 0.005 | Yes | 0.005 | Medium update rate |
| EMA 0.02 | Yes | 0.02 | Fast, aggressive updates |

All experiments use:
- **Inner updates**: 1000 (gradient steps per outer loop)
- **Outer updates**: 50 (total: 50,000 gradient steps)
- **Seed**: 100
- **Learning rate**: 3e-4
- **Magnetic coefficient**: 0.2

## Expected Runtime

- **Training per experiment**: ~10-30 minutes (depending on GPU)
- **Exploitability computation**: ~10-30 minutes per experiment
- **Total time for all 4 experiments**: ~2-4 hours

## Troubleshooting

### JAX not using GPU
```bash
# Verify GPU is detected
nvidia-smi

# Check JAX sees the GPU
uv run python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

### Missing log files
If exploitability computation fails with "file not found", make sure training completed first:
```bash
ls -lh logs/kuhn_poker/nash_pg/
```

### Check training progress
```bash
# Monitor training in real-time
tail -f logs/kuhn_poker/nash_pg/baseline_no_ema.json
```

## Next Steps

After running experiments:
1. Compare exploitability curves across different tau values
2. Analyze convergence speed and stability
3. Check if EMA provides better Nash equilibrium approximation
4. Look at the `mag_kl` metric to see reference network divergence

For more details, see `EMA_IMPLEMENTATION.md`.
