# Leduc Poker Parallel Pipeline

## Overview

A sophisticated pipeline that trains 4 Leduc Poker models in parallel and automatically computes exploitability as soon as each training completes, maximizing GPU utilization and minimizing total runtime.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL TRAINING                         │
│  (All 4 models train simultaneously on shared GPU)          │
├─────────────────────────────────────────────────────────────┤
│  Baseline ────────────► [DONE] ──► Exploit Baseline        │
│  EMA 0.001 ───────────► [DONE] ──► Exploit EMA 0.001       │
│  EMA 0.005 ────────────► [DONE] ──► Exploit EMA 0.005      │
│  EMA 0.02 ─────────────► [DONE] ──► Exploit EMA 0.02       │
└─────────────────────────────────────────────────────────────┘
     ↓ Training         ↓ Complete    ↓ Exploit starts immediately
```

### Key Features:
✅ **Parallel training**: All 4 models train simultaneously
✅ **Immediate exploitation**: Each model's exploitability computed as soon as training finishes
✅ **Parallel exploits**: Multiple exploitability computations run concurrently
✅ **Real-time monitoring**: Color-coded status updates
✅ **Automatic error handling**: Tracks success/failure of each stage

## Quick Start

### Run the Complete Pipeline

```bash
./scripts/run_leduc_pipeline.sh
```

This single command will:
1. Start all 4 training jobs in parallel
2. Monitor each training completion
3. Launch exploitability computation immediately after each training
4. Continue until all experiments complete
5. Provide final summary with success/failure status

### Expected Output

```
==========================================
Leduc Poker Parallel Training Pipeline
==========================================

Pipeline stages:
  1. Training (4 models in parallel)
  2. Exploitability (starts as each training completes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: Starting Training Jobs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[TRAIN] Starting: Baseline (no EMA)
[TRAIN] Baseline (no EMA) → PID 12345
[TRAIN] Starting: EMA tau=0.001
[TRAIN] EMA tau=0.001 → PID 12346
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2: Monitoring & Launching Exploits
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ [TRAIN] Baseline (no EMA) completed successfully!
[EXPLOIT] Starting: Baseline (no EMA)
...
```

### Status Updates

The pipeline shows real-time status:
- ⚙ = Running
- ✓ = Completed
- ⏳ = Pending
- ✗ = Failed

## Individual Scripts

If you need to run experiments individually:

### Training Only
```bash
./scripts/train_leduc_baseline.sh       # No EMA
./scripts/train_leduc_ema_0001.sh       # tau=0.001
./scripts/train_leduc_ema_0005.sh       # tau=0.005
./scripts/train_leduc_ema_002.sh        # tau=0.02
```

### Exploitability Only (after training)
```bash
./scripts/compute_exploit_leduc_baseline.sh
./scripts/compute_exploit_leduc_ema_0001.sh
./scripts/compute_exploit_leduc_ema_0005.sh
./scripts/compute_exploit_leduc_ema_002.sh
```

## Output Locations

After running the pipeline:

```
logs/leduc_poker/nash_pg/
├── baseline_no_ema.json          # Contains training metrics + exploitability
├── ema_tau_0001.json
├── ema_tau_0005.json
└── ema_tau_002.json

checkpoints/leduc_poker/nash_pg/
├── baseline_no_ema/checkpoint_*
├── ema_tau_0001/checkpoint_*
├── ema_tau_0005/checkpoint_*
└── ema_tau_002/checkpoint_*

runs/leduc_poker/nash_pg/           # TensorBoard logs
├── baseline_no_ema/
├── ema_tau_0001/
├── ema_tau_0005/
└── ema_tau_002/

logs/                               # Pipeline execution logs
├── train_leduc_baseline.log
├── train_leduc_ema_0001.log
├── exploit_leduc_baseline.log
└── exploit_leduc_ema_0001.log
```

## Performance Characteristics

### GPU Memory Usage
- **Leduc Poker per model**: ~1-2 GB
- **4 parallel trainings**: ~4-8 GB total
- **Training + Exploit**: ~6-10 GB (overlap period)
- **T4 GPU capacity**: 15 GB ✓ (plenty of headroom)

### Timeline Estimate

**Sequential approach** (not recommended):
```
Train 1 → Exploit 1 → Train 2 → Exploit 2 → Train 3 → Exploit 3 → Train 4 → Exploit 4
|─ 3min ─|─ 15min ─|─ 3min ─|─ 15min ─|─ 3min ─|─ 15min ─|─ 3min ─|─ 15min ─|
Total: ~72 minutes
```

**Parallel pipeline approach** (this script):
```
Train 1 ────────────────► Exploit 1 ───────────────────────►
Train 2 ────────────────► Exploit 2 ───────────────────────►
Train 3 ────────────────► Exploit 3 ───────────────────────►
Train 4 ────────────────► Exploit 4 ───────────────────────►
|─── 3 min ───|─────────── 15 min ──────────────|
Total: ~18 minutes
```

**Speedup: 4× faster!** 🚀

## Monitoring & Debugging

### View Training Progress
```bash
# Follow training logs in real-time
tail -f logs/train_leduc_baseline.log
tail -f logs/train_leduc_ema_0001.log
```

### View Exploit Progress
```bash
# Follow exploit logs in real-time
tail -f logs/exploit_leduc_baseline.log
```

### Check GPU Usage
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### View Results
```bash
# View exploitability data
cat logs/leduc_poker/nash_pg/baseline_no_ema.json | jq '.eval'

# Launch TensorBoard
tensorboard --logdir=runs/leduc_poker/nash_pg
```

## Troubleshooting

### Pipeline fails to start
- Check that all scripts are executable: `ls -lh scripts/*leduc*`
- Ensure no conflicting processes: `nvidia-smi`

### Training completes but exploit doesn't start
- Check log file exists: `ls -lh logs/leduc_poker/nash_pg/*.json`
- Check for errors: `cat logs/train_leduc_*.log | grep -i error`

### Out of GPU memory
- Leduc is larger than Kuhn, may use more memory
- Monitor with: `nvidia-smi`
- Consider running fewer experiments in parallel

### Clean up previous runs
```bash
# Remove old Leduc checkpoints and logs
rm -rf checkpoints/leduc_poker/nash_pg/*
rm -rf logs/leduc_poker/nash_pg/*.json
rm -rf runs/leduc_poker/nash_pg/*
rm -f logs/train_leduc_*.log logs/exploit_leduc_*.log
```

## Comparison with Kuhn Poker

| Aspect | Kuhn Poker | Leduc Poker |
|--------|------------|-------------|
| **State space** | Small (3 cards) | Medium (6 cards, 2 rounds) |
| **Action space** | 2 (Bet/Pass) | 3-4 (Fold/Call/Raise) |
| **Training time** | ~2 min | ~3-4 min |
| **Exploit time** | ~13 min | ~15-20 min |
| **Memory usage** | ~500 MB - 1 GB | ~1-2 GB |
| **Complexity** | Simple | Moderate |

## Configuration

All experiments use the same hyperparameters:
- **Inner updates**: 1000 (gradient steps per outer loop)
- **Outer updates**: 50 (total: 50,000 gradient steps)
- **Seed**: 100 (for reproducibility)
- **EMA tau values**: 0.001, 0.005, 0.02
- **Baseline**: Periodic cloning (no EMA)

## Next Steps

After pipeline completes:
1. Compare exploitability curves across different tau values
2. Analyze convergence speed and stability
3. Check if EMA provides better Nash equilibrium for Leduc
4. Compare results with Kuhn Poker experiments

For more details on EMA implementation, see `EMA_IMPLEMENTATION.md`.
