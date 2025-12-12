# Parallel vs Sequential Pipeline Comparison

## Quick Answer

**Use parallel** (`run_leduc_pipeline.sh`) - it's 3× faster with no downsides on your T4 GPU.

## Detailed Comparison

### Parallel Pipeline (`run_leduc_pipeline.sh`)

**Timeline:**
```
Time:     0min          3min              18min
          ├─────────────┼─────────────────┤
Training: [All 3 run simultaneously]
Exploit:               [Overlapping as trainings finish]

Wall Clock: ~18 minutes
```

**Pros:**
- ✅ **3× faster** (18 min vs 54 min)
- ✅ **Better GPU utilization** (keeps GPU busy)
- ✅ **Modern approach** (leverages parallelism)
- ✅ **Same total compute** (no wasted resources)

**Cons:**
- ⚠️ Higher GPU memory usage (~6-9 GB peak)
- ⚠️ Harder to debug if issues occur
- ⚠️ More complex script logic

**When to use:**
- **Default choice** - Use this unless you have a reason not to
- When you want results quickly
- When GPU memory is available (T4 has 15GB - plenty of headroom)

### Sequential Pipeline (`run_leduc_sequential.sh`)

**Timeline:**
```
Time:     0   3   18  21  36  39  54min
          ├───┼───┼───┼───┼───┼───┤
Exp 1:    [T] [E]
Exp 2:            [T] [E]
Exp 3:                    [T] [E]

Wall Clock: ~54 minutes
```

**Pros:**
- ✅ **Lower GPU memory** (~2-3 GB at a time)
- ✅ **Easier to debug** (one thing at a time)
- ✅ **Simpler script** (easier to understand)
- ✅ **Better for logging** (no interleaved output)

**Cons:**
- ❌ **3× slower** (54 min vs 18 min)
- ❌ **Poor GPU utilization** (GPU mostly idle)
- ❌ **Wastes time** (unnecessary waiting)

**When to use:**
- Debugging issues (easier to track errors)
- Limited GPU memory (not an issue on T4)
- Want clean, separated logs
- Learning/understanding the pipeline

## Performance Details

### Why Parallel Doesn't Slow Down Individual Jobs

Your setup: **T4 GPU (15 GB) + 8 vCPU + 50 GB RAM**

| Resource | Leduc per job | 3 parallel | Available | Utilization |
|----------|---------------|------------|-----------|-------------|
| **GPU Memory** | ~2 GB | ~6 GB | 15 GB | 40% |
| **GPU Compute** | ~20% | ~60% | 100% | 60% |
| **CPU** | 2 cores | 6 cores | 8 cores | 75% |
| **RAM** | 2 GB | 6 GB | 50 GB | 12% |

**Result:** All resources are underutilized, even with parallel execution. No slowdown expected.

### Actual Runtime Comparison

Based on Leduc Poker characteristics:

| Metric | Parallel | Sequential | Difference |
|--------|----------|------------|------------|
| **Training time** | ~3 min (all 3 at once) | ~9 min total | Same per job |
| **Exploit time** | ~15 min (overlapping) | ~45 min total | Same per job |
| **Wall clock** | **~18 min** | **~54 min** | **3× faster** |
| **GPU-hours** | 0.3 GPU-hours | 0.9 GPU-hours | Same total compute |
| **Cost** | Lower (less time) | Higher (more time) | 3× more expensive |

### Resource Contention

**Will parallel jobs slow each other down?**

For Leduc Poker on T4: **No significant slowdown** (<5%)

Why:
- Small models (~10K parameters)
- GPU is underutilized with 1 job
- Memory bandwidth is not saturated
- JAX handles concurrent execution well

**If you had 10× larger models:**
- Then yes, parallel might slow down 10-20%
- But Leduc/Kuhn are too small for this to matter

## Usage

### Parallel (Recommended)
```bash
./scripts/run_leduc_pipeline.sh

# Expected output:
# Expected total time: ~18 minutes (with overlapping exploits)
# Progress tracking with live timers
```

### Sequential (Debug/Learning)
```bash
./scripts/run_leduc_sequential.sh

# Expected output:
# Expected total time: 54m 0s
# Experiment 1/3: Baseline (no EMA)
# [TRAIN] Starting: Baseline (no EMA) (ETA: 3m 0s)
# ...
```

## Real-World Timing Examples

Based on typical T4 performance:

### Parallel Execution
```
00:00 - Pipeline starts, 3 trainings begin
00:03 - All trainings finish (± 30s variation)
00:03 - All 3 exploits begin immediately
00:18 - All exploits finish (± 2 min variation)

Total: 18 minutes ± 2 minutes
```

### Sequential Execution
```
00:00 - Exp 1 training starts
00:03 - Exp 1 training done, exploit starts
00:18 - Exp 1 complete
00:18 - Exp 2 training starts
00:21 - Exp 2 training done, exploit starts
00:36 - Exp 2 complete
00:36 - Exp 3 training starts
00:39 - Exp 3 training done, exploit starts
00:54 - Exp 3 complete

Total: 54 minutes ± 5 minutes
```

## Which Should You Use?

### Use Parallel If:
- ✅ You want results quickly (default choice)
- ✅ You have enough GPU memory (T4 15GB = yes)
- ✅ You're running production experiments
- ✅ You value your time

### Use Sequential If:
- ⚠️ You're debugging a problem
- ⚠️ You want to watch each job individually
- ⚠️ You're learning how the pipeline works
- ⚠️ You have severe GPU memory constraints (not your case)

## Bottom Line

**For your T4 setup: Use parallel. It's faster, more efficient, and has no downsides.**

The sequential script is provided as:
1. A learning tool to understand the pipeline
2. A debugging tool when something goes wrong
3. An alternative if you prefer simpler execution

But for normal usage, **parallel is strictly better** on your hardware.

## Script Locations

```bash
# Parallel (recommended)
./scripts/run_leduc_pipeline.sh

# Sequential (debugging/learning)
./scripts/run_leduc_sequential.sh
```

Both scripts include:
- Real-time progress tracking
- Elapsed/remaining time estimates
- Detailed timing breakdowns
- Success/failure tracking
- Final summary with total time
