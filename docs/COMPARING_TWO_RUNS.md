# Comparing Two Nash-PG Runs

This guide explains how to use the comparison scripts to evaluate two different Nash-PG training runs (e.g., adaptive KL vs fixed KL) on the same game.

## Overview

Two scripts are provided:
1. **`compare_two_runs_exploitability.py`** - Compares exploitability metrics
2. **`compare_two_runs_elo.py`** - Compares ELO ratings via head-to-head play

## Prerequisites

Before running comparisons, you need to have trained two models. For example:

```bash
# Train with adaptive KL (default in nash_pg.yaml)
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent=kuhn_poker \
    env=kuhn_poker \
    run_name="kuhn_poker/nash_pg_adaptive"

# Train with fixed KL (disable adaptive)
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    algorithm.adaptive_kl_penalty=false \
    agent=kuhn_poker \
    env=kuhn_poker \
    run_name="kuhn_poker/nash_pg_fixed"
```

This will create:
- Logs: `logs/kuhn_poker/nash_pg_adaptive.json` and `logs/kuhn_poker/nash_pg_fixed.json`
- Checkpoints: `checkpoints/kuhn_poker/nash_pg_adaptive/` and `checkpoints/kuhn_poker/nash_pg_fixed/`

## Exploitability Comparison

Compare how exploitable each model is over training:

```bash
# Basic usage
uv run scripts/compare_two_runs_exploitability.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL"
```

### Options

- `--env`: Environment name (required)
- `--run1`: First run name (required)
- `--run2`: Second run name (required)
- `--label1`: Label for first run (default: "Run 1")
- `--label2`: Label for second run (default: "Run 2")
- `--gpu`: GPU device ID (default: 0)
- `--skip-compute`: Skip computation, use existing exploitability data
- `--output`: Save results to JSON file

### Example with options

```bash
# Use GPU 1 and save results
uv run scripts/compare_two_runs_exploitability.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL" \
    --gpu 1 \
    --output results/exploit_comparison.json
```

### Output

The script will print a comparison table:

```
======================================================================
EXPLOITABILITY COMPARISON
======================================================================

Adaptive KL (kuhn_poker/nash_pg_adaptive):
  Final exploitability: 0.012345
  Mean exploitability:  0.015678
  Min exploitability:   0.010123

Fixed KL (kuhn_poker/nash_pg_fixed):
  Final exploitability: 0.018765
  Mean exploitability:  0.021234
  Min exploitability:   0.015432

Comparison:
  Adaptive KL is 34.23% BETTER (lower exploitability)
======================================================================
```

## ELO Comparison

Compare performance via head-to-head gameplay:

```bash
# Basic usage
uv run scripts/compare_two_runs_elo.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL"
```

### Options

- `--env`: Environment name (required)
- `--run1`: First run name (required)
- `--run2`: Second run name (required)
- `--label1`: Label for first run (default: "Run 1")
- `--label2`: Label for second run (default: "Run 2")
- `--games-per-pairing`: Number of games per comparison (default: 50)
- `--output`: Save results to JSON file

### Example with options

```bash
# Play 100 games per comparison and save results
uv run scripts/compare_two_runs_elo.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL" \
    --games-per-pairing 100 \
    --output results/elo_comparison.json
```

### Output

The script will print a comparison summary:

```
======================================================================
ELO COMPARISON (HEAD-TO-HEAD)
======================================================================

Adaptive KL (kuhn_poker/nash_pg_adaptive):
  Average win rate: 0.623
  Final win rate:   0.645

Fixed KL (kuhn_poker/nash_pg_fixed):
  Average win rate: 0.377
  Final win rate:   0.355

Number of comparisons: 11
Games per comparison: 50

Adaptive KL performs 24.6% better on average
======================================================================
```

## Complete Workflow

Here's a complete workflow for comparing adaptive vs fixed KL:

```bash
# 1. Train both models
./train_both.sh  # (create this script with both training commands)

# 2. Compute and compare exploitability
uv run scripts/compare_two_runs_exploitability.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL" \
    --output results/exploit_comparison.json

# 3. Compute and compare ELO
uv run scripts/compare_two_runs_elo.py \
    --env kuhn_poker \
    --run1 "kuhn_poker/nash_pg_adaptive" \
    --run2 "kuhn_poker/nash_pg_fixed" \
    --label1 "Adaptive KL" \
    --label2 "Fixed KL" \
    --games-per-pairing 100 \
    --output results/elo_comparison.json
```

## Understanding the Results

### Exploitability Metrics

- **Lower is better** - Lower exploitability means the policy is closer to Nash equilibrium
- **Final exploitability** - How exploitable the final trained policy is
- **Mean exploitability** - Average across all training steps
- **Min exploitability** - Best (lowest) exploitability achieved

### ELO/Win Rate Metrics

- **Higher win rate is better** - Shows which model performs better in head-to-head play
- **Average win rate** - Performance across all checkpoint steps
- **Final win rate** - Performance at the end of training

### What to Look For

When comparing adaptive vs fixed KL:

1. **Lower exploitability** with adaptive KL suggests it finds better Nash equilibria
2. **Higher win rate** with adaptive KL shows it's a stronger player
3. **More stable metrics** over training indicates better convergence
4. **Faster convergence** to low exploitability shows efficiency

## Tips

1. **Use same training settings** - Only vary the `adaptive_kl_penalty` parameter
2. **Run multiple seeds** - Use different random seeds for statistical significance
3. **Save results** - Use `--output` to keep JSON files for later analysis
4. **Monitor mag_coef** - Check TensorBoard logs to see how mag_coef adapts
5. **Different environments** - Test on multiple games (Kuhn Poker, Leduc Poker, etc.)

## Troubleshooting

**Issue**: "Log file not found"
- **Solution**: Check that training completed and created the log file in `logs/`

**Issue**: "Checkpoint directory not found"
- **Solution**: Ensure `logging.save_interval > 0` was set during training

**Issue**: "No common checkpoint steps"
- **Solution**: Make sure both runs used the same `num_inner_update` and `save_interval`

**Issue**: Exploitability computation is slow
- **Solution**: Use `--skip-compute` if you've already computed exploitability once
