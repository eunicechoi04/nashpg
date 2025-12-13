# EMA Stability Studies - Plotting Scripts

This directory contains plotting scripts for analyzing the stability of the reference policy (EMA claims) across different metrics.

## Available Studies

### Study 1: mag_kl vs Training Steps
**Script:** `plot_study1_mag_kl.py`  
**Metric:** `mag_kl = KL(π || ρ)`  
**Purpose:** Visualizes the divergence between the current policy and reference policy over time.

```bash
python scripts/plot_study1_mag_kl.py logs/leduc_poker/nash_pg/ema_tau_005.json \
    --output plots/study1_mag_kl.png \
    --labels "EMA τ=0.05"
```

### Study 2: Variance of mag_kl
**Script:** `plot_study2_mag_kl_variance.py`  
**Metric:** `mag_kl`  
**Purpose:** Computes rolling variance and standard deviation across seeds at each timestep.

```bash
python scripts/plot_study2_mag_kl_variance.py \
    logs/leduc_poker/nash_pg/seed1.json \
    logs/leduc_poker/nash_pg/seed2.json \
    logs/leduc_poker/nash_pg/seed3.json \
    --output plots/study2_mag_kl_variance.png \
    --labels "Seed 1" "Seed 2" "Seed 3" \
    --window 100
```

### Study 3: approx_kl vs Training Steps
**Script:** `plot_study3_approx_kl.py`  
**Metric:** `approx_kl = KL(π_new || π_old)`  
**Purpose:** Shows the magnitude of policy updates at each training step.

```bash
python scripts/plot_study3_approx_kl.py logs/leduc_poker/nash_pg/ema_tau_005.json \
    --output plots/study3_approx_kl.png
```

### Study 4: clip_frac vs Training Steps
**Script:** `plot_study4_clip_frac.py`  
**Metric:** `clip_frac`  
**Purpose:** Tracks the fraction of policy updates that are being clipped by PPO.

```bash
python scripts/plot_study4_clip_frac.py logs/leduc_poker/nash_pg/ema_tau_005.json \
    --output plots/study4_clip_frac.png
```

### Study 5: ppo_loss vs Training Steps
**Script:** `plot_study5_ppo_loss.py`  
**Metric:** `ppo_loss`  
**Purpose:** Monitors the PPO loss over training.

```bash
python scripts/plot_study5_ppo_loss.py logs/leduc_poker/nash_pg/ema_tau_005.json \
    --output plots/study5_ppo_loss.png
```

### Study 7: entropy vs Training Steps
**Script:** `plot_study7_entropy.py`  
**Metric:** `entropy`  
**Purpose:** Tracks policy entropy to understand exploration behavior.

```bash
python scripts/plot_study7_entropy.py logs/leduc_poker/nash_pg/ema_tau_005.json \
    --output plots/study7_entropy.png
```

## General Usage

All scripts follow the same command-line interface:

```bash
python scripts/plot_studyX_<metric>.py <log_file1> [log_file2] ... [OPTIONS]
```

### Common Options:
- `log_files`: One or more paths to JSON log files (required)
- `--output`, `-o`: Path to save the plot (optional, shows plot if not provided)
- `--labels`, `-l`: Custom labels for each log file (optional, uses filename by default)

### Multiple Log Files Example:

Compare different EMA tau values:
```bash
python scripts/plot_study1_mag_kl.py \
    logs/leduc_poker/nash_pg/ema_tau_001.json \
    logs/leduc_poker/nash_pg/ema_tau_005.json \
    logs/leduc_poker/nash_pg/ema_tau_010.json \
    --output plots/mag_kl_comparison.png \
    --labels "τ=0.01" "τ=0.05" "τ=0.10"
```

Compare different games:
```bash
python scripts/plot_study3_approx_kl.py \
    logs/kuhn_poker/nash_pg/run1.json \
    logs/leduc_poker/nash_pg/run1.json \
    --output plots/approx_kl_games.png \
    --labels "Kuhn Poker" "Leduc Poker"
```

## Data Format

Scripts expect JSON log files with the following structure:

```json
{
  "config": { ... },
  "train": [
    {
      "step": 10,
      "mag_kl": 0.010676993057131767,
      "approx_kl": 0.0016852751141414046,
      "clip_frac": 0.017148394137620926,
      "ppo_loss": -0.00631805881857872,
      "entropy": 0.920564591884613,
      ...
    },
    ...
  ]
}
```

## Dependencies

```bash
pip install matplotlib numpy pandas
```

## Tips

1. **Multiple Seeds**: For Study 2 (variance analysis), provide multiple log files from different seeds
2. **Output Format**: Plots are saved as PNG files at 300 DPI by default
3. **Interactive Mode**: Omit `--output` to display plots interactively
4. **Custom Labels**: Use meaningful labels when comparing multiple runs for clearer legends
