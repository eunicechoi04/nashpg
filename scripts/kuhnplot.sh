#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p plots/kuhn_ema

# Study 1: mag_kl vs Training Steps
echo "Running Study 1: mag_kl vs Training Steps..."
uv run scripts/plot_study1_mag_kl.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \  
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02"
    --output plots/kuhn_ema/study1_mag_kl_comparison.png

# Study 2: Variance of mag_kl (across different tau configurations)
echo "Running Study 2: Variance of mag_kl..."
uv run scripts/plot_study2_mag_kl_variance.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02" \
    --window 100 \
    --output plots/kuhn_ema/study2_mag_kl_variance.png

# Study 3: approx_kl vs Training Steps
echo "Running Study 3: approx_kl vs Training Steps..."
uv run scripts/plot_study3_approx_kl.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02" \
    --output plots/kuhn_ema/study3_approx_kl.png

# Study 4: clip_frac vs Training Steps
echo "Running Study 4: clip_frac vs Training Steps..."
uv run scripts/plot_study4_clip_frac.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02" \
    --output plots/kuhn_ema/study4_clip_frac.png

# Study 5: ppo_loss vs Training Steps
echo "Running Study 5: ppo_loss vs Training Steps..."
uv run scripts/plot_study5_ppo_loss.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02" \
    --output plots/kuhn_ema/study5_ppo_loss.png

# Study 7: entropy vs Training Steps
echo "Running Study 7: entropy vs Training Steps..."
uv run scripts/plot_study7_entropy.py \
    logs/kuhn_poker/nash_pg/baseline_no_ema.json \
    logs/kuhn_poker/nash_pg/ema_tau_0001.json \
    logs/kuhn_poker/nash_pg/ema_tau_0005.json \
    logs/kuhn_poker/nash_pg/ema_tau_002.json \
    --labels "No EMA" "τ=0.001" "τ=0.005" "τ=0.02" \
    --output plots/kuhn_ema/study7_entropy.png
echo "All plots generated successfully in plots/kuhn_ema/"

