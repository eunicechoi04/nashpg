#!/bin/bash
# Training script for Leduc Poker - Baseline (no EMA)

uv run train/nash_pg.py \
    agent=leduc_poker \
    env=leduc_poker \
    algorithm.use_ema=false \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=50 \
    run_name="leduc_poker/nash_pg/baseline_no_ema" \
    seed=100
