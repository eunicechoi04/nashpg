#!/bin/bash
# Training script for Leduc Poker - EMA with tau=0.005

uv run train/nash_pg.py \
    agent=leduc_poker \
    env=leduc_poker \
    algorithm.use_ema=true \
    algorithm.ema_tau=0.005 \
    algorithm.ema_update_freq=1 \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=50 \
    run_name="leduc_poker/nash_pg/ema_tau_0005" \
    seed=100
