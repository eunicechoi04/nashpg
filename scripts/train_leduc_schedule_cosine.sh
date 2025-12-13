#!/bin/bash
# Training script for Leduc Poker - Cosine Schedule

uv run train/nash_pg.py \
    agent=leduc_poker \
    env=leduc_poker \
    algorithm.alpha_schedule_type="cosine" \
    algorithm.alpha_high=0.4 \
    algorithm.alpha_target=0.2 \
    algorithm.T_warmup=10 \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=50 \
    run_name="leduc_poker/nash_pg/schedule_cosine" \
    seed=100
