#!/bin/bash
# This file run trainin agent -> compute exploitability -> compute Elo ratings
env_name=kuhn_poker
num_runs=4

uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent=$env_name \
    env=$env_name \
    run_name="$env_name/nash_pg"

uv run eval/compute_exploitability.py ./logs/$env_name/nash_pg.json rl