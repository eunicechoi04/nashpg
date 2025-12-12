#!/bin/bash
# Master script to run all Kuhn Poker EMA experiments

echo "Running all Kuhn Poker EMA experiments..."
echo "=========================================="

echo ""
echo "1. Running Baseline (no EMA)..."
# ./train_kuhn_baseline.sh

echo ""
echo "2. Running EMA with tau=0.001..."
./train_kuhn_ema_0001.sh

echo ""
echo "3. Running EMA with tau=0.005..."
./train_kuhn_ema_0005.sh

echo ""
echo "4. Running EMA with tau=0.02..."
./train_kuhn_ema_002.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
