#!/bin/bash
# Master script to run all Kuhn Poker EMA experiments IN PARALLEL

echo "=========================================="
echo "Running All EMA Experiments in PARALLEL"
echo "=========================================="
echo ""

# Run all 4 experiments in parallel (in background with &)
# Each uses ~1GB of GPU memory, T4 has 15GB total
./scripts/train_kuhn_baseline.sh &
PID1=$!

./scripts/train_kuhn_ema_0001.sh &
PID2=$!

./scripts/train_kuhn_ema_0005.sh &
PID3=$!

./scripts/train_kuhn_ema_002.sh &
PID4=$!

# Wait for all to complete
echo "Waiting for all parallel experiments to complete..."
wait $PID1
echo "✓ Baseline completed"

wait $PID2
echo "✓ EMA tau=0.001 completed"

wait $PID3
echo "✓ EMA tau=0.005 completed"

wait $PID4
echo "✓ EMA tau=0.02 completed"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
