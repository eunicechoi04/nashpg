#!/bin/bash
# Sequential Pipeline for Leduc Poker: Training → Exploitability Computation
#
# This script runs experiments one at a time (train → exploit → train → exploit)
# Slower than parallel but uses less GPU memory and is easier to debug

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Leduc Poker Sequential Training Pipeline"
echo "=========================================="
echo ""
echo "Running mode: Sequential (one at a time)"
echo ""

# Configuration
EXPERIMENTS=("baseline" "ema_0001" "ema_002")
NAMES=("Baseline (no EMA)" "EMA tau=0.001" "EMA tau=0.02")

# Time estimates (in seconds)
EXPECTED_TRAIN_TIME=180    # ~3 minutes
EXPECTED_EXPLOIT_TIME=900  # ~15 minutes
EXPECTED_TOTAL_PER_EXP=$((EXPECTED_TRAIN_TIME + EXPECTED_EXPLOIT_TIME))  # 18 minutes
EXPECTED_TOTAL_PIPELINE=$((EXPECTED_TOTAL_PER_EXP * 3))  # 54 minutes

# Track timing
declare -A TRAIN_START_TIME
declare -A TRAIN_END_TIME
declare -A TRAIN_DURATION
declare -A EXPLOIT_START_TIME
declare -A EXPLOIT_END_TIME
declare -A EXPLOIT_DURATION

PIPELINE_START_TIME=$(date +%s)

# Helper function to format time duration
format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local secs=$((seconds % 60))
    printf "%dm %ds" $minutes $secs
}

# Helper function to print progress
print_progress() {
    local current=$1
    local total=$2
    local elapsed=$(($(date +%s) - PIPELINE_START_TIME))
    local remaining=$((EXPECTED_TOTAL_PIPELINE - elapsed))
    if [ $remaining -lt 0 ]; then remaining=0; fi

    echo ""
    echo "=========================================="
    echo "Progress: $current/$total experiments"
    echo "Elapsed: $(format_duration $elapsed)"
    echo "Remaining: ~$(format_duration $remaining)"
    echo "=========================================="
    echo ""
}

# Create log directory if it doesn't exist
mkdir -p logs

echo "Pipeline timeline:"
echo "  Expected time per experiment: $(format_duration $EXPECTED_TOTAL_PER_EXP)"
echo "  Expected total time: $(format_duration $EXPECTED_TOTAL_PIPELINE)"
echo ""

# Track overall stats
total_experiments=${#EXPERIMENTS[@]}
completed_experiments=0
failed_experiments=0

# Run each experiment sequentially
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    name="${NAMES[$i]}"
    exp_num=$((i + 1))

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Experiment $exp_num/$total_experiments: $name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Phase 1: Training
    echo -e "${BLUE}[TRAIN]${NC} Starting: $name (ETA: $(format_duration $EXPECTED_TRAIN_TIME))"
    TRAIN_START_TIME[$exp]=$(date +%s)

    if ./scripts/train_leduc_${exp}.sh > logs/train_leduc_${exp}.log 2>&1; then
        TRAIN_END_TIME[$exp]=$(date +%s)
        TRAIN_DURATION[$exp]=$((TRAIN_END_TIME[$exp] - TRAIN_START_TIME[$exp]))
        echo -e "${GREEN}✓ [TRAIN]${NC} $name completed successfully! ($(format_duration ${TRAIN_DURATION[$exp]}))"
        echo ""

        # Phase 2: Exploitability
        echo -e "${YELLOW}[EXPLOIT]${NC} Starting: $name (ETA: $(format_duration $EXPECTED_EXPLOIT_TIME))"
        EXPLOIT_START_TIME[$exp]=$(date +%s)

        if ./scripts/compute_exploit_leduc_${exp}.sh > logs/exploit_leduc_${exp}.log 2>&1; then
            EXPLOIT_END_TIME[$exp]=$(date +%s)
            EXPLOIT_DURATION[$exp]=$((EXPLOIT_END_TIME[$exp] - EXPLOIT_START_TIME[$exp]))
            echo -e "${GREEN}✓ [EXPLOIT]${NC} $name completed successfully! ($(format_duration ${EXPLOIT_DURATION[$exp]}))"

            completed_experiments=$((completed_experiments + 1))

            echo ""
            echo -e "${GREEN}✓ Experiment $exp_num complete!${NC}"
            echo "  Training: $(format_duration ${TRAIN_DURATION[$exp]})"
            echo "  Exploit:  $(format_duration ${EXPLOIT_DURATION[$exp]})"
            local exp_total=$((TRAIN_DURATION[$exp] + EXPLOIT_DURATION[$exp]))
            echo "  Total:    $(format_duration $exp_total)"
        else
            echo -e "${RED}✗ [EXPLOIT]${NC} $name failed!"
            failed_experiments=$((failed_experiments + 1))
        fi
    else
        echo -e "${RED}✗ [TRAIN]${NC} $name failed!"
        failed_experiments=$((failed_experiments + 1))
    fi

    # Print progress after each experiment
    print_progress $((completed_experiments + failed_experiments)) $total_experiments
done

# Final summary
PIPELINE_END_TIME=$(date +%s)
TOTAL_PIPELINE_TIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Summary:"
echo "  ✓ Successful: $completed_experiments/$total_experiments"
echo "  ✗ Failed: $failed_experiments/$total_experiments"
echo "  ⏱ Total time: $(format_duration $TOTAL_PIPELINE_TIME)"
echo ""

if [ $completed_experiments -gt 0 ]; then
    echo "Individual experiment times:"
    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        name="${NAMES[$i]}"

        if [ -n "${TRAIN_DURATION[$exp]}" ] && [ -n "${EXPLOIT_DURATION[$exp]}" ]; then
            local exp_total=$((TRAIN_DURATION[$exp] + EXPLOIT_DURATION[$exp]))
            echo "  $name:"
            echo "    Train:   $(format_duration ${TRAIN_DURATION[$exp]})"
            echo "    Exploit: $(format_duration ${EXPLOIT_DURATION[$exp]})"
            echo "    Total:   $(format_duration $exp_total)"
        fi
    done
    echo ""
fi

echo "Results location:"
echo "  Logs: logs/leduc_poker/nash_pg/*.json"
echo "  Checkpoints: checkpoints/leduc_poker/nash_pg/*"
echo "  TensorBoard: runs/leduc_poker/nash_pg/*"
echo ""

if [ $failed_experiments -eq 0 ]; then
    echo -e "${GREEN}All experiments completed successfully! 🎉${NC}"
    exit 0
else
    echo -e "${RED}Some experiments failed. Check logs for details.${NC}"
    exit 1
fi
