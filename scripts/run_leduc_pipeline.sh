#!/bin/bash
# Parallel Pipeline for Leduc Poker: Training → Exploitability Computation
#
# This script:
# 1. Starts all 3 training jobs in parallel
# 2. Monitors each training completion
# 3. Immediately launches exploitability computation when training finishes
# 4. Runs exploitability computations in parallel as they become available
# 5. Provides real-time status updates

set -e

# Helper function to format time duration
format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local secs=$((seconds % 60))
    printf "%dm %ds" $minutes $secs
}

# Time estimates (in seconds)
EXPECTED_TRAIN_TIME=180
EXPECTED_EXPLOIT_TIME=900


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Leduc Poker Parallel Training Pipeline"
echo "=========================================="
echo ""
echo "Pipeline stages:"
echo "  1. Training (3 models in parallel) - ETA: $(format_duration $EXPECTED_TRAIN_TIME) each"
echo "  2. Exploitability (starts as each training completes) - ETA: $(format_duration $EXPECTED_EXPLOIT_TIME) each"
echo ""
echo "Expected total time: ~18 minutes (with overlapping exploits)"
echo ""

# Track overall pipeline start time
PIPELINE_START_TIME=$(date +%s)

# Create arrays to track jobs
declare -A TRAIN_PIDS
declare -A TRAIN_NAMES
declare -A TRAIN_STATUS
declare -A EXPLOIT_PIDS
declare -A EXPLOIT_STATUS
declare -A TRAIN_START_TIME
declare -A TRAIN_END_TIME
declare -A EXPLOIT_START_TIME
declare -A EXPLOIT_END_TIME

# Configuration
EXPERIMENTS=("baseline" "ema_0001" "ema_002")
NAMES=("Baseline (no EMA)" "EMA tau=0.001" "EMA tau=0.02")

# Time estimates (in seconds)
EXPECTED_TRAIN_TIME=180    # ~3 minutes
EXPECTED_EXPLOIT_TIME=900  # ~15 minutes

# Helper function to format time duration
format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local secs=$((seconds % 60))
    printf "%dm %ds" $minutes $secs
}

# Function to run training in background
run_training() {
    exp=$1
    name=$2

    TRAIN_START_TIME[$exp]=$(date +%s)
    echo -e "${BLUE}[TRAIN]${NC} Starting: $name (ETA: $(format_duration $EXPECTED_TRAIN_TIME))"
    ./scripts/train_leduc_${exp}.sh > logs/train_leduc_${exp}.log 2>&1 &
    local pid=$!

    TRAIN_PIDS[$exp]=$pid
    TRAIN_NAMES[$exp]=$name
    TRAIN_STATUS[$exp]="running"

    echo -e "${BLUE}[TRAIN]${NC} $name → PID $pid"
}

# Function to run exploitability in background
run_exploit() {
    exp=$1
    name=$2

    EXPLOIT_START_TIME[$exp]=$(date +%s)
    echo -e "${YELLOW}[EXPLOIT]${NC} Starting: $name (ETA: $(format_duration $EXPECTED_EXPLOIT_TIME))"
    ./scripts/compute_exploit_leduc_${exp}.sh > logs/exploit_leduc_${exp}.log 2>&1 &
    local pid=$!

    EXPLOIT_PIDS[$exp]=$pid
    EXPLOIT_STATUS[$exp]="running"

    echo -e "${YELLOW}[EXPLOIT]${NC} $name → PID $pid"
}

# Function to check if process is still running
is_running() {
    local pid=$1
    kill -0 $pid 2>/dev/null
}

# Function to print status summary
print_status() {
    local now=$(date +%s)

    echo ""
    echo "=========================================="
    echo "Current Status:"
    echo "=========================================="

    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        name="${NAMES[$i]}"

        local train_status="${TRAIN_STATUS[$exp]}"
        local exploit_status="${EXPLOIT_STATUS[$exp]:-pending}"

        local train_info=""
        local exploit_info=""

        # Calculate training time info
        if [ "$train_status" == "done" ] && [ -n "${TRAIN_END_TIME[$exp]}" ]; then
            duration=$((TRAIN_END_TIME[$exp] - TRAIN_START_TIME[$exp]))
            train_info="✓ ($(format_duration $duration))"
        elif [ "$train_status" == "running" ] && [ -n "${TRAIN_START_TIME[$exp]}" ]; then
            local elapsed=$((now - TRAIN_START_TIME[$exp]))
            local remaining=$((EXPECTED_TRAIN_TIME - elapsed))
            if [ $remaining -lt 0 ]; then remaining=0; fi
            train_info="⚙ ($(format_duration $elapsed) / ~$(format_duration $EXPECTED_TRAIN_TIME))"
        fi

        # Calculate exploit time info
        if [ "$exploit_status" == "done" ] && [ -n "${EXPLOIT_END_TIME[$exp]}" ]; then
            duration=$((EXPLOIT_END_TIME[$exp] - EXPLOIT_START_TIME[$exp]))
            exploit_info="✓ ($(format_duration $duration))"
        elif [ "$exploit_status" == "running" ] && [ -n "${EXPLOIT_START_TIME[$exp]}" ]; then
            local elapsed=$((now - EXPLOIT_START_TIME[$exp]))
            local remaining=$((EXPECTED_EXPLOIT_TIME - elapsed))
            if [ $remaining -lt 0 ]; then remaining=0; fi
            exploit_info="⚙ ($(format_duration $elapsed) / ~$(format_duration $EXPECTED_EXPLOIT_TIME))"
        elif [ "$exploit_status" == "pending" ]; then
            exploit_info="⏳ (waiting)"
        fi

        # Print status line
        if [ "$train_status" == "done" ] && [ "$exploit_status" == "done" ]; then
            echo -e "${GREEN}$name${NC}"
            echo -e "  Train: $train_info | Exploit: $exploit_info"
        elif [ "$train_status" == "done" ] && [ "$exploit_status" == "running" ]; then
            echo -e "${YELLOW}$name${NC}"
            echo -e "  Train: $train_info | Exploit: $exploit_info"
        elif [ "$train_status" == "done" ]; then
            echo -e "${YELLOW}$name${NC}"
            echo -e "  Train: $train_info | Exploit: $exploit_info"
        elif [ "$train_status" == "running" ]; then
            echo -e "${BLUE}$name${NC}"
            echo -e "  Train: $train_info | Exploit: $exploit_info"
        fi
    done
    echo "=========================================="
    echo ""
}

# Create log directory if it doesn't exist
mkdir -p logs

# Phase 1: Start all trainings in parallel
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 1: Starting Training Jobs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    run_training "${EXPERIMENTS[$i]}" "${NAMES[$i]}"
    sleep 2  # Stagger starts slightly
done

echo ""
echo "All training jobs launched!"
print_status

# Phase 2: Monitor trainings and launch exploitability as they complete
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 2: Monitoring & Launching Exploits${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

all_done=false
check_count=0

while [ "$all_done" = false ]; do
    all_done=true
    check_count=$((check_count + 1))

    # Check each experiment
    for exp in "${EXPERIMENTS[@]}"; do
        # Check training completion
        if [ "${TRAIN_STATUS[$exp]}" == "running" ]; then
            all_done=false
            if ! is_running ${TRAIN_PIDS[$exp]}; then
                set +e
                wait ${TRAIN_PIDS[$exp]}
                exit_code=$?
                set -e

                exit_code=$?

                TRAIN_END_TIME[$exp]=$(date +%s)
                if [ $exit_code -eq 0 ]; then
                    TRAIN_STATUS[$exp]="done"
                    duration=$((TRAIN_END_TIME[$exp] - TRAIN_START_TIME[$exp]))
                    echo -e "${GREEN}✓ [TRAIN]${NC} ${TRAIN_NAMES[$exp]} completed successfully! ($(format_duration $duration))"

                    # Immediately launch exploitability computation
                    sleep 1
                    run_exploit "$exp" "${TRAIN_NAMES[$exp]}"
                else
                    TRAIN_STATUS[$exp]="failed"
                    echo -e "${RED}✗ [TRAIN]${NC} ${TRAIN_NAMES[$exp]} failed with exit code $exit_code"
                fi

                print_status
            fi
        fi

        # Check exploitability completion
        if [ "${EXPLOIT_STATUS[$exp]}" == "running" ]; then
            all_done=false
            if ! is_running ${EXPLOIT_PIDS[$exp]}; then
                set +e
                wait ${EXPLOIT_PIDS[$exp]}
                exit_code=$?
                set -e
                exit_code=$?

                EXPLOIT_END_TIME[$exp]=$(date +%s)
                if [ $exit_code -eq 0 ]; then
                    EXPLOIT_STATUS[$exp]="done"
                    duration=$((EXPLOIT_END_TIME[$exp] - EXPLOIT_START_TIME[$exp]))
                    echo -e "${GREEN}✓ [EXPLOIT]${NC} ${TRAIN_NAMES[$exp]} completed successfully! ($(format_duration $duration))"
                else
                    EXPLOIT_STATUS[$exp]="failed"
                    echo -e "${RED}✗ [EXPLOIT]${NC} ${TRAIN_NAMES[$exp]} failed with exit code $exit_code"
                fi

                print_status
            fi
        fi

        # Check if this experiment still has work to do
        if [ "${TRAIN_STATUS[$exp]}" == "running" ] || [ "${EXPLOIT_STATUS[$exp]}" == "running" ]; then
            all_done=false
        fi
    done

    # Print periodic status update
    if [ $((check_count % 30)) -eq 0 ]; then
        print_status
    fi

    # Don't spin too fast
    sleep 2
done

# Final summary
PIPELINE_END_TIME=$(date +%s)
TOTAL_PIPELINE_TIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Count successes and failures
success_count=0
failure_count=0

for exp in "${EXPERIMENTS[@]}"; do
    if [ "${TRAIN_STATUS[$exp]}" == "done" ] && [ "${EXPLOIT_STATUS[$exp]}" == "done" ]; then
        success_count=$((success_count + 1))
    else
        failure_count=$((failure_count + 1))
    fi
done

echo "Summary:"
echo "  ✓ Successful: $success_count/3"
echo "  ✗ Failed: $failure_count/3"
echo "  ⏱ Total time: $(format_duration $TOTAL_PIPELINE_TIME)"
echo ""

if [ $failure_count -gt 0 ]; then
    echo "Failed experiments:"
    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        name="${NAMES[$i]}"

        if [ "${TRAIN_STATUS[$exp]}" != "done" ] || [ "${EXPLOIT_STATUS[$exp]}" != "done" ]; then
            echo "  - $name (Train: ${TRAIN_STATUS[$exp]}, Exploit: ${EXPLOIT_STATUS[$exp]})"
        fi
    done
    echo ""
fi

echo "Results location:"
echo "  Logs: logs/leduc_poker/nash_pg/*.json"
echo "  Checkpoints: checkpoints/leduc_poker/nash_pg/*"
echo "  TensorBoard: runs/leduc_poker/nash_pg/*"
echo ""

if [ $failure_count -eq 0 ]; then
    echo -e "${GREEN}All experiments completed successfully! 🎉${NC}"
    exit 0
else
    echo -e "${RED}Some experiments failed. Check logs for details.${NC}"
    exit 1
fi
