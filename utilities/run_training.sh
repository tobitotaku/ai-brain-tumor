#!/bin/bash
# Professional training launcher with diagnostics and resource management
# Handles stdout/stderr separately to prevent file descriptor issues
# Auto-detects and warns about stale processes

set -e  # Exit on error

CONFIG="${1:-../config/config_ultrafast_pca.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
PID_FILE=".training_pid"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Training Launcher - Professional Mode"
echo "=================================================="
echo "Config: $CONFIG"
echo "Logs: $LOG_DIR/training_*.log (created by train_cv.py)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Pre-flight checks
echo "Running pre-flight checks..."
echo ""

# Check for stale processes
STALE_COUNT=$(ps aux | grep "LokyProcess" | grep -v grep | awk '{print $2}' | while read pid; do
    if ! pgrep -q -f "train_cv.py"; then
        echo $pid
    fi
done | wc -l)

if [ "$STALE_COUNT" -gt 0 ]; then
    echo "⚠  WARNING: Found $STALE_COUNT stale LokyProcess workers from previous runs"
    echo "   These may slow down training by competing for resources"
    echo "   To clean up: ./cleanup_stale.sh"
    echo ""
fi

# Check system memory
FREE_GB=$(vm_stat | grep "Pages free" | awk '{printf "%.1f", $3/262144}')
echo "✓ Available RAM: ~${FREE_GB}GB / 48GB"

# Check configuration file
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
echo "✓ Config file found: $CONFIG"
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Training already running (PID: $OLD_PID)"
        echo "Stop it first with: kill $OLD_PID"
        exit 1
    else
        echo "Cleaning up stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "ERROR: Virtual environment not found"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "✓ Python: $PYTHON_VERSION"

# Set CPU threading for optimal performance on Mac
NUM_CORES=$(sysctl -n hw.logicalcpu)
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export OPENBLAS_NUM_THREADS=$NUM_CORES
export VECLIB_MAXIMUM_THREADS=$NUM_CORES
export NUMEXPR_NUM_THREADS=$NUM_CORES
echo "✓ CPU threads set to: $NUM_CORES cores"

# Launch training in background with proper I/O redirection
echo ""
echo "Starting training in background..."
echo "Command: python -u scripts/train_cv.py --config $CONFIG"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/training_*.log        # Live log (created by train_cv.py)"
echo ""
echo "Starting..."
echo ""
echo ""

# Use python -u for unbuffered output
# Note: train_cv.py creates its own timestamped log in logs/training_*.log
# We only capture stderr here for error handling
nohup python -u scripts/train_cv.py --config "$CONFIG" \
    2>> "$LOG_DIR/training_stderr.log" &

TRAINING_PID=$!
echo "$TRAINING_PID" > "$PID_FILE"

# Wait a moment to check if process started successfully
sleep 2

if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
    echo "✓ Training started successfully"
    echo "  PID: $TRAINING_PID"
    echo "  Monitor with: tail -f logs/training_*.log"
    echo "  Stop with: kill $TRAINING_PID"
    echo ""
    echo "Waiting for log file to be created..."
    sleep 2
    LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [ -f "$LOG_FILE" ]; then
        echo "Initial log output:"
        echo "-------------------"
        head -20 "$LOG_FILE"
    fi
else
    echo "✗ Training failed to start"
    echo "Check errors in: logs/training_stderr.log"
    rm "$PID_FILE"
    exit 1
fi
