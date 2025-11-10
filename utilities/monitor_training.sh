#!/bin/bash
# Professional training monitor - checks status and displays logs

PID_FILE=".training_pid"
LOG_DIR="logs"

if [ ! -f "$PID_FILE" ]; then
    echo "No training process found (PID file missing)"
    echo ""
    echo "Recent logs:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "=================================================="
echo "Training Monitor"
echo "=================================================="
echo ""

# Check if process is running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "✓ Training is RUNNING (PID: $PID)"
    echo ""
    
    # Get process stats
    echo "Process Information:"
    ps -p "$PID" -o pid,state,%cpu,%mem,etime,command | tail -n +2
    echo ""
    
    # Check for child processes (workers)
    CHILDREN=$(pgrep -P "$PID" 2>/dev/null | wc -l)
    echo "Active workers: $CHILDREN"
    if [ "$CHILDREN" -gt 0 ]; then
        echo "Worker details:"
        ps -p $(pgrep -P "$PID" | tr '\n' ',' | sed 's/,$//') -o pid,state,%cpu,%mem,command 2>/dev/null | head -10
    fi
    echo ""
else
    echo "✗ Training process is NOT running"
    echo "  PID $PID has terminated"
    rm "$PID_FILE"
    echo ""
fi

# Find most recent log files
LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
LATEST_ERR=$(ls -t "$LOG_DIR"/*.err 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "=================================================="
    echo "Latest Log: $LATEST_LOG"
    echo "=================================================="
    tail -30 "$LATEST_LOG"
    echo ""
fi

if [ -n "$LATEST_ERR" ] && [ -s "$LATEST_ERR" ]; then
    echo "=================================================="
    echo "Latest Errors: $LATEST_ERR"
    echo "=================================================="
    tail -20 "$LATEST_ERR"
    echo ""
fi

# Check for completion markers
if [ -n "$LATEST_LOG" ]; then
    if grep -q "TRAINING COMPLETE" "$LATEST_LOG" 2>/dev/null; then
        echo "✓ TRAINING COMPLETED SUCCESSFULLY"
    elif grep -qi "error\|exception\|failed" "$LATEST_LOG" 2>/dev/null; then
        echo "⚠ Potential errors detected - check full log"
    fi
fi

echo ""
echo "Useful commands:"
echo "  tail -f $LATEST_LOG     # Watch log in real-time"
echo "  tail -f $LATEST_ERR     # Watch errors in real-time"
echo "  kill $PID                     # Stop training"
echo "  grep -i error $LATEST_LOG     # Search for errors"
