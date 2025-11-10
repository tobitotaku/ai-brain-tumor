#!/bin/bash
# Training Monitor Script
# Usage: ./check_training.sh

PID_FILE=".training_pid"
LOG_FILE="training_filter_l1_optimized.log"

echo "═══════════════════════════════════════════════════════════"
echo "🔬 FILTER_L1 OPTIMIZED TRAINING - STATUS CHECK"
echo "═══════════════════════════════════════════════════════════"
echo "⏰ Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ No training PID file found"
    echo "   Training may not have been started yet"
    exit 1
fi

PID=$(cat "$PID_FILE")
echo "📌 Training PID: $PID"
echo ""

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process is RUNNING"
    echo ""
    echo "📊 Resource usage:"
    ps -p $PID -o pid,state,%cpu,%mem,etime,rss | head -2
    
    RSS=$(ps -p $PID -o rss= | xargs)
    RAM_GB=$(echo "scale=2; $RSS / 1024 / 1024" | bc)
    echo "   RAM: ${RAM_GB} GB"
    echo ""
else
    echo "❌ Process is NOT RUNNING (may have completed or crashed)"
    echo ""
fi

# Log file info
if [ -f "$LOG_FILE" ]; then
    LINES=$(wc -l < "$LOG_FILE")
    SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    echo "📝 Log file: $LOG_FILE"
    echo "   Lines: $LINES"
    echo "   Size: $SIZE"
    echo ""
    
    # Check for completion markers
    if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
        echo "🎉 TRAINING COMPLETED!"
        echo ""
    elif grep -q "ERROR\|Traceback" "$LOG_FILE" 2>/dev/null; then
        echo "⚠️  ERRORS DETECTED in log"
        echo ""
        echo "Last error:"
        grep -A 5 "ERROR\|Traceback" "$LOG_FILE" | tail -10
        echo ""
    fi
    
    # Show last progress
    echo "───────────────────────────────────────────────────────────"
    echo "📄 Last 25 lines of log:"
    echo "───────────────────────────────────────────────────────────"
    tail -25 "$LOG_FILE"
else
    echo "❌ Log file not found: $LOG_FILE"
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
echo "💡 Tips:"
echo "   - Watch live: tail -f $LOG_FILE"
echo "   - Kill training: kill $PID"
echo "   - Check errors: grep -A 10 'ERROR\\|Traceback' $LOG_FILE"
echo ""
