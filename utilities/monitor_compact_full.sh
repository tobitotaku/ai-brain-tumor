#!/bin/bash
# Quick training progress monitor

LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log found"
    exit 1
fi

echo "=================================================="
echo "Training Progress Monitor"
echo "=================================================="
echo "Log file: $LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if process is running
PID=$(ps aux | grep "train_cv.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    CPU=$(ps aux | grep "$PID" | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep "$PID" | grep -v grep | awk '{print $4}')
    echo "Status: ✅ RUNNING (PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%)"
else
    echo "Status: ⏸  STOPPED"
fi

echo ""
echo "Latest log entries:"
echo "--------------------------------------------------"
tail -30 "$LOG_FILE" | grep -E "STEP|fold|Results|ROC-AUC|Training|filter_l1|pca" || tail -30 "$LOG_FILE"
echo "--------------------------------------------------"

echo ""
echo "Model combinations trained:"
if [ -f "reports/tables/nested_cv_results.csv" ]; then
    echo "  $(tail -n +2 reports/tables/nested_cv_results.csv | cut -d',' -f1,2 | sort -u | wc -l) / 4 combinations"
    echo ""
    echo "  Completed:"
    tail -n +2 reports/tables/nested_cv_results.csv | cut -d',' -f1,2 | sort -u | sed 's/^/    - /'
else
    echo "  No results file yet"
fi

echo ""
echo "=================================================="
