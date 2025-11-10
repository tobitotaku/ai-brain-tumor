#!/bin/bash

# Detailed Training Monitor with Thread Analysis
# Shows real-time resource usage, thread counts, and memory pressure

MAIN_PID=66681
LOGFILE="logs/training_20251110_225121.log"
ERRFILE="logs/training_20251110_225121.err"

clear

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo "DETAILED TRAINING MONITOR - $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if process is running
    if ! pgrep -P $$ -f "train_cv.py" > /dev/null 2>&1; then
        if ps -p $MAIN_PID > /dev/null 2>&1; then
            echo "✓ Main process (PID $MAIN_PID) is RUNNING"
        else
            echo "✗ Main process is NOT RUNNING"
            echo ""
            echo "Training appears to have completed. Final status:"
            tail -5 "$LOGFILE"
            break
        fi
    else
        echo "✓ Main process (PID $MAIN_PID) is RUNNING"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "MEMORY & SYSTEM RESOURCES"
    echo "────────────────────────────────────────────────────────────────"
    
    # Get memory info
    MEMINFO=$(vm_stat | grep -E "Pages free:|Pages active:|Pages wired")
    FREE_PAGES=$(echo "$MEMINFO" | grep "Pages free" | awk '{print $3}' | tr -d '.')
    FREE_GB=$((FREE_PAGES / 262144))
    
    # Get main process info
    MAIN_INFO=$(ps -p $MAIN_PID -o pid,user,%cpu,%mem,rss,vsize,etime=)
    echo "$MAIN_INFO" | tail -1 | awk '{
        printf "Main Process: PID=%s | CPU=%s%% | RAM=%s%% | RSS=%s | Runtime=%s\n", $1, $3, $4, $5, $7
    }'
    
    echo "System: Free RAM ~${FREE_GB}GB | Load: $(uptime | awk -F'load average:' '{print $2}')"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "ACTIVE WORKER PROCESSES (LokyProcess)"
    echo "────────────────────────────────────────────────────────────────"
    
    # Count active workers
    WORKER_COUNT=$(pgrep -P $MAIN_PID -f "LokyProcess" | wc -l)
    echo "Active Workers: $WORKER_COUNT / 5"
    echo ""
    
    # Show worker details with sorted output
    ps aux | grep "LokyProcess" | grep -v grep | awk '{
        printf "  %s  CPU:%5.1f%%  RAM:%5.1f%%  RSS:%8s  [%s]\n",
        $2, $3, $4, $6/1024"MB", substr($0, index($0, "LokyProcess"), 12)
    }' | sort
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "THREAD ANALYSIS"
    echo "────────────────────────────────────────────────────────────────"
    
    # Get thread count for main process
    if [ -d "/proc/$MAIN_PID" ]; then
        THREAD_COUNT=$(ls -1 /proc/$MAIN_PID/task 2>/dev/null | wc -l)
    elif command -v lsof &> /dev/null; then
        THREAD_COUNT=$(lsof -p $MAIN_PID 2>/dev/null | wc -l)
    else
        THREAD_COUNT="?"
    fi
    
    # Alternative: use ps to count threads
    PS_THREADS=$(ps -p $MAIN_PID -L 2>/dev/null | wc -l)
    
    echo "Main Process Threads: ~$PS_THREADS"
    echo "Total Python Threads System-wide: $(ps aux | grep python | grep -v grep | wc -l)"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "STALE PROCESSES (Previous Runs)"
    echo "────────────────────────────────────────────────────────────────"
    
    STALE=$(ps aux | grep "LokyProcess" | grep -v grep | awk '{print $2}' | while read pid; do
        if ! pgrep -P $MAIN_PID $pid > /dev/null 2>&1; then
            echo $pid
        fi
    done | wc -l)
    
    if [ "$STALE" -gt 0 ]; then
        echo "⚠ WARNING: $STALE stale worker processes detected!"
        echo "These may be from previous runs and could slow down training."
        echo ""
        echo "Stale process details:"
        ps aux | grep "LokyProcess" | grep -v grep | awk '{print $2}' | while read pid; do
            if ! pgrep -P $MAIN_PID $pid > /dev/null 2>&1; then
                RUNTIME=$(ps -p $pid -o etime= 2>/dev/null)
                echo "  PID $pid | Runtime: $RUNTIME"
            fi
        done
        echo ""
        echo "To clean up: killall -9 'python' or restart Terminal"
    else
        echo "✓ No stale processes detected"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "TRAINING PROGRESS"
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -f "$LOGFILE" ]; then
        LAST_LOG=$(tail -3 "$LOGFILE")
        echo "$LAST_LOG"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Press Ctrl+C to exit | Auto-refresh in 5 seconds..."
    echo "════════════════════════════════════════════════════════════════"
    
    sleep 5
done
