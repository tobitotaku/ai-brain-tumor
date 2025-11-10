#!/bin/bash

# Cleanup stale processes from previous training runs

echo "Scanning for stale Python/LokyProcess workers..."
echo ""

# Find all LokyProcess and resource_tracker processes
ps aux | grep -E "LokyProcess|resource_tracker" | grep -v grep | awk '{print $2}' > /tmp/worker_pids.txt

if [ ! -s /tmp/worker_pids.txt ]; then
    echo "✓ No stale processes found"
    rm -f /tmp/worker_pids.txt
    exit 0
fi

echo "Found potential stale processes:"
while read pid; do
    RUNTIME=$(ps -p $pid -o etime= 2>/dev/null | xargs)
    CPU=$(ps -p $pid -o %cpu= 2>/dev/null | xargs)
    echo "  PID $pid | Runtime: $RUNTIME | CPU: $CPU%"
done < /tmp/worker_pids.txt

echo ""
read -p "Kill these processes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    while read pid; do
        kill -9 $pid 2>/dev/null
    done < /tmp/worker_pids.txt
    echo "✓ Stale processes killed"
    sleep 1
    echo "Verifying cleanup..."
    REMAINING=$(ps aux | grep -E "LokyProcess|resource_tracker" | grep -v grep | wc -l)
    
    if [ "$REMAINING" -eq 0 ]; then
        echo "✓ All stale processes cleaned up"
    else
        echo "⚠ $REMAINING processes still running"
    fi
else
    echo "Cancelled"
fi

rm -f /tmp/worker_pids.txt

