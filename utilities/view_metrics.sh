#!/bin/bash

# Professional Metrics Viewer
# Shows comprehensive training metrics and progress in real-time

set -e

LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
LATEST_METRICS=$(ls -t logs/training_*.metrics.json 2>/dev/null | head -1)
LATEST_PROGRESS=$(ls -t logs/training_*.progress 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training logs found"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "TRAINING METRICS DASHBOARD"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Show progress
if [ -f "$LATEST_PROGRESS" ]; then
    echo "📊 CURRENT PROGRESS"
    echo "────────────────────────────────────────────────────────────────────────────────"
    cat "$LATEST_PROGRESS"
    echo ""
fi

# Show metrics
if [ -f "$LATEST_METRICS" ]; then
    echo "📈 DETAILED METRICS"
    echo "────────────────────────────────────────────────────────────────────────────────"
    python3 << 'PYTHON'
import json
import sys
from pathlib import Path
from datetime import datetime

metrics_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not metrics_file or not metrics_file.exists():
    print("Metrics file not found")
    sys.exit(1)

with open(metrics_file) as f:
    metrics = json.load(f)

# Print stages
if 'stages' in metrics:
    print("Training Stages:")
    for stage_name, stage_data in metrics['stages'].items():
        status = stage_data.get('status', 'UNKNOWN')
        elapsed = stage_data.get('elapsed_seconds', 0)
        print(f"  {stage_name}:")
        print(f"    Status: {status}")
        print(f"    Elapsed: {elapsed:.1f}s")
        if 'details' in stage_data:
            for k, v in stage_data['details'].items():
                print(f"    {k}: {v}")

# Print folds
if 'folds' in metrics:
    print(f"\nFolds Completed: {len(metrics['folds'])}")
    for fold in metrics['folds'][-3:]:  # Last 3 folds
        print(f"  {fold['fold_type']} fold {fold['fold_num']}/{fold['total_folds']}")
        if fold.get('metrics'):
            for k, v in fold['metrics'].items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

PYTHON
    python3 - "$LATEST_METRICS" 2>/dev/null || echo "Could not parse metrics"
    echo ""
fi

# Show recent logs
echo "📝 RECENT LOG ENTRIES (last 20 lines)"
echo "────────────────────────────────────────────────────────────────────────────────"
tail -20 "$LATEST_LOG"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "To watch in real-time: tail -f $LATEST_LOG"
echo "To view metrics: view_metrics.sh"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
