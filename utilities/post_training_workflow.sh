#!/bin/bash
# Post-training workflow automation
# Runs after training completes: permutation test, report generation, artifact bundling

set -e

echo "=================================================="
echo "Post-Training Workflow - Compact Full Run"
echo "=================================================="
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if training is complete
if ps aux | grep "train_cv.py.*config_compact_full" | grep -v grep > /dev/null; then
    echo "⚠️  WARNING: Training is still running!"
    echo ""
    echo "Options:"
    echo "  1. Wait for training to complete, then run this script again"
    echo "  2. Run this script anyway (partial results)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Run this script after training completes."
        exit 1
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

echo ""
echo "=================================================="
echo "STEP 1: Verify Training Results"
echo "=================================================="

if [ ! -f "reports/tables/nested_cv_results.csv" ]; then
    echo "❌ ERROR: No training results found!"
    echo "   Expected: reports/tables/nested_cv_results.csv"
    exit 1
fi

# Count completed model combinations
COMBINATIONS=$(tail -n +2 reports/tables/nested_cv_results.csv | cut -d',' -f1,2 | sort -u | wc -l | tr -d ' ')
echo "✓ Found $COMBINATIONS / 4 model combinations"

if [ "$COMBINATIONS" -lt 4 ]; then
    echo "⚠️  Warning: Only $COMBINATIONS combinations found (expected 4)"
    echo "   Training may not be complete"
fi

echo ""
tail -n +2 reports/tables/nested_cv_results.csv | cut -d',' -f1,2 | sort -u | sed 's/^/  ✓ /'

echo ""
echo "=================================================="
echo "STEP 2: Run Permutation Sanity Check"
echo "=================================================="

read -p "Run permutation test? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "Running permutation test with 20 permutations..."
    echo "(This will take ~5-10 minutes)"
    echo ""
    
    python scripts/run_permutation_test.py \
        --config config/config_compact_full.yaml \
        --n-permutations 20 \
        --feature-route pca \
        --model random_forest
    
    echo ""
    echo "✓ Permutation test complete"
    echo "  Check: figures/modeling/permutation_*.png"
    echo "  Check: reports/tables/permutation_summary_*.csv"
else
    echo "⏭  Skipped permutation test"
fi

echo ""
echo "=================================================="
echo "STEP 3: Generate Analysis Report"
echo "=================================================="

python scripts/generate_compact_full_report.py

if [ -f "reports/COMPACT_FULL_REPORT.md" ]; then
    echo ""
    echo "✓ Report generated successfully"
    echo "  Location: reports/COMPACT_FULL_REPORT.md"
    echo ""
    echo "Preview:"
    head -30 reports/COMPACT_FULL_REPORT.md
else
    echo "❌ ERROR: Report generation failed"
fi

echo ""
echo "=================================================="
echo "STEP 4: Bundle Artifacts"
echo "=================================================="

./utilities/bundle_artifacts.sh

echo ""
echo "=================================================="
echo "STEP 5: Summary"
echo "=================================================="

echo ""
echo "Training Results:"
echo "-----------------"
if [ -f "reports/tables/summary_metrics.csv" ]; then
    echo ""
    cat reports/tables/summary_metrics.csv | head -6
fi

echo ""
echo "Files Generated:"
echo "----------------"
echo "  Models:      $(ls models/*.pkl 2>/dev/null | wc -l) files"
echo "  Tables:      $(ls reports/tables/*.csv 2>/dev/null | wc -l) files"
echo "  Figures:     $(find figures -name "*.png" 2>/dev/null | wc -l) files"
echo "  Archive:     $(ls -lh artifacts/compact_full_*.zip 2>/dev/null | tail -1 | awk '{print $5}')"

echo ""
echo "Quick Access:"
echo "-------------"
echo "  📊 Report:    reports/COMPACT_FULL_REPORT.md"
echo "  📁 Archive:   $(ls -t artifacts/compact_full_*.zip 2>/dev/null | head -1)"
echo "  📈 Metrics:   reports/tables/summary_metrics.csv"
echo "  🤖 Models:    models/*.pkl"

echo ""
echo "=================================================="
echo "Post-Training Workflow Complete!"
echo "=================================================="
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Next steps:"
echo "  1. Review: reports/COMPACT_FULL_REPORT.md"
echo "  2. Check permutation test results (should show baseline performance)"
echo "  3. If satisfied, scale up to academic full run"
echo "  4. Archive shared at: $(ls -t artifacts/compact_full_*.zip 2>/dev/null | head -1)"
echo ""
