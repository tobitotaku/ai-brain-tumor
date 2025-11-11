#!/bin/bash
# Quick Commands Reference for Compact Full Run

echo "=================================================="
echo "Compact Full GBM Run - Quick Reference"
echo "=================================================="
echo ""

cat << 'EOF'
🎯 CURRENT STATUS
==================

Training Status:
  ./utilities/monitor_compact_full.sh

Live Log:
  tail -f logs/training_*.log

Process Info:
  ps aux | grep train_cv.py | grep -v grep


📊 WHEN TRAINING COMPLETES
===========================

Automated Workflow (Recommended):
  ./utilities/post_training_workflow.sh

Manual Steps:
  1. Permutation test:
     python scripts/run_permutation_test.py
  
  2. Generate report:
     python scripts/generate_compact_full_report.py
  
  3. Bundle artifacts:
     ./utilities/bundle_artifacts.sh


🔍 VIEW RESULTS
================

Quick Summary:
  cat reports/tables/summary_metrics.csv

Detailed Metrics:
  cat reports/tables/nested_cv_results.csv

Full Report:
  cat reports/COMPACT_FULL_REPORT.md


🛠️ TROUBLESHOOTING
===================

Kill Training:
  pkill -9 -f "train_cv.py"
  rm -f .training_pid

Restart Training:
  ./utilities/run_training.sh config/config_compact_full.yaml

Clean Logs:
  rm -f logs/training_*.log logs/training_*.err


📁 OUTPUT LOCATIONS
====================

Tables:        reports/tables/*.csv
Figures:       figures/{modeling,calibration,shap}/*.png
Models:        models/*.pkl
Predictions:   outputs/*.csv
Archive:       artifacts/compact_full_*.zip
Report:        reports/COMPACT_FULL_REPORT.md


⚙️ CONFIGURATION
=================

Current Config:
  cat config/config_compact_full.yaml

Edit Config:
  code config/config_compact_full.yaml


🎓 SCALE TO ACADEMIC FULL
==========================

Key differences:
  Compact:   3x2 CV, 80 PCA, 400 prefilter, 120 final, 400 trees
  Academic:  5x3 CV, 120 PCA, 1000 prefilter, 200 final, 800 trees

Edit config/config_compact_full.yaml:
  - cv.outer_folds: 3 → 5
  - cv.inner_folds: 2 → 3
  - features.pca_components: 80 → 120
  - features.k_prefilter: 400 → 1000
  - features.k_final: 120 → 200
  - models.random_forest.n_estimators: 400 → 800


📚 DOCUMENTATION
=================

Setup Guide:           COMPACT_FULL_SETUP.md
Configuration Guide:   CONFIGURATION_GUIDE.md
Setup Instructions:    SETUP.md
Protocol:              docs/Protocol.md


🚀 EXPECTED TIMELINE
=====================

Training (4 combinations):
  - PCA + LR:         ~2-3 min
  - PCA + RF:         ~3-5 min
  - Filter_L1 + LR:   ~5-8 min
  - Filter_L1 + RF:   ~8-12 min
  Total:              ~20-30 min

Post-Processing:
  - Permutation:      ~5-10 min
  - Report:           ~1 min
  - Bundle:           ~1 min


✅ QUALITY CHECKS
==================

Before Accepting Results:
  1. All 4 combinations trained? (check nested_cv_results.csv)
  2. Permutation ROC ≈ 0.5?       (baseline check)
  3. Permutation PR ≈ 0.93?       (prevalence check)
  4. No train>>val gap?           (overfitting check)
  5. Bootstrap CIs calculated?    (uncertainty quantification)


🎯 SUCCESS CRITERIA
====================

✅ All 4 pipelines completed
✅ Nested CV results saved
✅ Summary metrics with CI
✅ Permutation test ≈ baseline
✅ Final models saved (.pkl)
✅ Report generated
✅ Artifacts bundled


📞 HELP & SUPPORT
==================

Training stuck?
  1. Check logs: tail -50 logs/training_*.log
  2. Check resources: top -o cpu
  3. Check memory: vm_stat

Out of memory?
  Reduce: pca_components, k_prefilter, k_final in config

Training failed?
  Check stderr: cat logs/training_stderr.log

EOF

echo ""
echo "=================================================="
echo "For full documentation: cat COMPACT_FULL_SETUP.md"
echo "=================================================="
