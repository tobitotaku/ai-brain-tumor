# Compact Full GBM Training Run - Setup Complete

## 🎯 Overview

This setup trains **4 model-feature combinations** using **nested 3×2 cross-validation**:

| Feature Route | Model | Description |
|---------------|-------|-------------|
| PCA | Logistic Regression (ElasticNet) | Linear baseline |
| PCA | Random Forest | Non-linear ensemble |
| Filter_L1 | Logistic Regression (ElasticNet) | Sparse feature selection |
| Filter_L1 | Random Forest | Feature importance + stability |

## 📁 Files Created

### Configuration
- `config/config_compact_full.yaml` - Main training configuration with all 4 combinations

### Scripts
- `scripts/run_permutation_test.py` - Sanity check with shuffled labels
- `scripts/generate_compact_full_report.py` - Comprehensive markdown report generator

### Utilities
- `utilities/run_training.sh` - Updated with CPU optimization (multi-threading)
- `utilities/monitor_compact_full.sh` - Real-time training progress monitor
- `utilities/bundle_artifacts.sh` - Bundle all outputs into timestamped archive

### Updates
- `src/features.py` - Updated to support `k_final` parameter for Filter_L1

## 🚀 Usage

### 1. Monitor Training Progress

The training is currently running. Monitor with:

```bash
# Quick status check
./utilities/monitor_compact_full.sh

# Follow live log
tail -f logs/training_*.log

# Watch CPU usage
watch -n 5 'ps aux | grep train_cv.py | grep -v grep'
```

### 2. When Training Completes

Check results:
```bash
# View summary metrics
cat reports/tables/summary_metrics.csv

# View detailed CV results
cat reports/tables/nested_cv_results.csv
```

### 3. Run Permutation Sanity Check

After training completes, verify no data leakage:

```bash
source .venv/bin/activate
python scripts/run_permutation_test.py --config config/config_compact_full.yaml --n-permutations 20

# Expected results with random labels:
# - ROC-AUC ≈ 0.5 (random chance)
# - PR-AUC ≈ 0.93 (class prevalence)
```

### 4. Generate Analysis Report

```bash
python scripts/generate_compact_full_report.py
# Creates: reports/COMPACT_FULL_REPORT.md
```

### 5. Bundle All Artifacts

```bash
./utilities/bundle_artifacts.sh
# Creates: artifacts/compact_full_<timestamp>.zip
```

## 📊 Expected Outputs

### Tables (`reports/tables/`)
- `nested_cv_results.csv` - Per-fold metrics for all 4 combinations
- `summary_metrics.csv` - Mean ± SD across folds
- `summary_metrics.tex` - LaTeX table for academic papers

### Figures (`figures/`)
- `modeling/` - ROC curves, PR curves, confusion matrices
- `calibration/` - Calibration plots
- `shap/` - Feature importance (if enabled)

### Models (`models/`)
- `final_model_pca_random_forest.pkl` - Best PCA+RF model
- `final_model_pca_lr_elasticnet.pkl` - Best PCA+LR model
- `final_model_filter_l1_random_forest.pkl` - Best Filter_L1+RF model
- `final_model_filter_l1_lr_elasticnet.pkl` - Best Filter_L1+LR model

### Outputs (`outputs/`)
- Out-of-fold predictions
- Probability estimates
- Feature importance scores

## 🔍 Current Training Status

**Progress:** 2/4 model combinations complete (50%)

✅ Completed:
- `pca + lr_elasticnet`
- `pca + random_forest`

🔄 In Progress:
- `filter_l1 + lr_elasticnet` (Outer fold 1/3)

⏳ Pending:
- `filter_l1 + random_forest`

## ⚙️ Configuration Highlights

### Compact Settings (Fast)
- **Outer folds:** 3 (vs 5 for academic)
- **Inner folds:** 2 (vs 3 for academic)
- **PCA components:** 80 (vs 120-200 for academic)
- **Filter_L1 k_prefilter:** 400 (vs 1000-2000 for academic)
- **Filter_L1 k_final:** 120 (vs 150-300 for academic)
- **RF n_estimators:** 400 (vs 800 for academic)
- **Bootstrap CI:** 1000 iterations (vs 2000-5000 for academic)

### CPU Optimization
- Multi-threaded execution using all logical cores
- Environment variables set: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Joblib backend: `loky` for robust parallelization

## 🎓 Scaling to Academic Full Run

When ready for the full academic run, update these values in config:

```yaml
cv:
  outer_folds: 5  # Increase from 3
  inner_folds: 3  # Increase from 2

features:
  pca_components: 120  # Increase from 80
  k_prefilter: 1000    # Increase from 400
  k_final: 200         # Increase from 120

models:
  random_forest:
    param_grid:
      classifier__n_estimators: [800]  # Increase from 400

evaluation:
  bootstrap_ci:
    n_bootstrap: 2000  # Increase from 1000
```

## ✅ Quality Checks

### Anti-Leakage Measures
- ✅ All preprocessing inside Pipeline
- ✅ Nested cross-validation
- ✅ Feature selection per fold
- ✅ No data touching before CV split

### Validation Steps
1. **Permutation test** - Verify performance not due to chance
2. **Bootstrap CI** - Quantify uncertainty
3. **Calibration** - Check probability estimates
4. **Cross-validation consistency** - Verify stable performance

## 📞 Troubleshooting

### Training Stuck?
```bash
# Check if process is alive
ps aux | grep train_cv.py

# Check recent activity
tail -50 logs/training_*.log

# Check system resources
top -o cpu | head -20
```

### Out of Memory?
Edit `config/config_compact_full.yaml`:
```yaml
features:
  pca_components: 50       # Reduce from 80
  k_prefilter: 200         # Reduce from 400
  k_final: 80              # Reduce from 120
```

### Training Failed?
```bash
# Check stderr
cat logs/training_stderr.log

# Restart from scratch
pkill -f train_cv.py
rm -f .training_pid
./utilities/run_training.sh config/config_compact_full.yaml
```

## 📈 Performance Expectations

### Realistic Expectations
- **PCA + Random Forest:** ROC-AUC 0.95-0.99 (strong performance)
- **PCA + LogisticRegression:** ROC-AUC 0.80-0.95 (good performance)
- **Filter_L1 + Random Forest:** ROC-AUC 0.95-0.99 (strong + interpretable)
- **Filter_L1 + LogisticRegression:** ROC-AUC 0.85-0.95 (interpretable baseline)

### Warning Signs
- **ROC-AUC = 1.000** - Check for leakage or overfitting
- **PR-AUC = 1.000** - Likely degenerate solution
- **Train >> Val** - Overfitting detected
- **Permutation ROC > 0.55** - Possible data leakage

## 🎯 Next Steps After Training

1. **Verify completion:** Check all 4 models trained
2. **Run permutation test:** Sanity check for leakage
3. **Generate report:** Comprehensive analysis
4. **Bundle artifacts:** Archive for sharing/backup
5. **Calibration analysis:** Check probability estimates
6. **SHAP analysis:** Interpret best model (optional)
7. **External validation:** Test on holdout if available

## 📦 Artifact Archive Contents

The bundled archive will contain:
```
compact_full_<timestamp>.zip
├── models/              # Trained .pkl files
├── reports/
│   ├── tables/          # CSV metrics
│   └── COMPACT_FULL_REPORT.md
├── figures/
│   ├── modeling/        # ROC, PR, confusion matrices
│   ├── calibration/     # Calibration plots
│   └── shap/            # Feature importance
├── outputs/             # Predictions, probabilities
├── logs/                # Training logs
├── config/              # Configuration used
└── MANIFEST.txt         # Contents listing
```

---

**Created:** 2025-11-11
**Status:** Training in progress (2/4 complete)
**Expected completion:** ~10-15 minutes for all 4 combinations
