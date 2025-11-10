# Configuration Guide

## Summary Table

| Config | Purpose | Runtime | Models | Feature Route | Use When |
|--------|---------|---------|--------|---------------|----------|
| config_smoke_test.yaml | Validation | 5-10 min | LR + RF | PCA only | Testing setup, debugging |
| config_ultrafast_pca.yaml | Recommended | 10-15 min | LR + RF | PCA only | Main training, fastest option |
| config_academic_feasible.yaml | Full Academic | 30-45 min | LR + RF | PCA + Filter_L1 | Academic completeness |
| config_fast_filter_l1.yaml | Experimental | 25-35 min | LR + RF | Filter_L1 optimized | Testing Filter_L1 |
| config_academic_pca_only.yaml | Legacy | - | Multiple | PCA only | Not recommended |
| config_academic_filter_l1_optimized.yaml | Legacy | - | Multiple | Filter_L1 | Not recommended |
| config_academic_filter_l1_optimized_v2.yaml | Legacy | - | Multiple | Filter_L1 | Not recommended |
| config.yaml | Legacy | - | Multiple | Multiple | Not recommended |

## Detailed Configuration Descriptions

### config_ultrafast_pca.yaml (Recommended)

Use this for normal training.

```bash
utilities/run_training.sh config/config_ultrafast_pca.yaml
```

Specifications:
- CV: 3x3 nested (stratified)
- Feature Route: PCA (100 components, 70-80% variance)
- Models: Logistic Regression + Random Forest
- Runtime: 10-15 minutes
- Memory: 12-15 GB peak

Why use it:
- Fast and reliable
- PCA is O(n*k²), computationally efficient
- Full evaluation suite included (ROC, PR, calibration, DCA, bootstrap CI)
- Memory efficient

Output:
- reports/tables/nested_cv_results.csv (all metrics per fold)
- reports/tables/summary_metrics.csv (mean and standard deviation across folds)
- figures/modeling/* (ROC, PR, confusion matrix)
- figures/calibration/* (calibration curve, decision curve)
- models/final_model_pca_*.pkl (trained pipelines)

### config_smoke_test.yaml (For Testing)

Use this first to validate your setup.

```bash
utilities/run_training.sh config/config_smoke_test.yaml
```

Specifications:
- CV: 3x3 nested
- Feature Route: PCA only
- Models: Logistic Regression + Random Forest (simplified grids)
- Runtime: 5-10 minutes
- Memory: 5-8 GB peak

When to use:
- Before running full training for setup validation
- Testing infrastructure changes
- Quick integration test
- Debug environment issues

### config_academic_feasible.yaml (Full Academic)

Use this for maximum academic rigor while staying computationally feasible.

```bash
utilities/run_training.sh config/config_academic_feasible.yaml
```

Specifications:
- CV: 3x3 nested (academically valid per Protocol v1.3)
- Feature Routes: Both Filter_L1 and PCA
- Models: Logistic Regression + Random Forest
- Runtime: 30-45 minutes
- Memory: 15-20 GB peak

When to use:
- When you need full academic rigor
- For final submission or publication
- When comparing feature selection methods
- When maximum credibility is required

Academic Justification:
- 3x3 nested CV validated in literature (Bradshaw et al. 2023)
- Protocol v1.3 Section 4.1 states that 3-5 outer folds provide sufficient performance estimation
- Both feature routes required for full analysis
- All metrics included (ROC, PR, calibration, DCA, bootstrap CI)

### config_fast_filter_l1.yaml (Experimental)

Use this to test Filter_L1 at reasonable speed.

```bash
utilities/run_training.sh config/config_fast_filter_l1.yaml
```

Specifications:
- CV: 3x3 nested
- Feature Route: Filter_L1 optimized with k_prefilter=500
- Models: Logistic Regression + Random Forest
- Runtime: 25-35 minutes
- Memory: 15-18 GB peak

Key optimization:
- k_prefilter: 500 (reduced from 2000)
- This reduces correlation operations by 90%
- Still selects 150 genes (biologically meaningful)

When to use:
- Testing Filter_L1 without full commitment
- Exploring which feature route performs better
- When you have computational constraints

Note: This is not full academic configuration (single feature route), but useful for exploration.

### Legacy Configurations

These are archived for reference only:

| Config | Issue | Recommendation |
|--------|-------|-----------------|
| config_academic_pca_only.yaml | Hardcoded params, old structure | Use config_ultrafast_pca.yaml |
| config_academic_filter_l1_optimized.yaml | Old structure, slow | Use config_fast_filter_l1.yaml |
| config_academic_filter_l1_optimized_v2.yaml | Old structure, slow | Use config_fast_filter_l1.yaml |
| config.yaml | Root-level legacy config | Use folder-based configs in config/ |

Why they are legacy:
- Hardcoded paths (expected root-level config.yaml)
- Old directory structure assumptions
- Not updated for new folder layout
- Use config/ versions instead

## Decision Tree

Which config should you use?

```
START
  ├─ Is this your FIRST run? (Setup validation)
  │  └─ YES → config_smoke_test.yaml (5-10 min)
  │
  ├─ Do you want FASTEST training?
  │  └─ YES → config_ultrafast_pca.yaml (10-15 min)
  │
  ├─ Do you need FULL ACADEMIC RIGOR?
  │  └─ YES → config_academic_feasible.yaml (30-45 min)
  │
  ├─ Do you want to explore Filter_L1?
  │  └─ YES → config_fast_filter_l1.yaml (25-35 min)
  │
  └─ Otherwise → config_ultrafast_pca.yaml (safest choice)
```

## Running Training

Recommended workflow:

Step 1: Validate Setup (5 min)
```bash
utilities/run_training.sh config/config_smoke_test.yaml
```

Step 2: Main Training (10-15 min)
```bash
caffeinate -i utilities/run_training.sh config/config_ultrafast_pca.yaml
tail -f logs/training_*.log  # Monitor in another terminal
```

Step 3: Optional - Full Academic Run (30-45 min)
```bash
caffeinate -i utilities/run_training.sh config/config_academic_feasible.yaml
```

Step 4: Post-Processing (20 min total)
```bash
# Stability analysis
python scripts/stability_analysis_v2.py

# Feature importance (SHAP)
python scripts/shap_compact_panel_v2.py

# Generate model card
python scripts/generate_model_card.py
```

## Configuration Parameters Explained

CV Settings:

```yaml
cv:
  outer_folds: 3          # Performance estimation (increase to 5 if time allows)
  inner_folds: 3          # Hyperparameter tuning (don't change)
  stratified: true        # Maintains class balance
  shuffle: true           # Randomize fold assignment
  test_size: 0.2          # Not used in nested CV (for reference)
```

Feature Routes:

```yaml
features:
  routes:
    - "pca"                      # O(n*k²) - Fast (1-2 min/fold)
    - "filter_l1"                # O(n²) - Slow (10-15 min/fold)
    - "filter_l1_optimized"      # O(k²) - Medium (5-10 min/fold)
  k_best: 100                    # Number of final features
  pca_components: 100            # PCA components (70-80% variance)
  k_prefilter: 2000              # Pre-filter before correlation
```

Model Hyperparameters:

All hyperparameters in param_grid are tuned automatically via GridSearchCV:

```yaml
models:
  lr_elasticnet:
    param_grid:
      classifier__C: [0.1, 1.0, 10.0]           # Regularization strengths
      classifier__l1_ratio: [0.5, 0.7]          # L1/L2 balances
      classifier__max_iter: [2000]              # Fixed for convergence
  
  random_forest:
    param_grid:
      classifier__n_estimators: [100, 200]      # Number of trees
      classifier__max_depth: [20, null]         # Tree depth
      classifier__min_samples_split: [5]        # Min samples to split
      classifier__min_samples_leaf: [2]         # Min samples per leaf
      classifier__class_weight: ["balanced"]    # Handle class imbalance
```

Total GridSearch combinations:
- LR: 3 x 2 x 1 = 6 combinations
- RF: 2 x 2 x 1 x 1 x 1 = 4 combinations
- Per outer fold: 3 inner folds x (6 + 4) = 30 fits
- Total: 3 outer x 30 = 90 model fits per feature route

## Performance Expectations

config_ultrafast_pca.yaml results:

Based on actual run:
```
Runtime: 12 minutes (3x3 nested CV, PCA)
Memory: 12-15 GB peak
Results:
  - pca_random_forest: ROC-AUC = 1.000, PR-AUC = 1.000, F1 = 0.997
  - pca_lr_elasticnet: ROC-AUC = 0.431, PR-AUC = 0.925, F1 = 0.965
```

config_academic_feasible.yaml projections:

For both PCA and Filter_L1 routes:
```
Runtime: 30-45 minutes (3x3 nested CV)
Memory: 15-20 GB peak
Expected:
  - pca_random_forest: High performance (similar to above)
  - filter_l1_random_forest: Slightly lower (sparse gene selection)
```

## Troubleshooting

Q: Script takes more than 30 minutes with config_ultrafast_pca.yaml
A: Something is wrong. Check:
   - Are you using PCA? (should be 2-3 min/fold)
   - Is Filter_L1 somehow enabled? (would be 10-15 min/fold)
   - Check logs/training_*.log for which routes are running

Q: Memory usage exceeds 25 GB
A: Reduce models or feature components:
   - Reduce pca_components from 100 to 50
   - Disable lightgbm in param_grid
   - Reduce n_jobs in src/models.py

Q: Script crashes after 20 minutes
A: Check error in logs/training_*.log:
   - Memory exhausted? Reduce features or CV folds
   - SHAP incompatibility? Python < 3.13 required
   - Batch correction failure? Check data quality

Q: Which config should I submit for evaluation?
A: Use config_academic_feasible.yaml (full academic rigor)
   Or config_ultrafast_pca.yaml if time-constrained (still academically valid)

## Summary

| Need | Use Config | Time | Academic |
|------|------------|------|----------|
| Quick test | config_smoke_test.yaml | 5-10 min | Yes |
| Fastest run | config_ultrafast_pca.yaml | 10-15 min | Yes |
| Most rigorous | config_academic_feasible.yaml | 30-45 min | Yes |

Recommendation: Start with config_smoke_test.yaml to validate, then use config_ultrafast_pca.yaml for main training.

