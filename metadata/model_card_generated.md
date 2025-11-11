# Model Card: GBM Classification Pipeline
*Auto-generated from training results*
**Generated:** 2025-11-11 22:31:03
---

## Model Details

**Project:** GBM Classification
**Developer:** Hogeschool Rotterdam Team
**Institution:** Hogeschool Rotterdam - Minor AI in Healthcare
**Date:** November 2025
**Best Model:** pca_random_forest

### Model Architecture

The pipeline consists of:

1. **Feature Selection:** FilterL1 (k=500: variance → correlation → L1 regularization) OR PCA (n=200 components)
2. **Preprocessing:** ComBat batch correction → Standard scaling
3. **Classifiers:** random_forest, lr_elasticnet

## Intended Use

**Primary Use Case:** Binary classification of gene expression profiles to distinguish between:
- Healthy brain tissue samples
- Glioblastoma Multiforme (GBM) tumor samples

**Intended Users:** Bioinformatics researchers, clinical research teams

**Out-of-Scope Use Cases:**
- Direct clinical diagnosis without expert review
- Application to other cancer types
- Use with different sequencing platforms without validation

## Training Data

**Source:** Combined gene expression dataset (18,635 samples × 18,858 genes)

**Dataset Size:** 18,635 samples
**Features:** 18,858 genes (ENSEMBL IDs)
**Class Distribution:** ~93% healthy, ~7% tumor (imbalanced)

**Preprocessing Steps:**
1. Low-variance gene removal (threshold=0.01)
2. High-correlation filtering (threshold=0.95)
3. Batch effect correction (ComBat)
4. Feature scaling (StandardScaler)

## Evaluation

**Validation Strategy:** Nested stratified cross-validation
- Outer folds: 3 (performance estimation)
- Inner folds: 2 (hyperparameter tuning)
- Hold-out test set: 20%

### Performance Metrics

**Model Comparison (Nested CV):**

| Model | ROC-AUC | PR-AUC | F1 | Accuracy |
|-------|---------|--------|-----|----------|
| random_forest | 1.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 1.000 | 1.000 | 0.998 | 0.996 |
| lr_elasticnet | 0.667 | 0.929 | 0.946 | 0.902 |
| lr_elasticnet | 0.443 | 0.918 | 0.619 | 0.465 |

**Best Model Detailed Performance (pca_random_forest):**

| Metric | Value (95% CI) |
|--------|----------------|
| Roc-Auc | 1.000 (95% CI: 1.000-1.000) |
| Pr-Auc | 1.000 (95% CI: 1.000-1.000) |
| Accuracy | 0.994 |
| Precision | 0.994 |
| Recall | 1.000 |
| F1 | 0.997 |
| Specificity | 0.916 |

### Calibration Analysis

Model predictions are calibrated using isotonic regression to ensure predicted probabilities accurately reflect true outcome frequencies.

## Limitations

1. **Class Imbalance:** Dataset is heavily skewed (~93% healthy), requiring balanced weighting
2. **Generalization:** Performance on external datasets not yet validated
3. **Batch Effects:** Model assumes batch correction adequately removes technical variation
4. **Feature Interpretation:** PCA components lack direct biological interpretability
5. **Computational Cost:** Filter-L1 route requires significant compute time for large gene sets

## Ethical Considerations

**Fairness:**
- No patient demographics available to assess fairness across subgroups
- Class imbalance may bias model toward healthy class predictions

**Privacy:**
- Gene expression data is de-identified
- No linkage to individual patient records

**Clinical Impact:**
- Model is NOT approved for clinical use
- Predictions must be validated by trained professionals
- False negatives (missing tumors) have severe clinical consequences

## Recommendations

**For Research Use:**
- Validate on independent external cohorts
- Perform biological pathway analysis on stable genes
- Investigate decision boundaries and misclassified samples

**For Future Improvement:**
- Collect more tumor samples to balance dataset
- Integrate multi-omic data (methylation, CNV, mutations)
- Develop ensemble methods combining multiple feature selection routes
- External validation on TCGA-GBM and other public datasets

## Citation

```
@misc{hogeschool2025gbm,
  author = {Hogeschool Rotterdam Team},
  title = {GBM Classification Pipeline: Gene Expression Analysis},
  year = {2025},
  institution = {Hogeschool Rotterdam},
  note = {Minor AI in Healthcare - Retake Project}
}
```

