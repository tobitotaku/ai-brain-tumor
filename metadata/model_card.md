# GBM Classification Model Card

## Model Description

### Model Summary
This is a machine learning model for binary classification of Glioblastoma (GBM) samples versus controls using gene expression data. The model was developed as part of the "AI in Healthcare" minor at Hogeschool Rotterdam.

**Model Name:** GBM Gene Expression Classifier  
**Version:** 1.0  
**Date:** November 2025  
**Framework:** scikit-learn

### Model Architecture
- **Type:** Ensemble comparison (Logistic Regression, Random Forest, LightGBM)
- **Input:** 18,752 normalized gene expression values
- **Output:** Binary prediction (0 = Healthy, 1 = GBM) + probability scores
- **Features:** 200 genes (filter_l1) or 200 PCA components

### Pipeline Components
1. **Feature Selection:**
   - Variance filtering (threshold: 0.01)
   - Correlation filtering (threshold: 0.95)
   - L1 regularization or PCA dimensionality reduction

2. **Batch Correction:** ComBat
   - Applied only on training folds (no leakage)

3. **Scaling:** StandardScaler
   - Fitted on training data only

4. **Classifier:** Multiple models compared (Logistic Regression, Random Forest, LightGBM)
   - Hyperparameters optimized via nested cross-validation

---

## Intended Use

### Primary Intended Uses
**Appropriate Uses:**
- Research tool for GBM biomarker discovery
- Exploratory analysis of gene expression patterns
- Educational demonstration of ML in healthcare
- Hypothesis generation for further clinical validation

### Out-of-Scope Uses
**Inappropriate Uses:**
- **Direct clinical diagnosis** - Model has not been clinically validated
- **Treatment decisions** - Not approved for clinical decision-making
- **Standalone screening tool** - Requires physician interpretation
- **Application to other cancer types** - Trained specifically for GBM

### User Warnings
**Important Limitations:**
- This model is a research prototype and has NOT been approved for clinical use
- Predictions should be interpreted by qualified healthcare professionals
- Model performance may vary on external datasets
- Batch effects must be properly corrected before applying to new data

---

## Training Data

### Dataset
- **Source:** Combined gene expression dataset from public repositories
- **Samples:** 18,635 total samples
  - Training: Full dataset via 5-fold CV
  - Class distribution: 93.3% Healthy (17,382), 6.7% GBM (1,253)
- **Features:** 18,752 genes after quality filtering (from original 18,858)
- **Format:** ENSEMBL gene IDs with normalized expression values

### Data Preprocessing
- Variance filtering (threshold: 0.01)
- Duplicate removal
- Batch effect correction (ComBat/Harmony)
- Standard scaling
- See `data_card.md` for full details

---

## Evaluation

### Evaluation Strategy
- **Method:** Nested Cross-Validation (5 outer × 5 inner folds)
- **Metrics:** ROC-AUC, PR-AUC, F1-score, Accuracy, Precision, Recall
- **Confidence Intervals:** Bootstrap with 1000 iterations (95% CI)

### Performance Metrics
Performance on held-out cross-validation folds:

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | TBD | TBD |
| PR-AUC | TBD | TBD |
| F1-Score | TBD | TBD |
| Accuracy | TBD | TBD |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| Specificity | TBD | TBD |

**Note:** Values to be populated after model training completion

### Calibration
- **Brier Score:** TBD
- **Expected Calibration Error (ECE):** TBD
- Model calibration assessed via reliability diagrams
- See `figures/calibration/` for calibration curves

### Decision Curve Analysis
- Net benefit analysis performed across threshold range 0.0-1.0
- See `figures/calibration/decision_curve_best.png`

---

## Model Explainability

### Feature Importance
Top 10 most important features (by SHAP values):

**Note:** Feature rankings to be populated after SHAP analysis completion

*Full feature importance available in `reports/tables/shap_summary.csv`*

### SHAP Analysis
- SHAP (SHapley Additive exPlanations) values computed for all predictions
- Visualizations available in `figures/shap/`:
  - Beeswarm plot (feature importance + direction)
  - Bar plot (mean absolute SHAP values)
  - Waterfall plots (individual sample explanations)
  - Dependence plots (feature interactions)

### Biological Interpretation
Biological relevance of identified features will be assessed through:
- Association with known GBM pathways
- Literature support for identified biomarkers
- Potential therapeutic targets

---

## Limitations & Biases

### Model Limitations

1. **Generalizability:**
   - Trained on specific cohort(s) - may not generalize to:
     - Different sequencing platforms
     - Different patient populations
     - Different tumor grades or subtypes
   - External validation required before clinical use

2. **Sample Size:**
   - Limited training data may affect rare subtype detection
   - Confidence intervals reflect this uncertainty

3. **Feature Selection:**
   - Model relies on gene panel that may miss novel biomarkers
   - Feature stability assessed but some variability expected

4. **Technical:**
   - Requires proper batch correction for new data
   - Sensitive to data preprocessing choices
   - Performance may degrade with low-quality samples

### Known Biases

1. **Data Bias:**
   - Training data sourced from TCGA and GTEx repositories
   - May not represent diverse patient demographics
   - Class imbalance (93.3% healthy, 6.7% GBM) may favor majority class

2. **Selection Bias:**
   - Training data from specific research cohorts
   - May not represent general GBM population

3. **Measurement Bias:**
   - Gene expression platform-specific effects
   - Batch effects partially addressed but residual effects possible

### Failure Cases
Model may perform poorly when:
- Batch effects are not properly corrected
- Input data quality is poor (high missing values, outliers)
- Applied to mixed-grade gliomas without GBM-specific training
- Samples from significantly different populations than training data

---

## Ethical Considerations

### Fairness & Equity
- **Demographic Analysis:** Patient demographics not available in aggregated dataset
- **Access:** Model predictions should not determine access to care
- **Bias Mitigation:** Batch correction applied to reduce technical bias

### Privacy
- Model does not store patient data
- All training data de-identified
- Predictions should be handled according to healthcare privacy regulations

### Clinical Impact
**Critical Warnings:**
- **NOT approved for clinical diagnosis**
- **NOT a substitute for physician judgment**
- **NOT validated for treatment selection**
- Should only be used as a research/decision-support tool

### Dual Use Concerns
- No identified dual-use concerns
- Model is designed solely for medical research

---

## Reproducibility

### Random Seeds
- Global random seed: 42
- Set for all random operations (CV splits, bootstrap, etc.)

### Dependencies
See `environment.yml` and `requirements.txt` for full environment specification.

Key dependencies:
- Python 3.9+
- scikit-learn
- pandas
- numpy
- lightgbm/xgboost (if applicable)
- shap
- matplotlib, seaborn

### Training Process
Full training can be reproduced via:
```bash
# 1. Preprocess data
python scripts/make_processed.py --config config.yaml

# 2. Train models with nested CV
python scripts/train_cv.py --config config.yaml

# 3. Generate SHAP explainability report
python scripts/shap_report.py --config config.yaml
```

### Configuration
All hyperparameters and settings stored in `config.yaml`

---

## Model Card Authors & Contact

**Affiliation:**
- Hogeschool Rotterdam
- Minor: AI in Healthcare
- Project: GBM Classification Retake

**Contact:**
- GitHub: https://github.com/tobitotaku/ai-brain-tumor

**Reviewers:**
- Hogeschool Rotterdam faculty (Minor AI in Healthcare)

---

## Model Updates & Versioning

### Version History
- **v1.0** (November 2025): Initial model release
  - Baseline performance established
  - Nested CV evaluation complete

### Future Improvements
Planned enhancements:
- External validation on independent cohorts
- Incorporation of additional clinical features
- Ensemble methods combining multiple feature selection strategies
- Subtype-specific models
- Integration with multi-modal data (imaging, clinical)

---

## References & Citations

### Citation
If you use this model, please reference:

```
GBM Gene Expression Classifier
Hogeschool Rotterdam Minor AI in Healthcare, 2025
GitHub: https://github.com/tobitotaku/ai-brain-tumor
```

### Related Work
See `docs/Protocol.md` for complete literature references on:
- GBM classification using gene expression
- Machine learning in oncology
- Model explainability and interpretability

---

## Additional Resources

- **Code Repository:** https://github.com/tobitotaku/ai-brain-tumor
- **Data Card:** See `metadata/data_card.md`
- **Documentation:** See `README.md`
- **Results:** See `reports/tables/`
- **Visualizations:** See `figures/`

---

**Model Card Version:** 1.0  
**Last Updated:** November 9, 2025  
**Status:** Research Prototype - Not for Clinical Use
