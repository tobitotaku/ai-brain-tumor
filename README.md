# GBM Classification from Gene Expression

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## Project Overview

This project implements a comprehensive machine learning pipeline for Glioblastoma (GBM) classification using gene expression data. Developed for the Minor AI in Healthcare at Hogeschool Rotterdam (Capstone Project - Retake), this work demonstrates best practices in:

- Leak-free preprocessing with proper fit/transform separation
- Nested cross-validation (3x3 or 5x3) for unbiased performance estimation  
- Multiple feature selection strategies (filter+L1, PCA)
- Model comparison (Logistic Regression, Random Forest, LightGBM)
- Comprehensive evaluation (ROC-AUC, PR-AUC, calibration, decision curves)
- Feature stability analysis via bootstrap resampling
- Compact gene panel development for clinical deployment
- Ethical AI documentation (data cards, model cards)
- Full reproducibility via configuration management and random seed control

**Research Protocol:** See `docs/Protocol.md` for complete academic documentation  
**Training Status:** See `TRAINING_STATUS.md` for current progress and results  
**Configuration Guide:** See `CONFIGURATION_GUIDE.md` for details on all available configurations

---

## Configuration Selection

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed guide on all available configurations.

**Quick Reference:**
- **First time:** `config_smoke_test.yaml` (5-10 min, validates setup)
- **Recommended:** `config_ultrafast_pca.yaml` (10-15 min, fast PCA-based training)
- **Full academic:** `config_academic_feasible.yaml` (30-45 min, complete protocol)

---

## Computationally Feasible Configuration

**Problem:** Original 5x3 nested CV training requires 2-3 hours, limiting team accessibility.

**Solution:** `config_academic_feasible.yaml` - A protocol-compliant configuration that:
- Maintains full academic rigor (nested CV, both feature routes, all metrics)
- Runs in 30-45 minutes instead of 2-3 hours
- Documented in Protocol v1.3 with literature support
- Includes all 5 academic enhancements

**Key Changes:**
- 3x3 nested CV (academically valid per Bradshaw et al. 2023)
- 100 features instead of 200 (core genes preserved)
- Focused hyperparameter grids (6 LR combos, 12 RF combos)
- LightGBM disabled (Random Forest covers tree-based models)

**See `FEASIBILITY_SOLUTION.md` for complete explanation and academic defense.**

---

## Key Features

### Methodology
- **Data Understanding**: Quality checks, EDA, class distribution analysis
- **Preprocessing**: Batch correction (ComBat/Harmony), variance filtering, scaling
- **Feature Selection**: Three routes (filter+L1, PCA, bio panel) + stability selection
- **Model Training**: Nested CV (5×5) with hyperparameter tuning
- **Evaluation**: ROC-AUC, PR-AUC, F1, calibration curves, decision curves
- **Explainability**: SHAP values, feature importance, single-sample explanations

### Technical Stack
- **ML Framework**: scikit-learn, LightGBM, XGBoost
- **Preprocessing**: ComBat (batch correction), StandardScaler
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML-based pipeline configuration

---

## Project Structure

```
.
├── config/                     # Configuration files
│   ├── config_ultrafast_pca.yaml           # ⭐ Recommended: Fast PCA
│   ├── config_fast_filter_l1.yaml          # Experimental: Filter_L1
│   ├── config_academic_feasible.yaml       # Academic config (30-45 min)
│   └── config_smoke_test.yaml              # Quick test (5-10 min)
│
├── utilities/                  # Helper scripts & launchers
│   ├── run_training.sh         # Main training launcher
│   ├── monitor_detailed.sh     # Advanced diagnostics
│   ├── cleanup_stale.sh        # Process cleanup utility
│   ├── diagnose_performance.py # Performance diagnostics
│   ├── generate_training_report.py
│   └── validate_project.py
│
├── docs/                       # Main documentation
│   └── Protocol.md             # Complete research protocol
│
├── docs_old/                   # Archived documentation
│   ├── TRAINING_STATUS.md
│   ├── PROJECT_STATUS.md
│   └── ...
│
├── data/
│   ├── raw/                    # Original gene expression data
│   ├── interim/                # Intermediate processing steps
│   └── processed/              # Final processed data
│
├── metadata/
│   ├── data_card.md            # Dataset documentation
│   ├── model_card.md           # Model documentation & ethics
│   └── biopanel.csv            # Curated gene list (optional)
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── data.py                 # Data loading & validation
│   ├── preprocess.py           # Batch correction & scaling
│   ├── features.py             # Feature selection methods
│   ├── models.py               # Model definitions
│   ├── pipeline.py             # Pipeline orchestration + nested CV
│   ├── eval.py                 # Evaluation metrics
│   ├── plots.py                # Visualization utilities
│   └── logging_config.py        # Enhanced logging system
│
├── scripts/                    # Executable scripts
│   ├── train_cv.py             # Nested CV training (main entry point)
│   ├── monitor_log.py          # Real-time log monitoring
│   └── shap_compact_panel.py   # SHAP explainability
│
├── notebooks/                  # Jupyter notebooks for EDA
│   ├── 00_eda.ipynb
│   └── archive/                # Previous versions
│
├── figures/                    # Generated plots
│   ├── eda/                    # Exploratory data analysis
│   ├── modeling/               # ROC, PR curves, confusion matrices
│   ├── calibration/            # Calibration & decision curves
│   └── shap/                   # SHAP visualizations
│
├── reports/                    # Generated reports
│   ├── tables/                 # CSV tables with metrics
│   └── html/                   # HTML reports (optional)
│
├── models/                     # Saved trained models
│   └── final_model_*.pkl
│
├── logs/                       # Training logs (auto-generated, .gitignore'd)
│   └── training_*.log
│
├── .gitignore                  # Git ignore rules
├── config.yaml                 # Legacy main config (deprecated)
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── SETUP.md                    # Installation instructions
```

**Key Changes for Organization:**
- ✅ Configs moved to `config/` folder
- ✅ Helper scripts moved to `utilities/` folder
- ✅ Documentation reorganized (`docs/` and `docs_old/`)
- ✅ Logs not committed (added to `.gitignore`)
- ✅ Cleaner root directory

---

## Quick Start

### Prerequisites

- **Python 3.10-3.13** (NOT 3.14 - SHAP incompatibility)
- **16GB+ RAM** (32GB recommended for full training)
- **~10GB disk space**
- **macOS/Linux/Windows** (instructions below for macOS/Linux)

### 1. Environment Setup

**Option A: Using Conda (Recommended)**
```bash
# Create conda environment from specification
conda env create -f environment.yml

# Activate environment
conda activate gbm-retake

# Verify Python version (should be 3.10-3.13)
python --version
```

**Option B: Using pip + venv**
```bash
# Ensure Python version is compatible
python3 --version  # Must be 3.10-3.13

# Create virtual environment
python3 -m venv .venv

# Activate environment (macOS/Linux)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**⚠️ Important:** Do not use Python 3.14 due to SHAP library incompatibility with numba.

### 2. Data Preparation

Place your data files in `data/raw/`:
- `gene_expression.csv` - Gene expression matrix (samples × genes)
- `metadata.csv` - Clinical metadata with columns: `sample_id`, `label`, `batch`, etc.

**Don't have data yet?** Run the preprocessing script to generate template files:
```bash
python scripts/make_processed.py --config config.yaml
```

This creates example data files that show the expected format.

### 3. Run the Complete Pipeline

**Quick Smoke Test (5-10 minutes - RECOMMENDED FIRST):**
```bash
# Validate setup with reduced configuration
utilities/run_training.sh config/config_smoke_test.yaml
```
This runs 3×3 CV with PCA only to verify everything works before the full run.

**⭐ RECOMMENDED: Ultrafast PCA-Based Training (10-15 minutes):**
```bash
# Use caffeinate on macOS to prevent laptop sleep
caffeinate -i utilities/run_training.sh config/config_ultrafast_pca.yaml

# Monitor progress in real-time
tail -f logs/training_*.log
```

**What this does:**
- Nested CV: 3 outer x 3 inner folds (fast and academically valid)
- Feature route: PCA (100 components)
- Models: Logistic Regression + Random Forest
- Evaluation: ROC-AUC, PR-AUC, calibration, decision curves, bootstrap CI
- Runtime: approximately 10-15 minutes
- Memory: approximately 12-15 GB peak

**Alternative: Filter_L1 Experimental (25-35 minutes):**
```bash
caffeinate -i utilities/run_training.sh config/config_fast_filter_l1.yaml
```

**Legacy: Full Academic Training (30-45 minutes):**
```bash
caffeinate -i utilities/run_training.sh config/config_academic_feasible.yaml
```

**Step 2: Feature Stability Analysis (15-20 minutes):**
```bash
python scripts/stability_analysis.py
```
Performs bootstrap stability selection (n=100) to identify consistently selected genes.

**Step 3: Compact Gene Panel Training (5 minutes):**
```bash
python scripts/shap_compact_panel.py
```
Trains a clinical-grade model on top-30 stable genes.

**Step 4: Generate Model Card (< 1 minute):**
```bash
python scripts/generate_model_card.py
```
Auto-populates `metadata/model_card_generated.md` with results and ethical considerations.

---

## Configuration

All pipeline settings are in `config.yaml`. Key parameters:

```yaml
random_state: 42                    # Fixed seed for reproducibility

cv:
  outer_folds: 5                    # Performance estimation (stratified)
  inner_folds: 3                    # Hyperparameter tuning (stratified)
  stratified: true                  # Preserve class distribution
  shuffle: true

preprocessing:
  batch_correction: "combat"        # Options: combat, harmony, none
  scaler: "standard"                # Options: standard, robust, minmax
  variance_threshold: 0.01          # Remove low-variance genes

features:
  routes: ["filter_l1", "pca"]      # Feature selection methods
  k_best: 200                       # Top genes for filter_l1
  pca_components: 200               # PCA components (captures ~80-90% variance)

models:
  lr_elasticnet:                    # Logistic Regression with L1+L2 penalties
    enabled: true
    param_grid:
      classifier__C: [0.001, 0.01, 0.1, 1.0, 10.0]
      classifier__l1_ratio: [0.3, 0.5, 0.7]
  
  random_forest:                    # Ensemble of decision trees
    enabled: true
    param_grid:
      classifier__n_estimators: [100, 200, 300]
      classifier__max_depth: [10, 20, 30, null]
      classifier__class_weight: ["balanced", "balanced_subsample"]
  
  lightgbm:                         # Gradient boosting (optimized)
    enabled: true
    param_grid:
      classifier__n_estimators: [100, 200, 300]
      classifier__learning_rate: [0.01, 0.05, 0.1]
      classifier__num_leaves: [31, 50, 100]

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
  calibration:
    enabled: true
    n_bins: 10
  decision_curve:
    enabled: true
  bootstrap_ci:
    enabled: true
    n_bootstrap: 1000
    confidence: 0.95
```

**Customizing the pipeline:** Edit `config.yaml` — no code changes needed!  
**Quick testing:** Use `config_smoke_test.yaml` (3×3 CV, PCA only, reduced grids)

---

## Results & Outputs

### Generated Artifacts

#### Figures
- `figures/eda/`: Class distribution, PCA variance explained
- `figures/modeling/`: ROC curves, PR curves, confusion matrices
- `figures/calibration/`: Calibration curves, decision curves (net benefit)
- `figures/modeling/stability_*`: Feature stability visualizations (bar charts, heatmaps)
- `figures/shap/`: Feature importance for compact gene panel

#### Tables
- `reports/tables/nested_cv_results.csv` - All fold-level metrics for all models
- `reports/tables/summary_metrics.csv` - Mean ± SD across folds (human-readable)
- `reports/tables/summary_metrics.tex` - LaTeX table for publications
- `reports/tables/metrics_ci_*.csv` - Metrics with bootstrap 95% CI
- `reports/tables/stability_panel.csv` - Top stable genes with selection frequencies
- `reports/tables/data_quality_report.csv` - Data preprocessing statistics

#### Models
- `models/final_model_{feature}_{model}.pkl` - Trained sklearn pipelines with metadata
- `models/final_model_compact_panel.pkl` - Clinical-grade model (top-30 genes)

#### Documentation
- `metadata/model_card_generated.md` - Auto-generated model card
- `metadata/data_card.md` - Dataset documentation
- `docs/Protocol.md` - Complete research protocol

---

## Methodology Details

### Nested Cross-Validation (5x3)

```
Outer Loop (5 folds, stratified):           Performance Estimation
└── Inner Loop (3 folds, stratified):       Hyperparameter Tuning
    ├── Feature Selection (fit on train)
    ├── Batch Correction (fit on train)
    ├── Scaling (fit on train)
    └── Classifier Training
    
Test Fold Evaluation (unseen data)
```

**Critical Design:**
- No data leakage: All preprocessing fitted only on training folds
- Unbiased estimates: Outer loop provides honest performance metrics
- Reproducible: Fixed random seed (42) throughout all operations
- Stratified: Preserves 93.3% healthy / 6.7% GBM ratio in all folds

**Computational Cost:**
- Per model: 5 outer x 3 inner x N_hyperparams = approximately 15-45 fits
- Total (all configs): approximately 90 model trainings
- Runtime: 2-3 hours on M4 Pro (48GB RAM)

### Feature Selection Routes

**1. Filter + L1 Regularization (`filter_l1`)**
- **Stage 1:** Variance filter (threshold=0.01) → removes uninformative genes
- **Stage 2:** Correlation filter (threshold=0.95) → removes redundancy
- **Stage 3:** L1 Logistic Regression → selects top 200 discriminative genes
- **Output:** Interpretable gene list suitable for pathway analysis
- **Advantage:** Biologically meaningful features
- **Trade-off:** Computationally expensive (~10-15 min per outer fold)

**2. Principal Component Analysis (`pca`)**
- **Method:** Unsupervised dimensionality reduction via SVD
- **n_components:** 200 (captures ~80-90% variance)
- **Output:** Orthogonal linear combinations of all genes
- **Advantage:** Fast computation (~1-2 min per outer fold), captures complex patterns
- **Trade-off:** Reduced interpretability (PCs are gene combinations)

**3. Stability Selection (Post-hoc)**
- **Method:** Bootstrap resampling (n=100) + frequency counting
- **Threshold:** Genes selected in ≥70% of bootstrap iterations
- **Purpose:** Identify robust biomarkers that generalize across samples
- **Output:** Ranked list of stable genes → top-30 used for compact panel

### Evaluation Metrics

**Primary Metrics:**
- **ROC-AUC:** Threshold-independent discrimination metric (0.5 = random, 1.0 = perfect)
- **PR-AUC:** Precision-Recall AUC (more informative for imbalanced data)

**Secondary Metrics:**
- **Classification:** Accuracy, Precision, Recall, F1, Specificity, Sensitivity
- **Probabilistic:** Brier score, Log loss

**Advanced Analyses:**
- **Calibration Curves:** Assess whether predicted probabilities reflect true frequencies
  - Expected Calibration Error (ECE)
  - Reliability diagrams (10 bins)
- **Decision Curve Analysis:** Quantify clinical utility via net benefit
  - Compare model to "treat all" and "treat none" baselines
  - Identify optimal operating points for clinical decisions
- **Confidence Intervals:** Bootstrap resampling (n=1000, 95% CI) for uncertainty quantification

**Reporting:**
- Mean ± SD across 5 outer folds (inter-fold variability)
- 95% bootstrap CI (statistical uncertainty)
- LaTeX tables for publication-ready results

---

## Explainability & Interpretability

### SHAP Analysis

SHAP (SHapley Additive exPlanations) provides:
- **Global importance**: Which genes matter most overall?
- **Directional effects**: Do high values increase GBM probability?
- **Feature interactions**: How do genes work together?
- **Individual explanations**: Why was this sample classified as GBM?

### Visualizations

- **Beeswarm Plot**: Shows feature importance + direction for all samples
- **Bar Plot**: Mean absolute SHAP values (global importance)
- **Waterfall Plot**: Step-by-step explanation for single sample
- **Dependence Plot**: Relationship between gene expression and SHAP value

---

## Documentation

### Ethical AI Documentation

- **Data Card** (`metadata/data_card.md`): Dataset characteristics, limitations, biases
- **Model Card** (`metadata/model_card.md`): Model details, performance, ethical considerations

### Code Documentation

- **Docstrings**: All functions have detailed docstrings
- **Type Hints**: Functions include type annotations
- **Comments**: Complex logic explained inline

---

## Ethical Considerations

### Important Limitations

**This is a RESEARCH PROTOTYPE:**
- **NOT approved for clinical diagnosis**
- **NOT a substitute for physician judgment**
- **NOT validated for treatment decisions**

### Responsible Use

- Use only for research and educational purposes
- Validate on external datasets before any clinical consideration
- Interpret results with domain expertise
- Consider fairness and bias in your specific context

See `metadata/model_card.md` for full ethical analysis.

---

## Testing & Validation

### Quality Assurance

- Data quality checks in preprocessing
- Syntax validation in all modules
- Pipeline leak-free design verified
- Metrics calculation validated against sklearn

### External Validation

For clinical translation, this model MUST be:
1. Validated on independent external cohorts
2. Assessed for fairness across demographics
3. Calibrated on target population
4. Reviewed by clinical domain experts

---

## Contributing

This is a student project for the Hogeschool Rotterdam retake.

**Institution:** Hogeschool Rotterdam  
**Program:** Minor AI in Healthcare  
**Project:** GBM Classification - Retake (November 2025)


---

## Acknowledgments

- Hogeschool Rotterdam - Minor AI in Healthcare teaching staff
- TCGA & GTEx for public gene expression data
- Open-source ML community (scikit-learn, SHAP, etc.)

---

## Contact

For questions about this project:
- **GitHub Issues**: [Create an issue](https://github.com/tobitotaku/ai-brain-tumor/issues)

---

**Last Updated:** November 12, 2025  
**Version:** 1.1  
**Status:** Ready for Evaluation
