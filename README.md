# GBM Classification from Gene Expression 🧬

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📋 Project Overview

This project implements a comprehensive, production-ready machine learning pipeline for **Glioblastoma (GBM) classification** using gene expression data. Developed for the **Minor AI in Healthcare** at Hogeschool Rotterdam, this work demonstrates best practices in:

- ✅ **Leak-free preprocessing** with proper fit/transform separation
- ✅ **Nested cross-validation** for unbiased performance estimation  
- ✅ **Multiple feature selection strategies** (filter+L1, PCA, biological panel)
- ✅ **Explainability** via SHAP analysis
- ✅ **Calibration & decision curve analysis**
- ✅ **Ethical AI documentation** (data cards, model cards)
- ✅ **Full reproducibility** via configuration management

---

## 🎯 Key Features

### 🔬 Methodology
- **Data Understanding**: Quality checks, EDA, class distribution analysis
- **Preprocessing**: Batch correction (ComBat/Harmony), variance filtering, scaling
- **Feature Selection**: Three routes (filter+L1, PCA, bio panel) + stability selection
- **Model Training**: Nested CV (5×5) with hyperparameter tuning
- **Evaluation**: ROC-AUC, PR-AUC, F1, calibration curves, decision curves
- **Explainability**: SHAP values, feature importance, single-sample explanations

### 🛠️ Technical Stack
- **ML Framework**: scikit-learn, LightGBM, XGBoost
- **Preprocessing**: ComBat (batch correction), StandardScaler
- **Explainability**: SHAP
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML-based pipeline configuration

---

## 📁 Project Structure

```
.
├── config.yaml                 # Central configuration file
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python dependencies
├── README.md                   # This file
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
│   ├── pipeline.py             # Pipeline orchestration
│   ├── eval.py                 # Evaluation metrics
│   └── plots.py                # Visualization utilities
│
├── scripts/                    # Executable scripts
│   ├── make_processed.py       # Data preprocessing
│   ├── train_cv.py             # Nested CV training
│   └── shap_report.py          # SHAP explainability
│
├── notebooks/                  # Jupyter notebooks for EDA
│   ├── 00_eda.ipynb
│   └── 01_sanity_checks.ipynb
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
└── models/                     # Saved trained models
    └── final_model_*.pkl
```

---

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Using Conda (Recommended)**
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gbm-retake
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

**Step 1: Preprocess Data**
```bash
python scripts/make_processed.py --config config.yaml
```
- Loads raw data
- Performs quality checks
- Removes duplicates & low-variance genes
- Creates EDA visualizations
- Saves processed data

**Step 2: Train Models with Nested CV**
```bash
python scripts/train_cv.py --config config.yaml
```
- Tests all feature selection × model combinations
- Performs nested cross-validation (5×5)
- Evaluates with comprehensive metrics
- Creates ROC curves, calibration plots, etc.
- Saves best model

**Step 3: Generate SHAP Explainability Report**
```bash
python scripts/shap_report.py --config config.yaml
```
- Computes SHAP values
- Creates beeswarm, bar, waterfall, dependence plots
- Saves feature importance rankings

---

## ⚙️ Configuration

All pipeline settings are in `config.yaml`:

```yaml
random_state: 42

cv:
  outer_folds: 5
  inner_folds: 5

preprocessing:
  batch_correction: "combat"  # or "harmony", "none"
  scaler: "standard"          # or "robust", "minmax"
  variance_threshold: 0.01

features:
  routes: ["filter_l1", "pca", "bio_panel"]
  k_best: 200
  pca_components: 50

models:
  lr_elasticnet:
    enabled: true
    param_grid: {...}
  random_forest:
    enabled: true
    param_grid: {...}
  lightgbm:
    enabled: true
    param_grid: {...}
```

**Customize the pipeline** by editing `config.yaml` - no code changes needed!

---

## 📊 Results & Outputs

### Generated Artifacts

#### 📈 Figures
- `figures/eda/`: Class distribution, PCA variance
- `figures/modeling/`: ROC curves, PR curves, confusion matrices
- `figures/calibration/`: Calibration curves, decision curves
- `figures/shap/`: SHAP beeswarm, bar, waterfall, dependence plots

#### 📋 Tables
- `reports/tables/nested_cv_results.csv` - All model performances
- `reports/tables/metrics_ci_*.csv` - Metrics with bootstrap CI
- `reports/tables/shap_summary.csv` - Feature importance rankings
- `reports/tables/data_quality_report.csv` - Data quality metrics

#### 🤖 Models
- `models/final_model_*.pkl` - Trained sklearn pipeline with metadata

---

## 🔍 Methodology Details

### Feature Selection Routes

1. **Filter + L1 (`filter_l1`)**
   - Variance filter → Correlation filter → L1-regularized Logistic Regression
   - Selects top K most discriminative genes

2. **PCA (`pca`)**
   - Unsupervised dimensionality reduction
   - Captures N components explaining most variance

3. **Biological Panel (`bio_panel`)**
   - Uses curated list of GBM-relevant genes
   - Incorporates domain knowledge

### Nested Cross-Validation

```
Outer Loop (5 folds):           Performance Estimation
└── Inner Loop (5 folds):       Hyperparameter Tuning
    ├── Feature Selection
    ├── Batch Correction
    ├── Scaling
    └── Classifier Training
```

- **No data leakage**: Preprocessing fitted only on training folds
- **Unbiased estimates**: Outer loop provides honest performance metrics
- **Reproducible**: Fixed random seeds throughout

### Evaluation Metrics

- **Discrimination**: ROC-AUC, PR-AUC
- **Classification**: Accuracy, Precision, Recall, F1
- **Calibration**: Brier score, ECE, reliability curves
- **Clinical Utility**: Decision curve analysis (net benefit)
- **Uncertainty**: Bootstrap 95% confidence intervals

---

## 🔬 Explainability & Interpretability

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

## 📚 Documentation

### Ethical AI Documentation

- **Data Card** (`metadata/data_card.md`): Dataset characteristics, limitations, biases
- **Model Card** (`metadata/model_card.md`): Model details, performance, ethical considerations

### Code Documentation

- **Docstrings**: All functions have detailed docstrings
- **Type Hints**: Functions include type annotations
- **Comments**: Complex logic explained inline

---

## ⚠️ Ethical Considerations

### ⛔ Important Limitations

**This is a RESEARCH PROTOTYPE:**
- ❌ **NOT approved for clinical diagnosis**
- ❌ **NOT a substitute for physician judgment**
- ❌ **NOT validated for treatment decisions**

### ✅ Responsible Use

- Use only for research and educational purposes
- Validate on external datasets before any clinical consideration
- Interpret results with domain expertise
- Consider fairness and bias in your specific context

See `metadata/model_card.md` for full ethical analysis.

---

## 🧪 Testing & Validation

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

## 🤝 Contributing

This is a student project for the Hogeschool Rotterdam retake. For questions or collaboration:

**Team Members:**
- Musab Sivrikaya
- Jim Tronchet
- Ozeir Moradi  
- Tim Grootscholten
- Tobias Roessingh

**Institution:** Hogeschool Rotterdam  
**Program:** Minor AI in Healthcare  
**Project:** GBM Classification - Retake (November 2025)


---

## 🙏 Acknowledgments

- Hogeschool Rotterdam - Minor AI in Healthcare teaching staff
- TCGA & GTEx for public gene expression data
- Open-source ML community (scikit-learn, SHAP, etc.)

---

## 📧 Contact

For questions about this project:
- **GitHub Issues**: [Create an issue](https://github.com/tobitotaku/ai-brain-tumor/issues)

---

**Last Updated:** November 5, 2025  
**Version:** 1.0  
**Status:** ✅ Ready for Evaluation
