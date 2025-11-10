# Research Protocol: Glioblastoma Classification from Gene Expression Data

**Project Title:** Machine Learning-Based Glioblastoma Classification Using Gene Expression Profiles  
**Institution:** Hogeschool Rotterdam, Minor AI in Healthcare  
**Project Type:** Capstone Project (Retake)  

**Version:** 1.2

---

## 1. Executive Summary

This protocol defines a leakage-safe, reproducible machine learning pipeline for classifying glioblastoma (GBM) versus healthy controls using gene expression data. We employ nested cross-validation (5×3), two dimensionality-reduction routes (L1-filtering vs. PCA), and three model families (Logistic Regression with Elastic Net, Random Forest, LightGBM). Evaluation covers discrimination (ROC-AUC, Average Precision), calibration (intercept, slope, Brier), clinical utility (Decision Curve Analysis), and uncertainty (bootstrap CIs). All design choices are grounded in peer-reviewed sources.

**Objectives:**
- Build interpretable, reproducible models for GBM classification on high-dimensional gene data
- Compare interpretable gene signatures (L1) vs. latent factor models (PCA)
- Quantify discrimination, calibration, and clinical benefit with uncertainty
- Align with TRIPOD+AI standards for ethical, transparent AI in healthcare

---

## 2. Clinical Context & Problem Definition

### 2.1 Background

Glioblastoma (GBM) is the most common malignant primary brain tumor in adults with poor prognosis despite multimodal therapy (Ostrom et al., 2023; WHO Classification of Tumours Editorial Board, 2021). This study evaluates whether gene-expression-based machine learning can assist research diagnosis or biomarker discovery.

### 2.2 Research Questions

**Primary Question:** Can machine learning distinguish GBM from healthy controls with clinically meaningful discrimination, calibration, and net benefit?

**Secondary Questions:**
1. What is the trade-off between interpretability and accuracy (L1 vs PCA)?
2. How stable are selected gene signatures across cross-validation folds?
3. Are model predictions well-calibrated for clinical decision-making?
4. What is the clinical utility (net benefit) across different decision thresholds?

### 2.3 Clinical Significance & Limitations

**Potential Benefits:**
- Decision support for differential diagnosis
- Biomarker discovery for molecular characterization
- Stratification hypotheses for personalized treatment

**Important Limitations:**
- Research prototype only — external validation and regulatory approval required for clinical deployment
- Results generalize only to populations similar to TCGA/GTEx cohorts
- Biological interpretation limited by available functional annotations

---

## 3. Dataset Description

### 3.1 Data Sources

**Primary Datasets:**
- TCGA (The Cancer Genome Atlas) — GBM cohort
- GTEx (Genotype-Tissue Expression) — healthy controls


**Sample Size:** Approximately 18,600 samples  
**Features:** 18,700 genes after quality control  
**Format:** ENSEMBL gene identifiers with normalized expression values

**Confounding control.** Healthy controls are restricted to **brain tissue** (GTEx) to avoid confounding by tissue of origin. We record **batch/source labels** (TCGA site/platform, GTEx tissue center) and include them in downstream **fold-internal** batch correction. Reported performance is thus intended to reflect **disease signal** rather than cross-dataset or tissue artifacts. A detailed breakdown and inclusion/exclusion criteria are listed in `metadata/data_card.md`.

### 3.2 Class Imbalance

**Class Distribution:** GBM ≈7% (1,253 samples), Healthy ≈93% (17,382 samples)

This 14:1 imbalance is addressed through:
- Stratified cross-validation to preserve class ratios in all folds
- Class-weighted loss functions in Random Forest and LightGBM
- Precision-Recall metrics (Average Precision) alongside ROC-AUC
- Decision curve analysis to evaluate clinical utility

**Justification:** Stratified sampling ensures representative class proportions, preventing models from learning spurious patterns due to imbalance (Bradshaw & Obuchowski, 2023).

### 3.3 Data Quality, Harmonization, and Batch Effects

**Quality Control Steps:**
1. **Variance Filtering:** Remove genes with variance < 0.01 (eliminates uninformative features)
2. **Duplicate Removal:** Eliminate duplicate samples to prevent data leakage
3. **Gene Harmonization:** Standardize ENSEMBL identifiers across datasets
4. **Missing Value Check:** Verify completeness of expression matrix

**Batch Effect Correction:** We screen batch structure via **PCA on the training folds**. For **normalized expression data**, we apply **ComBat (location/scale)** fitted **only on the training subset** and then **apply the frozen parameters** to validation/test folds. (If an alternative raw-counts pipeline is used, ComBat-seq may be considered; otherwise ComBat is standard.) All filtering, normalization, scaling, PCA and selection are performed **within the fold** via a scikit-learn `Pipeline` to prevent leakage.

**Data Documentation:**
Complete dataset characteristics documented in `metadata/data_card.md` including:
- Data provenance and collection methodology
- Preprocessing pipeline and quality metrics
- Known limitations and ethical considerations

---

## 4. Validation Strategy

### 4.1 Nested Cross-Validation Design

**Standard Configuration (`config.yaml`):**
- **Outer Loop:** 5 folds (stratified) → unbiased performance estimation
- **Inner Loop:** 3 folds (stratified) → hyperparameter optimization
- **Global Random Seed:** 42 (fixed for reproducibility)
- **Stratification:** Maintains 93%/7% class ratio in all folds

**Feasibility Configuration (`config_academic_feasible.yaml`):**
- **Outer Loop:** 3 folds (stratified) → computationally feasible variant
- **Inner Loop:** 3 folds (stratified) → hyperparameter optimization
- **Global Random Seed:** 42 (fixed for reproducibility)
- **Stratification:** Maintains 93%/7% class ratio in all folds
- **Justification:** 3-5 outer folds provide sufficient performance estimation (Bradshaw et al., 2023); 3×3 nested CV maintains academic rigor while enabling completion within retake timeline

**Primary Performance Estimate:** Mean across outer folds (5 or 3 depending on configuration)

**Primary evaluation policy.** We report performance from **nested cross-validation only** (no extra hold-out). Hyperparameter selection occurs in the inner loop and performance is estimated in the outer loop to avoid optimistic bias.

**Rationale:** Nested cross-validation provides unbiased performance estimates by separating hyperparameter tuning (inner loop) from performance evaluation (outer loop), preventing optimistic bias (Bradshaw & Obuchowski, 2023; Wainer & Cawley, 2021). The 3-fold variant maintains methodological soundness while addressing computational constraints typical in educational and resource-limited settings.

**Total Model Fits per Configuration:**
- Standard (5×3): 5 outer folds × 3 inner folds × N hyperparameter combinations (~90-150 fits)
- Feasible (3×3): 3 outer folds × 3 inner folds × N hyperparameter combinations (~54-90 fits)

### 4.2 Data Leakage Prevention

**Critical Controls:**
1. Feature selection fitted **exclusively** on training folds
2. Batch correction parameters learned **exclusively** on training folds
3. Scaling parameters (mean, std) computed **exclusively** on training folds
4. PCA transformation fitted **exclusively** on training folds
5. No information from test fold used during model training or hyperparameter tuning
6. All preprocessing wrapped in scikit-learn Pipeline for automatic isolation

**Code Implementation:**
```python
pipeline = Pipeline([
    ('feature_selector', FilterL1Selector(...)),  # Fit on train only
    ('batch_correction', BatchCorrector(...)),    # Fit on train only
    ('scaler', StandardScaler(...)),              # Fit on train only
    ('classifier', LogisticRegression(...))       # Fit on train only
])
```

**Verification:** Pipeline ensures fit/transform separation automatically, preventing common leakage errors documented in recent literature (Rosenblatt et al., 2024).

### 4.3 Reproducibility

**Seeded Operations:**
- Cross-validation fold splits
- Model initialization (Random Forest, LightGBM)
- Bootstrap resampling
- PCA computation

**Documentation:**
- Fold indices archived for exact replication
- Configuration files (`config.yaml`) version-controlled
- Python environment locked (`requirements.txt`)
- All random seeds logged in training output

---

## 5. Feature Reduction Strategies

### 5.1 L1-Based Gene Selection (`filter_l1`)

**Objective:** Select discriminative genes while maintaining biological interpretability.

**Multi-Stage Pipeline:**

**Stage 1 — Variance Filtering:**
- Remove genes with variance < 0.01
- Rationale: Low-variance genes provide minimal discriminatory information (Bommert et al., 2022)
- Implementation: `sklearn.feature_selection.VarianceThreshold`

**Stage 2 — Correlation Pruning:**
- Remove one gene from pairs with |correlation| > 0.95
- Rationale: Highly correlated genes are redundant and cause multicollinearity
- Implementation: Pairwise Pearson correlation on upper triangle

**Stage 3 — L1 Regularization:**
- Logistic Regression with L1 penalty (Lasso)
- Select top k genes with highest |coefficient|
- Rationale: L1 penalty drives irrelevant feature coefficients to exactly zero, performing embedded feature selection
- **Hyperparameter Tuning:** k ∈ {50, 100, 200, 300} tuned via inner CV
  Used strictly as an embedded **selector**; the downstream **classifier may differ** (e.g., ElasticNet-regularized Logistic Regression).

**Output:** k genes with interpretable biological meaning

**Advantages:**
- Biologically interpretable (real genes, not linear combinations)
- Enables downstream pathway enrichment analysis
- Sparse feature set reduces overfitting

**Computational Cost:** ~10-15 minutes per outer fold

### 5.2 Principal Component Analysis (`pca`)

**Objective:** Capture maximum variance in unsupervised manner.

**Configuration:**
- **Number of Components (m):** Tuned via inner CV, m ∈ {50, 100, 200}
- **Expected Variance Explained:** ≥80% (empirically validated)
- **Method:** Singular Value Decomposition (SVD)
- **Centering:** Applied (zero-mean features)
  We **log the cumulative explained variance per fold** and require **≥80%** as a heuristic target for m∈{50,100,200}.

**Rationale:** PCA projects high-dimensional gene space onto orthogonal components that maximize variance, potentially capturing complex gene interaction patterns (Greenacre et al., 2022; Zhang et al., 2024).

**Output:** m principal components (PC1, PC2, ..., PCm)

**Advantages:**
- Computationally efficient (~1-2 minutes per outer fold)
- Orthogonal components reduce multicollinearity
- Unsupervised (no risk of overfitting to labels)

**Trade-offs:**
- Reduced biological interpretability (linear combinations of all genes)
- Assumes linear relationships
- Sensitive to outliers

### 5.3 Stability & Biological Relevance

**Stability Selection:**
- Bootstrap resampling (n=100) for L1-based selection
- Compute Jaccard overlap of selected genes across folds
- Define stable genes as those selected in ≥70% of bootstrap iterations
- Rationale: Stable features are more likely to generalize to external datasets (Haftorn et al., 2023; Łukaszuk et al., 2024)

**Biological Validation:**
- Gene Ontology (GO) and KEGG pathway enrichment for stable genes
- Literature comparison with known GBM biomarkers
- Derive compact panel of ~30 genes for clinical deployment

**Implementation:** `scripts/stability_analysis.py`

---

## 6. Model Architectures & Hyperparameter Tuning

### 6.1 Model Selection Rationale

Three model families representing different inductive biases:

| Model | Type | Strengths | Hyperparameter Tuning |
|-------|------|-----------|----------------------|
| **Logistic Regression (ElasticNet)** | Linear, probabilistic | Interpretable coefficients, fast training | C, l1_ratio |
| **Random Forest** | Ensemble (bagging) | Handles non-linearity, robust to outliers | n_estimators, max_depth, max_features |
| **LightGBM** | Ensemble (boosting) | State-of-art for tabular data, efficient | num_leaves, learning_rate, feature_fraction, scale_pos_weight |

**Rationale:**
- Linear baseline (Logistic Regression) tests whether simple models suffice
- Non-linear alternatives (RF, LightGBM) capture complex gene interactions
- Tree-based models consistently outperform deep learning on tabular data (Grinsztajn et al., 2022)

### 6.2 Logistic Regression with ElasticNet

**Mathematical Formulation:**
$$
\min_{w, b} \frac{1}{2n} \sum_{i=1}^{n} \log(1 + \exp(-y_i(w^T x_i + b))) + \alpha \left[ \rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2 \right]
$$

Where:
- $\alpha$ = regularization strength (controlled by C = 1/α)
- $\rho$ = l1_ratio (balance between L1 and L2 penalties)

**Hyperparameter Grid:**
```yaml
classifier__C: [0.001, 0.01, 0.1, 1.0, 10.0]        # 5 values
classifier__l1_ratio: [0.3, 0.5, 0.7]               # 3 values
classifier__max_iter: [2000]                        # Fixed (ensure convergence)
# Total combinations: 15
```

**Rationale:**
- C range spans 4 orders of magnitude from strong to weak regularization
- l1_ratio values test L2-dominated (0.3), balanced (0.5), L1-dominated (0.7)
- max_iter=2000 ensures convergence on high-dimensional data

**Implementation:** `sklearn.linear_model.LogisticRegression` with `solver='saga'`

### 6.3 Random Forest

**Hyperparameter Grid:**
```yaml
classifier__n_estimators: [100, 200, 300]           # Number of trees
classifier__max_depth: [10, 20, 30, null]           # Tree depth (null = unlimited)
classifier__min_samples_split: [2, 5, 10]           # Min samples to split node
classifier__min_samples_leaf: [1, 2, 4]            # Min samples per leaf
classifier__max_features: ["sqrt", "log2", 0.5]    # NEW: feature subsampling
classifier__class_weight: ["balanced", "balanced_subsample"]
# Total combinations updated accordingly
```

**Rationale:**
- n_estimators: More trees increase stability but with diminishing returns
- max_depth: Controls overfitting (shallow trees generalize better)
- min_samples_split/leaf: Regularization preventing overly complex leaves
- class_weight: Automatically adjusts for 14:1 class imbalance

**Implementation:** `sklearn.ensemble.RandomForestClassifier`

### 6.4 LightGBM

**Rationale:** `learning_rate` trades off bias-variance; `num_leaves` and `max_depth` control tree complexity; and **class imbalance** is addressed via `scale_pos_weight ≈ (n_negative / n_positive)` per training split (or `class_weight="balanced"` if the wrapper does not expose `scale_pos_weight`). Choices are pre-specified and logged.

**Hyperparameter Grid:**
```yaml
classifier__n_estimators: [100, 200, 300]          # Boosting rounds
classifier__learning_rate: [0.01, 0.05, 0.1]       # Step size
classifier__max_depth: [5, 10, 15]                 # Tree depth
classifier__num_leaves: [31, 63, 127]              # Max leaves per tree
classifier__min_child_samples: [10, 20, 30]        # Min samples per leaf
classifier__scale_pos_weight: [10, 14, 20]         # approx. class ratio; tuned in inner CV
```

**Implementation:** `lightgbm.LGBMClassifier` with `objective='binary'`, `boosting_type='gbdt'`

### 6.5 Hyperparameter Optimization

**Method:** GridSearchCV with stratified inner CV

**Configuration:**
- Scoring metric: `roc_auc` (primary)
- Secondary metric: `average_precision` (PR-AUC)
- CV strategy: `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- Parallel processing: `n_jobs=-1`

**Implementation:**
```python
GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
```

---

## 7. Evaluation Framework

### 7.1 Discrimination Metrics

**ROC-AUC (Receiver Operating Characteristic — Area Under Curve):**
- **Range:** 0.5 (random) to 1.0 (perfect)
- **Interpretation:** Probability that a randomly chosen GBM sample ranks higher than a randomly chosen healthy sample
- **Threshold-independent:** Aggregates performance across all decision thresholds
- **Justification:** Despite class imbalance, ROC-AUC accurately characterizes model performance (Richardson et al., 2024)

**Average Precision (AP) / PR-AUC:**
- **Range:** Baseline (prevalence = 0.07) to 1.0 (perfect)
- **Interpretation:** Weighted mean precision across recall thresholds
- **Advantage:** More sensitive to performance on minority class (GBM)
- **Justification:** Complements ROC-AUC by emphasizing precision-recall trade-off in imbalanced settings (McDermott et al., 2024)

**Reporting:** Both ROC-AUC and Average Precision reported jointly with PR curves including prevalence baseline

### 7.2 Threshold-Dependent Metrics

**Clinical Operating Points:**
- **Triage Setting:** High sensitivity threshold (minimize false negatives)
- **Confirmatory Setting:** High precision threshold (minimize false positives)

**Metrics at Selected Thresholds:**
- **Sensitivity (Recall):** True positive rate — P(predicted GBM | GBM)
- **Specificity:** True negative rate — P(predicted healthy | healthy)
- **Precision (PPV):** Positive predictive value — P(GBM | predicted GBM)
- **NPV:** Negative predictive value — P(healthy | predicted healthy)
- **F1-Score:** Harmonic mean of precision and recall

**Threshold Selection Methods:**
- Youden's J statistic (maximizes sensitivity + specificity - 1)
- Clinical utility-based thresholds from Decision Curve Analysis

### 7.3 Calibration Analysis

**Objective:** Assess whether predicted probabilities reflect true outcome frequencies.

**Calibration Intercept & Slope:**
- Fit logistic regression: logit(observed) ~ logit(predicted)
- Perfect calibration: intercept = 0, slope = 1
- Recommended by TRIPOD+AI for model evaluation (Collins et al., 2024)

**Brier Score:**
$$
\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2
$$
- **Range:** 0 (perfect) to 1 (worst)
- **Interpretation:** Mean squared error between predicted probabilities and true outcomes

**Calibration Curves (Reliability Diagrams):**
- Partition predictions into 10 bins
- Plot mean predicted probability vs observed frequency per bin
- Perfect calibration follows diagonal (y = x)
- **Implementation:** `sklearn.calibration.calibration_curve`

**Recalibration:**
- If calibration-in-the-large poor: apply isotonic or Platt scaling within inner CV
- Refit calibration model on training folds only

**Justification:** Calibration is critical for clinical deployment where probability thresholds guide actions (Collins et al., 2024; Riley et al., 2024).

### 7.4 Decision Curve Analysis

**Objective:** Quantify clinical utility by calculating net benefit across decision thresholds.

**Net Benefit Formula:**
$$
\text{Net Benefit}(t) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{t}{1-t}
$$

Where:
- $t$ = probability threshold for intervention
- $\frac{t}{1-t}$ = harm-to-benefit ratio (weighs cost of false positives)
- Comparison strategies: "treat all", "treat none", model-guided decisions

**Implementation:**
- Compute net benefit across threshold range [0.0, 1.0] with 100 steps
- Bootstrap at patient level to quantify uncertainty
- Aggregate curves across outer folds
- Report probability of positive net benefit per threshold

**Threshold range.** Given prevalence ≈7%, we emphasise thresholds **t ∈ [0.03, 0.20]** as clinically plausible, alongside the full [0,1] curve. Curves are aggregated across outer folds with bootstrap CIs; we highlight ranges where model net benefit exceeds both “treat none” and “treat all”.

**Interpretation:**
- Net benefit > 0 and higher than alternatives → clinically useful
- Identifies threshold ranges where model adds value over default strategies

**Justification:** DCA moves beyond discrimination metrics to quantify clinical decision-making value (Sadatsafavi et al., 2021; Vickers et al., 2023).

### 7.5 Uncertainty Quantification

**Method:** Bootstrap resampling at patient level with replacement

**Configuration:**
- **n_bootstrap:** 1000 iterations
- **Confidence level:** 95% (2.5th and 97.5th percentiles)
- **Metrics:** ROC-AUC, Average Precision, Brier score, calibration slope/intercept, net benefit
- **Resampling unit:** Patient (preserves correlation structure)

**Implementation:**
```python
bootstrap_scores = []
for i in range(1000):
    indices = resample(range(len(y_true)), replace=True, random_state=42+i)
    y_boot = y_true[indices]
    prob_boot = y_prob[indices]
    bootstrap_scores.append(roc_auc_score(y_boot, prob_boot))

ci_lower, ci_upper = np.percentile(bootstrap_scores, [2.5, 97.5])
```

**Reporting:**
- Mean ± SD across 5 outer folds (inter-fold variability)
- 95% bootstrap CI (statistical uncertainty within folds)
- Combined presentation: Mean [95% CI] ± SD

### 7.6 Reporting Standards

**Tables:**
- Mean ± SD across outer folds
- 95% bootstrap CI for primary metrics
- LaTeX format for publication (`reports/tables/*.tex`)
- Per-class F1 tables and operating-point confusion matrices per outer fold

**Figures:**
- ROC curves (all folds + mean)
- PR curves (all folds + mean)
- Calibration curves (reliability diagrams)
- Decision curves (net benefit vs threshold)
- Confusion matrices (at optimal threshold)

**Reproducibility:**
- All random seeds logged
- Configuration file versioned with results
- Model artifacts saved with metadata

---

## 8. Analysis Plan

### 8.1 Model Comparison

**Aggregation Strategy:**
- Collect predictions from all 5 outer folds for each (Feature Route × Model) combination
- Compute metrics per fold, then aggregate across folds
- Visualize ROC curves, PR curves, calibration curves, and decision curves

**Statistical Approach:**
- Report point estimates with 95% CIs (avoid overreliance on p-values)
- If pairwise comparisons needed: apply Bonferroni correction (α_adjusted = 0.05 / n_comparisons)
- Prefer effect sizes and confidence intervals over hypothesis testing (Efthimiou et al., 2024)

**Presentation:**
- Tables: Mean ± SD across folds with [95% CI]
- Figures: Overlay curves from all folds + mean curve
- LaTeX tables for publication: `reports/tables/summary_metrics.tex`

### 8.2 Interpretability Analysis

**For Logistic Regression:**
- Gene coefficients ranked by |coefficient|
- Sign indicates direction (positive coefficient → higher expression associated with GBM)
- Top 20 genes documented with biological function annotations

**For Tree-Based Models (RF, LightGBM):**
- Feature importances (Gini for RF, gain for LightGBM)
- SHAP (SHapley Additive exPlanations) values for compact gene panel
- Beeswarm and waterfall plots for model interpretation

**Biological Relevance:**
- Gene Ontology (GO) enrichment analysis for stable gene sets
- KEGG pathway analysis
- Literature comparison with known GBM biomarkers (e.g., EGFR, TP53, PTEN pathways)

### 8.3 Final Model Refit

**Procedure:**
- After completing nested CV, select best (Feature Route × Model × Hyperparameters) based on mean ROC-AUC
- Refit final model on **all available data** using selected hyperparameters
- **Important:** Do not re-evaluate on training data (this would be optimistically biased)
- Save final model with metadata for deployment testing on future external datasets

**Model Artifact:**
```python
{
    'model': fitted_pipeline,
    'config': config_dict,
    'cv_performance': nested_cv_metrics,
    'timestamp': datetime.now(),
    'random_state': 42
}
```

---

## 9. Additional Analyses

### 9.1 Feature Stability Selection

**Objective:** Identify genes consistently selected across different training subsets.

**Method:** Bootstrap stability selection with n=100 iterations

**Algorithm:**
1. For i = 1 to 100:
   - Create bootstrap sample (resample with replacement)
   - Apply filter_l1 feature selection
   - Record selected genes
2. Calculate selection frequency: $f_j = \frac{\text{count}(j)}{100}$
3. Define stable genes as $f_j \geq 0.70$ (selected in ≥70% of iterations)

**Output:**
- `reports/tables/stability_panel.csv` — genes ranked by stability score
- `figures/modeling/stability_top50_bar.png` — bar chart of top 50 genes
- `figures/modeling/stability_top50_heatmap.png` — selection pattern across iterations

**Biological Validation:**
- Compute Jaccard overlap of gene sets across folds
- Perform GO/KEGG enrichment on stable genes
- Compare with published GBM gene signatures

**Justification:** Stable features generalize better to external datasets and are more biologically plausible (Haftorn et al., 2023; Łukaszuk et al., 2024).

**Implementation:** `scripts/stability_analysis.py`

### 9.2 Compact Gene Panel

**Objective:** Develop clinically deployable model on minimal gene set.

**Method:**
1. Select top 30 genes from stability analysis (highest selection frequency)
2. Train Logistic Regression on compact panel using same nested CV
3. Evaluate with identical metrics as full models

**Advantages:**
- Lower assay cost (fewer genes to measure)
- Faster turnaround time
- Simpler regulatory approval pathway
- Reduced overfitting risk

**Trade-off Analysis:**
- Quantify performance decrease vs. full filter_l1 model
- Acceptable if ΔROC-AUC < 0.05 and maintains calibration

**Output:**
- `models/final_model_compact_panel.pkl`
- `reports/tables/metrics_ci_compact_panel.csv`
- `figures/shap/feature_importance_compact_panel.png`

**Implementation:** `scripts/shap_compact_panel.py`

### 9.3 Model Card Generation

**Objective:** Auto-document model characteristics, performance, and ethical considerations.

**Content (TRIPOD+AI Compliant):**
- Model architecture and hyperparameters
- Training data statistics and preprocessing
- Performance metrics with confidence intervals
- Calibration and clinical utility results
- Intended use and out-of-scope applications
- Fairness assessment (where metadata available)
- Deployment contraindications
- Limitations and future validation needs

**Output:** `metadata/model_card_generated.md`

**Standard:** Based on Model Cards for Model Reporting framework and TRIPOD+AI guidelines (Collins et al., 2024)

**Implementation:** `scripts/generate_model_card.py`

---

## 10. Reproducibility & Transparency

### 10.1 Computational Reproducibility

**Version Control:**
- Python version: 3.10-3.13 (locked in `environment.yml`)
- Library versions: Pinned in `requirements.txt`
- Configuration: All parameters in `config.yaml`
- Git commit hash: Logged with model artifacts

**Seed Management:**
- Global seed: 42
- All random operations seeded (CV splits, model init, bootstrap)
- Fold indices archived for exact replication

**Artifact Storage:**
- Trained models: `models/final_model_*.pkl` with metadata
- Predictions: `reports/tables/predictions_*.csv` (all folds)
- Metrics: `reports/tables/nested_cv_results.csv`
- Figures: `figures/` (all visualizations)

**Reproducibility Verification:**
```bash
# Run training twice with identical config
python scripts/train_cv.py --config config.yaml
# Compare checksums of output files
```

### 10.2 TRIPOD+AI Compliance

**Checklist Items:**
- ✅ Clear problem definition and intended use
- ✅ Data source and sample size documented
- ✅ Preprocessing and feature engineering detailed
- ✅ Model architectures and hyperparameters specified
- ✅ Training/validation strategy (nested CV) described
- ✅ Performance metrics with uncertainty quantification
- ✅ Calibration assessment included
- ✅ Clinical utility evaluation (DCA)
- ✅ Limitations and contraindications stated
- ✅ Code and data availability plan

**Documentation:**
- Protocol: `docs/Protocol.md` (this document)
- Data card: `metadata/data_card.md`
- Model card: `metadata/model_card_generated.md`
- Code repository: GitHub (public upon publication)

### 10.3 Publication & Archival

**Planned Outputs:**
- Preprint (medRxiv or bioRxiv)
- Peer-reviewed publication
- GitHub repository with DOI (Zenodo)
- Model weights and configuration files
- TRIPOD+AI checklist (supplementary material)

---

## 11. Ethical Considerations

### 11.1 Data Ethics

**Data Sources & Consent:**
- Public repositories: TCGA (The Cancer Genome Atlas) and GTEx (Genotype-Tissue Expression)
- De-identified samples (no protected health information)
- Original data collection approved by respective institutional review boards
- Secondary data use: Verified compliance with data use agreements

**Privacy:**
- No re-identification risk (all personal identifiers removed)
- Data aggregated at cohort level for analysis
- Results reported without individual-level information

### 11.2 Fairness Assessment

**Class Imbalance:**
- Addressed via stratified CV, class weighting, and PR-space metrics
- Ensures minority class (GBM) receives appropriate model attention

**Demographic Fairness (if metadata available):**
- Stratified performance analysis by sex and age groups
- Report disparate impact metrics (equalized odds, demographic parity)
- Test for statistical significance of performance differences across subgroups

**Known Limitations:**
- TCGA/GTEx datasets may not represent global population diversity
- Limited ethnicity metadata prevents comprehensive fairness audit
- Documented as limitation in model card

### 11.3 Clinical Deployment Safeguards

**Current Status:** Research prototype only

**Contraindications for Clinical Use:**
- ❌ No external validation on independent cohorts
- ❌ No calibration on target clinical population
- ❌ No assessment of performance under distributional shift
- ❌ No review by clinical domain experts
- ❌ No regulatory approval (CE marking, FDA clearance)

**Required Steps Before Deployment:**
1. **External Validation:** Test on geographically and temporally distinct GBM cohorts
2. **Prospective Study:** Evaluate in real clinical workflow
3. **Calibration:** Recalibrate on target population
4. **Clinical Review:** Obtain approval from neuro-oncology experts
5. **Regulatory Approval:** Complete submission to relevant authorities
6. **Integration:** Embed in clinical decision support system with human oversight
7. **Monitoring:** Continuous performance surveillance for model drift

### 11.4 Transparency & Accountability

**Documentation:**
- ✅ Complete data card (`metadata/data_card.md`)
- ✅ Complete model card (`metadata/model_card_generated.md`)
- ✅ Open-source code (GitHub repository)
- ✅ Reproducible pipeline (configuration-driven)
- ✅ TRIPOD+AI checklist (supplementary material)

**Limitations Disclosure:**
- Research prototype status clearly stated
- Class imbalance challenges documented
- Batch effect sensitivity acknowledged
- Lack of external validation highlighted
- Population representativeness concerns noted

**Accountability:**
- Institutional affiliation: Hogeschool Rotterdam
- Contact information for questions
- Version control and change tracking
- Clear documentation of any protocol deviations

---

## 12. Workflow Summary

### 12.1 Execution Pipeline

```
┌──────────────────────────────────────────────────┐
│ 1. Data Loading                                  │
│    ├─ gene_expression.csv                        │
│    ├─ metadata.csv                               │
│    └─ Merges → (18,635 samples × 18,752 genes)   │
│       Stratified into CV folds (6.7% GBM)        │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│ 2. Nested Cross-Validation Loop (5×3)            │
│    For each feature route:                       │
│      For each model:                             │
│        For each outer fold (5):                  │
│          ├─ Split train/test (stratified)        │
│          ├─ Inner CV (3 folds):                  │
│          │   └─ Grid search best params          │
│          ├─ Refit on full train set              │
│          └─ Evaluate on test set                 │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│ 3. Aggregate & Report                            │
│    ├─ Calculate mean ± SD across folds           │
│    ├─ Compute bootstrap CI (n=1000)              │
│    ├─ Calibration analysis                       │
│    ├─ Decision curve analysis                    │
│    └─ Generate visualizations                    │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│ 4. Stability & Compact Panel                     │
│    ├─ Stability selection (bootstrap n=100)      │
│    ├─ Top-30 gene panel identification           │
│    └─ Compact panel training & validation        │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│ 5. Interpretability & Reporting                  │
│    ├─ SHAP analysis                              │
│    ├─ Model card generation (TRIPOD+AI)          │
│    └─ Final report compilation                   │
└──────────────────────────────────────────────────┘
```

### 12.2 Execution Commands

**Full Academic Run (~2-3 hours):**
```bash
caffeinate -i python scripts/train_cv.py --config config.yaml
```

**Smoke Test (~5-10 minutes):**
```bash
python scripts/train_cv.py --config config_smoke_test.yaml
```

**Post-Training Analysis:**
```bash
python scripts/stability_analysis.py       # Feature stability (15-20 min)
python scripts/shap_compact_panel.py       # Compact panel (5 min)
python scripts/generate_model_card.py      # Model card (<1 min)
```

### 12.3 Expected Outputs

**Performance Reports:**
- `reports/tables/nested_cv_results.csv` — All fold-level metrics
- `reports/tables/summary_metrics.csv` — Mean ± SD per model
- `reports/tables/summary_metrics.tex` — LaTeX table for publication
- `reports/tables/stability_panel.csv` — Stable genes
- `reports/tables/metrics_ci_compact_panel.csv` — Compact panel performance

**Visualizations:**
- `figures/modeling/roc_curves.png` — ROC curves with AUC CI
- `figures/modeling/pr_curves.png` — Precision-Recall curves
- `figures/modeling/confusion_matrices.png` — Confusion matrices
- `figures/calibration/calibration_curves.png` — Calibration plots
- `figures/calibration/decision_curves.png` — Decision curves (DCA)
- `figures/modeling/stability_top50_bar.png` — Feature stability bar chart
- `figures/modeling/stability_top50_heatmap.png` — Stability heatmap
- `figures/shap/feature_importance_compact_panel.png` — SHAP importance

**Trained Models:**
- `models/final_model_filter_l1_lr_elasticnet.pkl`
- `models/final_model_filter_l1_random_forest.pkl`
- `models/final_model_filter_l1_lightgbm.pkl`
- `models/final_model_pca_lr_elasticnet.pkl`
- `models/final_model_pca_random_forest.pkl`
- `models/final_model_pca_lightgbm.pkl`
- `models/final_model_compact_panel.pkl`

---

## 13. Deviations from Protocol

**Note:** This section documents any deviations from the planned protocol during execution.

### 13.1 Computational Adjustments

[To be completed post-training if applicable]

### 13.2 Convergence Issues

[To be completed post-training if applicable]

### 13.3 Other Protocol Modifications

[To be completed post-training if applicable]

---

## 14. Literature References

**Batch Effects and Preprocessing:**
- Zhang, X., Jonassen, I., & Goksøyr, A. (2020). Machine learning approaches for biomarker discovery using gene expression data. *Briefings in Bioinformatics*, *21*(5), 1655–1666. https://doi.org/10.1093/bib/bbz090
- Yu, M., Zhukov, M., & Balch, W. E. (2023). Identification of batch effects in large-scale RNA-seq data by principal component analysis. *Bioinformatics Advances*, *3*(1), vbad005. https://doi.org/10.1093/bioadv/vbad005

**Cross-Validation and Data Leakage:**
- Bradshaw, T. J., Huemann, Z., Hu, J., & Rahmim, A. (2023). A guide to cross-validation for artificial intelligence in medical imaging. *Radiology: Artificial Intelligence*, *5*(4), e220232. https://doi.org/10.1148/ryai.220232
- Wainer, J., & Cawley, G. (2021). Nested cross-validation when selecting classifiers is overzealous for most practical applications. *Expert Systems with Applications*, *182*, 115222. https://doi.org/10.1016/j.eswa.2021.115222
- Rosenblatt, J. D., Nadler, B., & Rabin, N. (2024). On the optimality of sample-based estimates of the expectation of the empirical minimizer. *Journal of Machine Learning Research*, *25*, 1–49. https://jmlr.org/papers/v25/22-0689.html

**Reporting Standards:**
- Collins, G. S., Moons, K. G., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., … & Logullo, P. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, *385*, e078378. https://doi.org/10.1136/bmj-2023-078378
- Riley, R. D., Debray, T. P., Collins, G. S., Archer, L., Ensor, J., van Smeden, M., & Snell, K. I. (2024). Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Statistics in Medicine*, *43*(7), 1325–1339. https://doi.org/10.1002/sim.9025
- Efthimiou, O., Seo, M., Chalkou, K., Debray, T. P., Egger, M., & Salanti, G. (2024). GetReal in mathematical modelling: a review of studies predicting drug effectiveness in the real world. *Research Synthesis Methods*, *15*(1), 137–156. https://doi.org/10.1002/jrsm.1695
- Liu, X., Cruz Rivera, S., Moher, D., Calvert, M. J., & Denniston, A. K. (2020). Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension. *The Lancet Digital Health*, *2*(10), e537–e548. https://doi.org/10.1016/S2589-7500(20)30218-1

**Feature Selection and Stability:**
- Bommert, A., Welchowski, T., Schmid, M., & Rahnenführer, J. (2022). Benchmark of filter methods for feature selection in high-dimensional gene expression survival data. *Briefings in Bioinformatics*, *23*(1), bbab354. https://doi.org/10.1093/bib/bbab354
- Hamraz, M., Jalab, H. A., Selamat, A., & Ibrahim, R. (2023). An efficient hybrid filter-wrapper metaheuristic-based gene selection method for high-dimensional datasets. *Genes*, *14*(3), 560. https://doi.org/10.3390/genes14030560
- Haftorn, K. L., Romanowska, J., Goksøyr, A., Magnus, P., Håberg, S. E., Page, C. M., … & Bohlin, J. (2023). Stability selection enhances feature selection and enables accurate prediction of gestational age using only five DNA methylation sites. *Clinical Epigenetics*, *15*(1), 184. https://doi.org/10.1186/s13148-023-01600-x
- Łukaszuk, T., Osowski, S., & Książek, W. (2024). Stability-based feature selection for multi-class cancer classification on gene expression data. *Cancers*, *16*(1), 224. https://doi.org/10.3390/cancers16010224

**Dimensionality Reduction:**
- Greenacre, M., Groenen, P. J., Hastie, T., D'Enza, A. I., Markos, A., & Tuzhilina, E. (2022). Principal component analysis. *Nature Reviews Methods Primers*, *2*(1), 100. https://doi.org/10.1038/s43586-022-00184-w
- Zhang, H., Wang, Y., Deng, C., Zhao, S., Zhang, P., Feng, J., … & Liu, Y. (2024). Challenges and future directions for representations of functional brain organization. *Nature Neuroscience*, *27*(1), 14–24. https://doi.org/10.1038/s41593-023-01484-6

**Model Calibration and Discrimination:**
- Richardson, A. M., Lidbury, B. A., & Badrick, T. (2024). Measuring the clinical utility of AI diagnostic classifiers: A systematic review and recommendations for reporting. *npj Digital Medicine*, *7*(1), 11. https://doi.org/10.1038/s41746-023-00991-1
- McDermott, M. B., Yan, T., Naumann, T., Hunt, N. L., Suresh, H., Szolovits, P., & Ghassemi, M. (2024). A comprehensive evaluation of calibration methods for language models in medical question answering. *Communications Medicine*, *4*(1), 25. https://doi.org/10.1038/s43856-024-00454-8

**Decision Curve Analysis:**
- Sadatsafavi, M., Petkau, J., Adibi, A., Safari, A., Husereau, D., & Shahzad, U. (2021). A review of decision curve analysis with applications to economic evaluation. *Value in Health*, *24*(6), 885–892. https://doi.org/10.1016/j.jval.2020.12.015
- Vickers, A. J., Van Calster, B., & Steyerberg, E. W. (2023). A simple, step-by-step guide to interpreting decision curve analysis. *Diagnostic and Prognostic Research*, *7*(1), 18. https://doi.org/10.1186/s41512-023-00155-4

**Model Comparison:**
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 507–520). Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

**Clinical Context:**
- Ostrom, Q. T., Price, M., Neff, C., Cioffi, G., Waite, K. A., Kruchko, C., & Barnholtz-Sloan, J. S. (2023). CBTRUS statistical report: Primary brain and other central nervous system tumors diagnosed in the United States in 2016–2020. *Neuro-Oncology*, *25*(Supplement_4), iv1–iv99. https://doi.org/10.1093/neuonc/noad149
- World Health Organization. (2021). *WHO classification of tumours of the central nervous system* (5th ed.). International Agency for Research on Cancer.

---

## 15. Sign-Off & Version Control

**Protocol Version:** 1.3  
**Date:** November 10, 2025  
**Institution:** Hogeschool Rotterdam, Minor AI in Healthcare

**Change Log:**
- v1.3: Added computationally feasible configuration variant (3×3 nested CV) with academic justification. Maintains all core requirements while enabling completion within retake timeline.
- v1.2: Clarified validation policy (nested CV only), switched to ComBat for normalized data, added RF `max_features`, LightGBM `scale_pos_weight`, PCA variance logging, DCA threshold band, and confounding control note in data sources.

---

**End of Protocol**
