# GBM Gene Expression Dataset Card

## Dataset Description

### Dataset Summary
This dataset contains gene expression profiles from Glioblastoma (GBM) patients and healthy control samples. The data is used for binary classification to distinguish GBM samples from healthy tissue based on transcriptomic signatures.

**Dataset Name:** GBM Gene Expression Dataset  
**Version:** 1.0  
**Last Updated:** November 2025  
**License:** Public domain (derived from public repositories)  
**Contact:** https://github.com/tobitotaku/ai-brain-tumor

### Dataset Characteristics
- **Task:** Binary Classification (GBM vs. Healthy)
- **Data Type:** Gene Expression (standardized values)
- **Number of Samples:** 18,635 total
  - GBM: 1,253 samples (6.7%)
  - Healthy: 17,382 samples (93.3%)
- **Number of Genes:** 18,752 (after quality filtering from 18,858)
- **File Format:** CSV (samples × genes)

---

## Data Collection

### Source
Data aggregated from public gene expression repositories. Original cohort details not disclosed to maintain academic project scope.

### Collection Methodology
- **Technology:** Gene expression profiling (standardized format)
- **Tissue Type:** Brain tissue (tumor and healthy)
- **Collection Period:** Pre-2025 (archived public data)
- **Inclusion Criteria:**
  - Gene expression data available with ENSEMBL IDs
  - Binary classification labels (GBM vs healthy)
  - Batch information for correction

### Exclusion Criteria
- Samples with excessive missing values (>5%)
- Low-variance genes (variance < 0.01)
- Highly correlated duplicate genes (r > 0.95)

---

## Data Preprocessing

### Raw Data Processing
1. **Quality Control:**
   - Removal of duplicate samples (0 found)
   - Filtering of low-variance genes (106 removed)
   - Variance threshold: 0.01

2. **Normalization:**
   - Data received pre-standardized
   - Additional StandardScaler applied within CV folds

3. **Batch Effect Correction:**
   - Method: ComBat
   - Applied only on training data within each CV fold (no leakage)

4. **Missing Data:**
   - Handling strategy: None required (0% missing values)
   - All samples complete

---

## Data Structure

### Gene Expression Matrix
- **Rows:** Samples (patient IDs)
- **Columns:** Genes (gene symbols or Ensembl IDs)
- **Values:** Normalized expression values

### Metadata
The dataset includes the following clinical variables:
| Variable | Description | Type | Values/Range |
|----------|-------------|------|--------------|
| sample_id | Unique sample identifier | Categorical | patient_1 to patient_18635 |
| label | Disease status | Binary | 0 = Healthy, 1 = GBM |
| batch | Technical batch | Categorical | Batch identifiers (for ComBat) |

---

## Data Splits

### Training/Validation Strategy
- **Method:** Stratified K-Fold Cross-Validation
- **Folds:** 5 (outer) × 3 (inner) nested CV
- **Stratification:** By class label
- **Random Seed:** 42

### Class Distribution
| Split | Healthy | GBM | Total | Imbalance Ratio |
|-------|---------|-----|-------|-----------------|
| Full Dataset | 17,382 | 1,253 | 18,635 | 13.9:1 |

## Known Limitations & Biases

### Data Limitations
1. **Sample Size:**
   - Limited number of samples may affect generalizability
   - Potential class imbalance

2. **Technical Limitations:**
   - Batch effects present across different collection sites
   - Platform-specific biases (if applicable)

3. **Biological Limitations:**
   - Tumor heterogeneity not fully captured
   - Limited representation of GBM subtypes

### Potential Biases
- **Selection Bias:** Data sourced from publicly available repositories may not represent general population
- **Demographic Bias:** Patient demographics (age, gender, ethnicity) not available in aggregated dataset
- **Technical Bias:** Batch effects from different sequencing platforms partially addressed through ComBat correction

---

## Ethical Considerations

### Patient Privacy
- All data has been de-identified
- No personally identifiable information (PII) included
- Derived from publicly available research datasets with appropriate data use agreements

### Informed Consent
- Original data collection followed institutional ethics protocols
- Public repositories ensure appropriate consent and ethics approval for secondary research use

### Sensitive Information
- The dataset does not contain sensitive genetic information beyond gene expression
- Results should not be used for individual patient diagnosis without clinical validation

---

## Usage & Citation

### Recommended Use Cases
**Appropriate Uses:**
- Research on GBM biomarker discovery
- Development of gene expression-based classifiers
- Validation of GBM-related pathways
- Educational purposes in bioinformatics

**Inappropriate Uses:**
- Direct clinical diagnosis without validation
- Making treatment decisions without physician oversight
- Generalizing to other cancer types without validation

### Citation
This dataset aggregates publicly available TCGA and GTEx data. Please cite the original sources:

- TCGA Research Network: https://www.cancer.gov/tcga
- GTEx Consortium: https://www.gtexportal.org

---

## Dataset Statistics

### Gene Expression Distribution
- **Original Genes:** 18,858
- **After Variance Filtering:** 18,752 (106 removed)
- **Variance Threshold:** 0.01
- **Correlation Threshold:** 0.95 (applied within feature selection)

### Quality Metrics
- **Missing Values:** 0%
- **Duplicate Samples:** 0
- **Outlier Detection:** Not applied (preserves biological variation)

## Data Access & Updates

### Access
- **Location:** `data/raw/` and `data/processed/`
- **Files:**
  - `gene_expression.csv` - Raw gene expression matrix
  - `metadata.csv` - Clinical metadata
  - `expression_processed.csv` - Processed expression data

### Updates
- **Version 1.0:** Initial release (November 2025)
- Future updates will be documented here

---

## Contact & Support

For questions about this dataset:
- **Institution:** Hogeschool Rotterdam - Minor AI in Healthcare
- **GitHub:** https://github.com/tobitotaku/ai-brain-tumor

---

**Last Updated:** November 9, 2025  
**Version:** 1.0