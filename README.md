# GBM Classification from Gene Expression

## Project Overview

This project develops a machine learning pipeline to classify Glioblastoma Multiforme (GBM) tumor samples versus normal brain tissue using gene expression data.

## Data Sources

All data is sourced from the UCSC Xena platform:
- Main website: https://xena.ucsc.edu/
- Cohort used: TCGA TARGET GTEx (https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx)

The dataset includes:
- Normal brain tissue samples from GTEx
- Glioblastoma tumor samples from TCGA-GBM
- Lower-grade glioma samples from TCGA-LGG

## How to Run

Execute the notebooks in this order:

1. **`preprocessing.ipynb`** - Load raw TCGA/GTEx data and filter brain tissue samples
2. **`preprocessing_gbm_healthy.ipynb`** - Select GBM and healthy samples, perform batch correction
3. **`modeling_gbm_healthy.ipynb`** - Train and evaluate classification models

## Project Structure

```
data/
  raw/              # Original data files from UCSC Xena
  processed/        # Cleaned, normalized data
*.ipynb             # Analysis notebooks (run in order above)
```

## Authors

- Musab Sivrikaya
- Jim Tronchet
- Ozeir Moradi
- Tim Grootscholten
- Tobias Roessingh
