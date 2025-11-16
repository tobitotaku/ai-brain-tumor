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

## Project Structure

```
data/
  raw/              # Original data files
  processed/        # Cleaned, normalized data
*.ipynb             # Analysis notebooks
```

## Authors

- Musab Sivrikaya
- Jim Tronchet
- Ozeir Moradi
- Tim Grootscholten
- Tobias Roessingh
