# GBM Classification from Gene Expression

## 🔬 Project Overview
This project builds a machine learning pipeline to classify Glioblastoma Multiforme (GBM) tumor samples versus normal brain tissue using gene expression data.


## 🧬 Data Sources
- **Normal samples**: GTEx TPM matrix (`gene_tpm_v10_brain_cerebellum.gct`), plaats in `data/raw/` (niet in versiebeheer)
- **Tumor samples**: TCGA-GBM van Genomic Data Commons (GDC), download en plaats in `data/raw/` (niet in versiebeheer)


## 📁 Project Structure
```
data/
  raw/              # Original data files (.gct, .csv) (niet in git)
  processed/        # Cleaned, normalized data (niet in git)
notebooks/          # Jupyter notebooks voor elke stap
scripts/            # Python scripts (preprocessing, modeling, etc.)
results/            # Plots, metrics, and selected genes
models/             # Saved model objects
```


## 📦 Environment Setup
Clone deze repository en maak de Conda-omgeving aan met:

```bash
conda env create -f environment.yml
conda activate gbm-classificatie
```

De belangrijkste packages zijn:
- python=3.10
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter
- pycombat

Zie `environment.yml` voor de volledige lijst.

## 🚀 Workflow
1. Load and clean data
2. Exploratory analysis (PCA, clustering)
3. Normalize and correct batch effects
4. Train Random Forest & benchmark models
5. Select top features (genes)
6. Evaluate performance & interpret genes

---

## 📌 Notes
- Clinical labels must be merged manually using GTEx/TCGA metadata.
- Tumor data from TCGA must be harmonized (gene IDs, format).

## 🧠 Authors
- Musab Sivrikaya
- Jim Tronchet
- Ozeir Moradi
- Tim Grootscholten
- Tobias Roessingh

## 📜 License
MIT License
