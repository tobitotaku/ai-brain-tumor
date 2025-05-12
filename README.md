# ai-brain-tumor

# GBM Classification from Gene Expression

## 🔬 Project Overview
This project builds a machine learning pipeline to classify Glioblastoma Multiforme (GBM) tumor samples versus normal brain tissue using gene expression data.

## 🧬 Data Sources
- **Normal samples**: GTEx TPM matrix (`gene_tpm_v10_brain_cerebellum.gct`)
- **Tumor samples**: TCGA-GBM from Genomic Data Commons (GDC)

## 📁 Project Structure
```
/data/
  raw/              # Original data files (.gct, .csv)
  processed/        # Cleaned, normalized data
  splits/           # Train/test splits
/notebooks/         # Jupyter notebooks for each stage
/scripts/           # Python scripts (preprocessing, modeling, etc.)
/results/           # Plots, metrics, and selected genes
/models/            # Saved model objects
```  

## 📦 Environment Setup
Use the following Conda environment for reproducibility:

```yaml
name: gbm-classifier
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - statsmodels
  - jupyterlab
  - umap-learn
  - openpyxl
  - pycombat
```

Save this as `environment.yml` and create your environment:
```bash
conda env create -f environment.yml
conda activate gbm-classifier
```

## 🚀 Workflow
1. Load and clean data
2. Exploratory analysis (PCA, clustering)
3. Normalize and correct batch effects
4. Train Random Forest & benchmark models
5. Select top features (genes)
6. Evaluate performance & interpret genes

---

## 📓 Starter Notebook: 01_load_gct.ipynb

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load GCT file (GTEx normal brain expression)
gct_file = '../data/raw/gene_tpm_v10_brain_cerebellum.gct'
with open(gct_file) as f:
    _ = [next(f) for _ in range(2)]  # Skip 2 metadata lines

# Read as TSV from line 3 onward
df = pd.read_csv(gct_file, sep='\t', skiprows=2)

# Preview
df.head()

# Basic stats
print(f"Number of genes: {df.shape[0]}")
print(f"Number of samples: {df.shape[1] - 2}")

# Plot TPM distribution
sample_cols = df.columns[2:]
df_log = df[sample_cols].apply(lambda x: np.log2(x + 1))
plt.figure(figsize=(10,5))
sns.boxplot(data=df_log, orient='h', fliersize=0.5)
plt.title('TPM Distribution (log2 transformed)')
plt.xlabel('log2(TPM + 1)')
plt.show()

# Save cleaned version
df.to_csv('../data/processed/brain_normal_tpm.csv', index=False)
```

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
