#!/usr/bin/env python3
"""
Convert old data format to new pipeline format.
This script converts the old combined_labeled_standardized.csv to the new format.
"""

import pandas as pd
import numpy as np

# Load old format
df = pd.read_csv("data/raw/combined_labeled_standardized.csv")

# Extract gene expression matrix
gene_cols = [c for c in df.columns if c not in ("patient_id", "healthy")]
expression_df = df[["patient_id"] + gene_cols].copy()
expression_df.rename(columns={"patient_id": "sample_id"}, inplace=True)

# Create metadata
metadata_df = df[["patient_id", "healthy"]].copy()
metadata_df.rename(columns={"patient_id": "sample_id", "healthy": "label"}, inplace=True)

# Add batch column (we'll assign based on healthy vs unhealthy)
metadata_df["batch"] = metadata_df["label"].map({1: "healthy_batch", 0: "tumor_batch"})

# Save in new format
expression_df.to_csv("data/raw/gene_expression.csv", index=False)
metadata_df.to_csv("data/raw/metadata.csv", index=False)

print(f"✓ Created gene_expression.csv: {expression_df.shape}")
print(f"✓ Created metadata.csv: {metadata_df.shape}")
print(f"✓ Samples: {len(expression_df)}")
print(f"✓ Genes: {len(gene_cols)}")
print(f"✓ Healthy: {(metadata_df['label']==1).sum()}")
print(f"✓ Tumor: {(metadata_df['label']==0).sum()}")
