#!/usr/bin/env python3
"""
Data Preprocessing Script
=========================
This script loads raw gene expression data, performs quality checks,
and saves processed data for downstream analysis.

Usage:
    python scripts/make_processed.py --config config.yaml

Author: Musab 0988932
Date: November 2025
"""

import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    GeneExpressionDataLoader,
    check_data_quality,
    remove_duplicates,
    filter_low_variance_genes
)
from src.plots import plot_class_distribution, plot_pca_variance
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """
    Main preprocessing pipeline.
    
    Parameters:
        config_path: Path to configuration YAML file.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    processed_dir = Path(config['paths']['processed'])
    figures_dir = Path(config['paths']['figures']) / 'eda'
    reports_dir = Path(config['paths']['reports']) / 'tables'
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load raw data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 60)
    
    loader = GeneExpressionDataLoader(
        data_path=config['paths']['raw']
    )
    
    try:
        expression_df, metadata_df = loader.load_data()
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Please ensure gene_expression.csv and metadata.csv are in data/raw/")
        logger.info("Creating example template files...")
        
        # Create template files
        create_template_data(Path(config['paths']['raw']))
        logger.info("Template files created. Please replace with your actual data.")
        return
    
    logger.info(f"Loaded {expression_df.shape[0]} samples with {expression_df.shape[1]} genes")
    
    # Step 2: Quality checks
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Data quality checks")
    logger.info("=" * 60)
    
    quality_report = check_data_quality(expression_df, metadata_df)
    
    # Save quality report
    quality_df = pd.DataFrame([quality_report]).T
    quality_df.columns = ['Value']
    quality_df.to_csv(reports_dir / 'data_quality_report.csv')
    logger.info(f"Quality report saved to {reports_dir / 'data_quality_report.csv'}")
    
    # Step 3: Clean data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Data cleaning")
    logger.info("=" * 60)
    
    # Remove duplicates
    expression_df, metadata_df = remove_duplicates(expression_df, metadata_df)
    
    # Remove low-variance genes
    variance_threshold = config['preprocessing'].get('variance_threshold', 0.01)
    expression_df = filter_low_variance_genes(expression_df, threshold=variance_threshold)
    
    logger.info(f"After cleaning: {expression_df.shape[0]} samples × {expression_df.shape[1]} genes")
    
    # Step 4: Exploratory visualizations
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Exploratory data analysis")
    logger.info("=" * 60)
    
    # Class distribution
    if 'label' in metadata_df.columns:
        y = metadata_df['label'].values
        class_labels = {0: 'Control', 1: 'GBM'}
        
        plot_class_distribution(
            y,
            labels=class_labels,
            title='Class Distribution - GBM vs Control',
            save_path=figures_dir / 'class_distribution.png'
        )
        logger.info("Created class distribution plot")
    
    # PCA for visualization
    logger.info("Computing PCA...")
    pca = PCA(n_components=min(50, expression_df.shape[1]))
    pca.fit(expression_df)
    
    plot_pca_variance(
        pca.explained_variance_ratio_,
        n_components=min(50, len(pca.explained_variance_ratio_)),
        title='PCA Explained Variance',
        save_path=figures_dir / 'pca_variance.png'
    )
    logger.info("Created PCA variance plot")
    
    # Step 5: Save processed data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Saving processed data")
    logger.info("=" * 60)
    
    expression_df.to_csv(processed_dir / 'expression_processed.csv')
    metadata_df.to_csv(processed_dir / 'metadata_processed.csv')
    
    logger.info(f"Saved processed data to {processed_dir}/")
    logger.info(f"  - expression_processed.csv: {expression_df.shape}")
    logger.info(f"  - metadata_processed.csv: {metadata_df.shape}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final dataset: {expression_df.shape[0]} samples × {expression_df.shape[1]} genes")
    logger.info(f"Quality report: {reports_dir / 'data_quality_report.csv'}")
    logger.info(f"Figures: {figures_dir}/")
    logger.info(f"Processed data: {processed_dir}/")


def create_template_data(data_path: Path):
    """
    Create template data files for demonstration.
    
    Parameters:
        data_path: Path to save template files.
    """
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create example gene expression data
    np.random.seed(42)
    n_samples = 100
    n_genes = 1000
    
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    
    # Simulate gene expression with some structure
    expression_data = np.random.randn(n_samples, n_genes) + 5
    expression_df = pd.DataFrame(
        expression_data,
        index=sample_ids,
        columns=gene_names
    )
    expression_df.index.name = 'sample_id'
    
    expression_df.to_csv(data_path / 'gene_expression.csv')
    logger.info(f"Created template: {data_path / 'gene_expression.csv'}")
    
    # Create example metadata
    metadata_df = pd.DataFrame({
        'sample_id': sample_ids,
        'label': np.random.choice([0, 1], n_samples),
        'batch': np.random.choice(['A', 'B', 'C'], n_samples),
        'age': np.random.randint(30, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples)
    })
    
    metadata_df.to_csv(data_path / 'metadata.csv', index=False)
    logger.info(f"Created template: {data_path / 'metadata.csv'}")
    
    # Create example bio panel
    bio_panel_genes = np.random.choice(gene_names, 50, replace=False)
    bio_panel_df = pd.DataFrame({'gene': bio_panel_genes})
    
    metadata_path = Path('metadata')
    metadata_path.mkdir(exist_ok=True)
    bio_panel_df.to_csv(metadata_path / 'biopanel.csv', index=False)
    logger.info(f"Created template: {metadata_path / 'biopanel.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess gene expression data for GBM classification'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    main(args.config)
