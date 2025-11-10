"""
Data Loading Module
===================
This module handles loading and initial validation of gene expression data
and associated metadata for GBM classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneExpressionDataLoader:
    """
    Load and validate gene expression data and metadata.
    
    This class handles loading of gene expression matrices and clinical metadata,
    performs basic validation, and ensures data integrity for downstream analysis.
    
    Attributes:
        data_path (Path): Path to the raw data directory.
        expression_file (str): Name of the gene expression file.
        metadata_file (str): Name of the metadata file.
    """
    
    def __init__(
        self,
        data_path: str = "data/raw",
        expression_file: str = "gene_expression.csv",
        metadata_file: str = "metadata.csv"
    ):
        """
        Initialize the data loader.
        
        Parameters:
            data_path: Path to directory containing raw data files.
            expression_file: Filename for gene expression matrix.
            metadata_file: Filename for clinical metadata.
        """
        self.data_path = Path(data_path)
        self.expression_file = expression_file
        self.metadata_file = metadata_file
        
    def load_expression_data(self) -> pd.DataFrame:
        """
        Load gene expression data from CSV file.
        
        Expected format: rows = samples, columns = genes.
        First column should contain sample IDs.
        
        Returns:
            DataFrame with gene expression values (samples × genes).
            
        Raises:
            FileNotFoundError: If expression file does not exist.
            ValueError: If data format is invalid.
        """
        file_path = self.data_path / self.expression_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Expression file not found: {file_path}")
        
        logger.info(f"Loading gene expression data from {file_path}")
        
        # Load data with first column as index (sample IDs)
        df = pd.read_csv(file_path, index_col=0)
        
        # Validate data
        if df.empty:
            raise ValueError("Expression data is empty")
        
        if not df.apply(lambda s: pd.to_numeric(s, errors='coerce').notna().all()).all():
            logger.warning("Non-numeric values detected in expression data")
        
        logger.info(f"Loaded expression data: {df.shape[0]} samples, {df.shape[1]} genes")
        
        return df
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load clinical metadata from CSV file.
        
        Expected format: rows = samples, columns = clinical variables.
        Must contain at least 'sample_id' and 'label' columns.
        
        Returns:
            DataFrame with clinical metadata.
            
        Raises:
            FileNotFoundError: If metadata file does not exist.
            ValueError: If required columns are missing.
        """
        file_path = self.data_path / self.metadata_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        
        logger.info(f"Loading metadata from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['sample_id', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in metadata: {missing_cols}")
        
        # Set sample_id as index
        df = df.set_index('sample_id')
        
        logger.info(f"Loaded metadata: {df.shape[0]} samples, {df.shape[1]} variables")
        
        return df
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both expression data and metadata.
        
        Returns:
            Tuple of (expression_df, metadata_df).
        """
        expression_df = self.load_expression_data()
        metadata_df = self.load_metadata()
        
        # Validate sample alignment
        common_samples = expression_df.index.intersection(metadata_df.index)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples between expression data and metadata")
        
        if len(common_samples) < len(expression_df):
            logger.warning(
                f"Sample mismatch: {len(expression_df)} expression samples, "
                f"{len(metadata_df)} metadata samples, {len(common_samples)} common"
            )
        
        # Align dataframes to common samples
        expression_df = expression_df.loc[common_samples]
        metadata_df = metadata_df.loc[common_samples]
        
        logger.info(f"Final aligned dataset: {len(common_samples)} samples")
        
        return expression_df, metadata_df


def load_gene_list(gene_list_path: str) -> List[str]:
    """
    Load a curated list of genes from a file.
    
    Expected format: CSV file with a column named 'gene' or 'gene_symbol'.
    
    Parameters:
        gene_list_path: Path to gene list file.
        
    Returns:
        List of gene symbols.
        
    Raises:
        FileNotFoundError: If gene list file does not exist.
    """
    path = Path(gene_list_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Gene list file not found: {path}")
    
    logger.info(f"Loading gene list from {path}")
    
    df = pd.read_csv(path)
    
    # Try to find gene column
    gene_col = None
    for col in ['gene', 'gene_symbol', 'Gene', 'Gene_Symbol']:
        if col in df.columns:
            gene_col = col
            break
    
    if gene_col is None:
        # Assume first column contains genes
        gene_col = df.columns[0]
        logger.warning(f"No standard gene column found, using first column: {gene_col}")
    
    genes = df[gene_col].dropna().unique().tolist()
    
    logger.info(f"Loaded {len(genes)} genes from list")
    
    return genes


def check_data_quality(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> Dict[str, any]:
    """
    Perform data quality checks on loaded data.
    
    Checks include:
    - Missing values
    - Duplicate samples
    - Class distribution
    - Value ranges
    
    Parameters:
        expression_df: Gene expression dataframe.
        metadata_df: Clinical metadata dataframe.
        
    Returns:
        Dictionary containing quality metrics.
    """
    quality_report = {}
    
    # Check missing values
    quality_report['expression_missing'] = expression_df.isna().sum().sum()
    quality_report['expression_missing_pct'] = (
        quality_report['expression_missing'] / expression_df.size * 100
    )
    
    quality_report['metadata_missing'] = metadata_df.isna().sum().to_dict()
    
    # Check duplicates
    quality_report['duplicate_samples'] = expression_df.index.duplicated().sum()
    quality_report['duplicate_genes'] = expression_df.columns.duplicated().sum()
    
    # Check class distribution
    if 'label' in metadata_df.columns:
        quality_report['class_distribution'] = metadata_df['label'].value_counts().to_dict()
        quality_report['class_balance'] = (
            metadata_df['label'].value_counts(normalize=True).to_dict()
        )
    
    # Check value ranges
    quality_report['expression_min'] = float(expression_df.min().min())
    quality_report['expression_max'] = float(expression_df.max().max())
    quality_report['expression_mean'] = float(expression_df.mean().mean())
    quality_report['expression_std'] = float(expression_df.std().mean())
    
    # Check for constant genes (zero variance)
    zero_var_genes = (expression_df.std() == 0).sum()
    quality_report['zero_variance_genes'] = int(zero_var_genes)
    
    logger.info("Data quality check completed")
    for key, value in quality_report.items():
        logger.info(f"  {key}: {value}")
    
    return quality_report


def remove_duplicates(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    strategy: str = "first"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate samples or genes from the dataset.
    
    Parameters:
        expression_df: Gene expression dataframe.
        metadata_df: Clinical metadata dataframe.
        strategy: How to handle duplicates ('first', 'last', 'mean').
        
    Returns:
        Tuple of cleaned (expression_df, metadata_df).
    """
    # Remove duplicate samples
    if expression_df.index.duplicated().any():
        logger.warning(f"Found {expression_df.index.duplicated().sum()} duplicate samples")
        
        if strategy == "mean":
            # Average expression values for duplicate samples
            expression_df = expression_df.groupby(expression_df.index).mean()
            metadata_df = metadata_df[~metadata_df.index.duplicated(keep='first')]
        else:
            expression_df = expression_df[~expression_df.index.duplicated(keep=strategy)]
            metadata_df = metadata_df[~metadata_df.index.duplicated(keep=strategy)]
    
    # Remove duplicate genes (keep first occurrence)
    if expression_df.columns.duplicated().any():
        logger.warning(f"Found {expression_df.columns.duplicated().sum()} duplicate genes")
        expression_df = expression_df.loc[:, ~expression_df.columns.duplicated(keep='first')]
    
    return expression_df, metadata_df


def filter_low_variance_genes(
    expression_df: pd.DataFrame,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Remove genes with variance below threshold.
    
    Low-variance genes provide little discriminative power and can be removed
    to reduce dimensionality.
    
    Parameters:
        expression_df: Gene expression dataframe.
        threshold: Minimum variance threshold.
        
    Returns:
        Filtered dataframe with high-variance genes only.
    """
    gene_variance = expression_df.var()
    high_var_genes = gene_variance[gene_variance >= threshold].index
    
    n_removed = len(expression_df.columns) - len(high_var_genes)
    logger.info(f"Removed {n_removed} low-variance genes (threshold={threshold})")
    
    return expression_df[high_var_genes]


if __name__ == "__main__":
    # Example usage
    loader = GeneExpressionDataLoader(data_path="data/raw")
    
    try:
        expression_df, metadata_df = loader.load_data()
        
        # Quality checks
        quality_report = check_data_quality(expression_df, metadata_df)
        
        # Clean data
        expression_df, metadata_df = remove_duplicates(expression_df, metadata_df)
        expression_df = filter_low_variance_genes(expression_df, threshold=0.01)
        
        print(f"\nFinal dataset: {expression_df.shape[0]} samples × {expression_df.shape[1]} genes")
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Please ensure data files are in the data/raw directory")
