"""
Feature Stability Analysis
===========================
Analyzes feature selection stability via bootstrap resampling.

Author: Musab 0988932
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import logging

from src.data import GeneExpressionDataLoader
from src.features import FilterL1Selector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bootstrap_stability_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    selector_params: Dict,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform bootstrap stability selection.
    
    Parameters:
        X: Feature matrix (samples × genes)
        y: Target labels
        selector_params: Parameters for FilterL1Selector
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction of samples per bootstrap
        random_state: Random seed
        
    Returns:
        DataFrame with columns [gene, selection_frequency, mean_importance]
    """
    np.random.seed(random_state)
    n_samples = len(X)
    sample_size = int(n_samples * sample_fraction)
    
    feature_counts = {}
    feature_importance_sum = {}
    
    logger.info(f"Running {n_bootstrap} bootstrap iterations...")
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap stability"):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y[indices]
        
        # Fit selector
        selector = FilterL1Selector(**selector_params)
        try:
            selector.fit(X_boot, y_boot)
            
            # Track selected features
            if selector.selected_features_ is not None:
                for feat, importance in zip(
                    selector.selected_features_,
                    selector.feature_importance_
                ):
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
                    feature_importance_sum[feat] = (
                        feature_importance_sum.get(feat, 0) + abs(importance)
                    )
        except Exception as e:
            logger.warning(f"Bootstrap {i+1} failed: {e}")
            continue
    
    # Calculate statistics
    stability_data = []
    for gene in feature_counts:
        frequency = feature_counts[gene] / n_bootstrap
        mean_importance = feature_importance_sum[gene] / feature_counts[gene]
        stability_data.append({
            'gene': gene,
            'selection_frequency': frequency,
            'mean_importance': mean_importance
        })
    
    df = pd.DataFrame(stability_data)
    df = df.sort_values('selection_frequency', ascending=False).reset_index(drop=True)
    
    logger.info(f"Found {len(df)} unique features across {n_bootstrap} iterations")
    logger.info(f"Top gene: {df.iloc[0]['gene']} (freq={df.iloc[0]['selection_frequency']:.2f})")
    
    return df


def plot_stability_bar(
    stability_df: pd.DataFrame,
    top_n: int = 50,
    save_path: Path = None
):
    """
    Plot bar chart of top N most stable features.
    
    Parameters:
        stability_df: DataFrame from bootstrap_stability_selection
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    top_features = stability_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bars = ax.barh(
        range(len(top_features)),
        top_features['selection_frequency'],
        color=plt.cm.viridis(top_features['selection_frequency'])
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['gene'], fontsize=8)
    ax.set_xlabel('Selection Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Top {top_n} Most Stable Features\n(Bootstrap N=100)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add frequency labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(
            row['selection_frequency'] + 0.02,
            i,
            f"{row['selection_frequency']:.2f}",
            va='center',
            fontsize=7
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stability bar chart to {save_path}")
    
    plt.close()


def plot_stability_heatmap(
    stability_df: pd.DataFrame,
    top_n: int = 50,
    save_path: Path = None
):
    """
    Plot heatmap of selection frequency and importance.
    
    Parameters:
        stability_df: DataFrame from bootstrap_stability_selection
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    top_features = stability_df.head(top_n).copy()
    
    # Normalize importance for visualization
    top_features['importance_norm'] = (
        top_features['mean_importance'] / top_features['mean_importance'].max()
    )
    
    # Create matrix for heatmap
    matrix = top_features[['selection_frequency', 'importance_norm']].T
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(
        matrix,
        cmap='RdYlGn',
        annot=False,
        cbar_kws={'label': 'Normalized Score'},
        yticklabels=['Selection Frequency', 'Mean Importance (norm)'],
        xticklabels=top_features['gene'],
        ax=ax
    )
    
    ax.set_title(
        f'Feature Stability Heatmap (Top {top_n})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Gene', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved stability heatmap to {save_path}")
    
    plt.close()


def main():
    """Main execution function."""
    # Load config
    config_path = Path("config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_dir = Path("figures/modeling")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    logger.info("Loading processed data...")
    loader = GeneExpressionDataLoader()
    X, y, metadata = loader.load_data(
        expression_path="data/processed/expression_processed.csv",
        metadata_path="data/processed/metadata_processed.csv"
    )
    
    logger.info(f"Loaded data: {X.shape[0]} samples × {X.shape[1]} genes")
    
    # Get selector parameters from config
    selector_params = {
        'k_best': config['feature_selection']['filter_l1']['k_best'],
        'variance_threshold': config['feature_selection']['variance_threshold'],
        'correlation_threshold': config['feature_selection']['correlation_threshold'],
        'C': 1.0  # Default for stability analysis
    }
    
    logger.info(f"Selector params: {selector_params}")
    
    # Run bootstrap stability selection
    stability_df = bootstrap_stability_selection(
        X=X,
        y=y,
        selector_params=selector_params,
        n_bootstrap=100,
        sample_fraction=0.8,
        random_state=42
    )
    
    # Save results
    csv_path = output_dir / "stability_panel.csv"
    stability_df.to_csv(csv_path, index=False)
    logger.info(f"Saved stability panel to {csv_path}")
    
    # Create visualizations
    plot_stability_bar(
        stability_df,
        top_n=50,
        save_path=fig_dir / "stability_top50_bar.png"
    )
    
    plot_stability_heatmap(
        stability_df,
        top_n=50,
        save_path=fig_dir / "stability_top50_heatmap.png"
    )
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("STABILITY ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total unique features selected: {len(stability_df)}")
    logger.info(f"Features selected in ≥50% of bootstraps: {(stability_df['selection_frequency'] >= 0.5).sum()}")
    logger.info(f"Features selected in 100% of bootstraps: {(stability_df['selection_frequency'] == 1.0).sum()}")
    logger.info(f"\nTop 10 most stable features:")
    for i, row in stability_df.head(10).iterrows():
        logger.info(f"  {i+1}. {row['gene']} (freq={row['selection_frequency']:.3f}, importance={row['mean_importance']:.4f})")


if __name__ == "__main__":
    main()
