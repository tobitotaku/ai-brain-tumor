"""
Feature Stability Analysis (Config-Driven)
===========================================
Analyzes feature selection stability based on configured feature routes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import logging

from src.data import GeneExpressionDataLoader
from src.features import FilterL1Selector, PCASelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_feature_selector(route: str, config: Dict):
    """Get the appropriate feature selector based on config route."""
    if route == "filter_l1":
        return FilterL1Selector(
            k_best=config['features']['k_best'],
            variance_threshold=config['preprocessing']['variance_threshold'],
            correlation_threshold=config['preprocessing'].get('correlation_threshold', 0.95),
            C=1.0
        )
    elif route == "pca":
        return PCASelector(
            n_components=config['features']['pca_components']
        )
    else:
        raise ValueError(f"Unknown feature route: {route}")


def bootstrap_stability_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    selector,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform bootstrap stability selection with given selector.
    
    Parameters:
        X: Feature matrix (samples × genes)
        y: Target labels
        selector: Feature selector instance
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction of samples per bootstrap
        random_state: Random seed
        
    Returns:
        DataFrame with columns [feature, selection_frequency, mean_importance]
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
        y_boot = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]
        
        # Fit selector
        try:
            selector_copy = selector.__class__(**selector.get_params())
            selector_copy.fit(X_boot, y_boot)
            
            # Track selected features
            if hasattr(selector_copy, 'selected_features_') and selector_copy.selected_features_ is not None:
                selected = selector_copy.selected_features_
                importances = getattr(selector_copy, 'feature_importance_', [1.0] * len(selected))
                
                for feat, importance in zip(selected, importances):
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
                    feature_importance_sum[feat] = (
                        feature_importance_sum.get(feat, 0) + abs(importance)
                    )
        except Exception as e:
            logger.warning(f"Bootstrap {i+1} failed: {str(e)[:100]}")
            continue
    
    if not feature_counts:
        logger.error("No features selected in any bootstrap iteration!")
        return pd.DataFrame()
    
    # Calculate statistics
    stability_data = []
    for feature in feature_counts:
        frequency = feature_counts[feature] / n_bootstrap
        mean_importance = feature_importance_sum[feature] / feature_counts[feature]
        stability_data.append({
            'feature': feature,
            'selection_frequency': frequency,
            'mean_importance': mean_importance
        })
    
    df = pd.DataFrame(stability_data)
    df = df.sort_values('selection_frequency', ascending=False).reset_index(drop=True)
    
    logger.info(f"Found {len(df)} unique features across {n_bootstrap} iterations")
    if len(df) > 0:
        logger.info(f"Top feature: {df.iloc[0]['feature']} (freq={df.iloc[0]['selection_frequency']:.2f})")
    
    return df


def plot_stability_analysis(df: pd.DataFrame, output_path: Path, top_n: int = 50):
    """Create visualizations for stability analysis."""
    if len(df) == 0:
        logger.warning("No data to plot")
        return
    
    df_top = df.head(top_n)
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(df_top)), df_top['selection_frequency'], color='steelblue')
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=8)
    ax.set_xlabel('Selection Frequency', fontsize=10)
    ax.set_title(f'Top {top_n} Most Stable Features (Bootstrap n=100)', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'stability_top50_bar.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved bar plot to {output_path / 'stability_top50_bar.png'}")
    plt.close()
    
    # Heatmap of top features
    if len(df_top) > 5:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = df_top[['feature', 'selection_frequency']].head(30)
        stability_matrix = np.expand_dims(data['selection_frequency'].values, axis=0)
        sns.heatmap(stability_matrix, cmap='YlGn', cbar_kws={'label': 'Frequency'}, ax=ax)
        ax.set_xticklabels(data['feature'].values, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(['Stability'])
        ax.set_title('Top 30 Features - Stability Heatmap', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'stability_top50_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path / 'stability_top50_heatmap.png'}")
        plt.close()


def main():
    """Main execution function."""
    # Load config
    config_path = Path("config/config_ultrafast_pca.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Feature routes: {config['features']['routes']}")
    
    # Create output directories
    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_dir = Path("figures/modeling")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    logger.info("Loading processed data...")
    loader = GeneExpressionDataLoader()
    X, metadata = loader.load_data()
    y = metadata['label']
    
    logger.info(f"Loaded data: {X.shape[0]} samples × {X.shape[1]} genes")
    
    # For each configured feature route
    for route in config['features']['routes']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing stability for route: {route}")
        logger.info(f"{'='*60}")
        
        # Get selector for this route
        selector = get_feature_selector(route, config)
        logger.info(f"Using selector: {selector.__class__.__name__}")
        
        # Run bootstrap stability selection
        stability_df = bootstrap_stability_selection(
            X=X,
            y=y,
            selector=selector,
            n_bootstrap=100,
            sample_fraction=0.8,
            random_state=42
        )
        
        if len(stability_df) == 0:
            logger.warning(f"No results for route {route}, skipping...")
            continue
        
        # Save results
        output_file = output_dir / f'stability_panel_{route}.csv'
        stability_df.to_csv(output_file, index=False)
        logger.info(f"Saved stability results to {output_file}")
        
        # Create visualizations
        plot_stability_analysis(stability_df, fig_dir, top_n=50)
        
        # Print summary
        logger.info(f"\nStability Analysis Summary ({route}):")
        logger.info(f"  Total unique features: {len(stability_df)}")
        logger.info(f"  Features selected ≥70%: {len(stability_df[stability_df['selection_frequency'] >= 0.7])}")
        logger.info(f"  Features selected ≥50%: {len(stability_df[stability_df['selection_frequency'] >= 0.5])}")
        logger.info(f"  Top 10 features:")
        for idx, row in stability_df.head(10).iterrows():
            logger.info(f"    {idx+1}. {row['feature']:20s} | Freq: {row['selection_frequency']:.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("Stability analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
