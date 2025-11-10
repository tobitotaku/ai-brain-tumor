"""
SHAP Analysis on Compact Gene Panel (Config-Driven)
===================================================
Performs feature importance analysis based on trained models in reports/tables.
No hardcoded assumptions about gene selection method.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import logging
from typing import List, Tuple

from src.data import GeneExpressionDataLoader
from src.pipeline import build_pipeline
from src.eval import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_feature_importance_from_models(
    models_dir: Path,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Extract feature importance from saved models in models/ directory.
    Works with Random Forest, LightGBM, or other tree-based models.
    
    Parameters:
        models_dir: Path to models directory
        top_n: Number of top features to select
        
    Returns:
        DataFrame with features and their mean importance scores
    """
    logger.info("Loading trained models to extract feature importances...")
    
    all_importances = {}
    model_count = 0
    
    # Look for final trained models
    for model_file in models_dir.glob("final_model_pca_*.pkl"):
        try:
            logger.info(f"Loading {model_file.name}...")
            with open(model_file, 'rb') as f:
                loaded = pickle.load(f)
            
            # Handle both dict and direct pipeline formats
            if isinstance(loaded, dict):
                pipeline = loaded.get('pipeline')
            else:
                pipeline = loaded
            
            if pipeline is None:
                logger.warning(f"Could not extract pipeline from {model_file.name}")
                continue
            
            # Extract the classifier from pipeline
            classifier = pipeline.named_steps.get('classifier')
            if classifier is None:
                logger.warning(f"No classifier in {model_file.name}, skipping...")
                continue
            
            # Get feature importances based on model type
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Generate feature names - use PC if PCA is in pipeline
                feature_selector = pipeline.named_steps.get('feature_selector')
                if hasattr(feature_selector, 'n_components'):
                    # PCA selector
                    feature_names = [f'PC{i}' for i in range(len(importances))]
                else:
                    # Try to get from pipeline, otherwise use generic names
                    try:
                        feature_names = pipeline[:-1].get_feature_names_out()
                    except:
                        feature_names = [f'Feature{i}' for i in range(len(importances))]
                
                for name, score in zip(feature_names, importances):
                    if name not in all_importances:
                        all_importances[name] = []
                    all_importances[name].append(float(score))
                
                model_count += 1
                logger.info(f"  Extracted {len(importances)} feature importances")
            else:
                logger.warning(f"Model {model_file.name} has no feature_importances_ attribute")
                
        except Exception as e:
            logger.warning(f"Failed to load {model_file.name}: {str(e)[:100]}")
            continue
    
    if not all_importances:
        logger.error("Could not extract any feature importances from models!")
        return pd.DataFrame()
    
    # Calculate mean importance across models
    importance_data = []
    for feature, scores in all_importances.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        importance_data.append({
            'feature': feature,
            'mean_importance': mean_score,
            'std_importance': std_score,
            'n_models': len(scores)
        })
    
    df = pd.DataFrame(importance_data)
    df = df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"Extracted importances from {model_count} models")
    logger.info(f"Total features: {len(df)}")
    logger.info(f"Top 10 features:")
    for idx, row in df.head(10).iterrows():
        logger.info(f"  {idx+1}. {row['feature']:20s} | Importance: {row['mean_importance']:.6f}")
    
    return df


def plot_feature_importance(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30
):
    """Create visualizations for feature importance."""
    if len(df) == 0:
        logger.warning("No data to plot")
        return
    
    df_top = df.head(top_n)
    
    # Bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(df_top)), df_top['mean_importance'], 
            xerr=df_top['std_importance'], color='steelblue', capsize=5)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=9)
    ax.set_xlabel('Mean Importance (±std)', fontsize=11)
    ax.set_title(f'Top {top_n} Most Important Features (Random Forest)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance_compact_panel.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to {output_path / 'feature_importance_compact_panel.png'}")
    plt.close()
    
    # Line plot showing importance decline
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(min(100, len(df))), df['mean_importance'].head(100), 
            marker='o', linestyle='-', linewidth=2, markersize=4, color='steelblue')
    ax.fill_between(range(min(100, len(df))), 
                     df['mean_importance'].head(100), 
                     alpha=0.3, color='steelblue')
    ax.set_xlabel('Feature Rank', fontsize=11)
    ax.set_ylabel('Mean Importance', fontsize=11)
    ax.set_title('Feature Importance Ranking (Top 100)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved importance ranking plot to {output_path / 'feature_importance_ranking.png'}")
    plt.close()


def main():
    """Main execution function."""
    # Load config
    config_path = Path("config/config_ultrafast_pca.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    # Create output directories
    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_dir = Path("figures/shap")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract feature importances from models
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error(f"Models directory not found at {models_dir}")
        logger.info("Please run training first: utilities/run_training.sh config/config_ultrafast_pca.yaml")
        return
    
    importance_df = extract_feature_importance_from_models(models_dir, top_n=30)
    
    if len(importance_df) == 0:
        logger.error("Could not extract feature importances. Exiting.")
        return
    
    # Save results
    output_file = output_dir / 'feature_importance_summary.csv'
    importance_df.to_csv(output_file, index=False)
    logger.info(f"Saved feature importance summary to {output_file}")
    
    # Create visualizations
    plot_feature_importance(importance_df, fig_dir, top_n=30)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("Feature Importance Analysis Summary:")
    logger.info("="*60)
    logger.info(f"Total features analyzed: {len(importance_df)}")
    logger.info(f"Mean importance (top 10): {importance_df.head(10)['mean_importance'].mean():.6f}")
    logger.info(f"Mean importance (all): {importance_df['mean_importance'].mean():.6f}")
    logger.info(f"\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {idx+1}. {row['feature']:20s} | Score: {row['mean_importance']:.6f} (±{row['std_importance']:.6f})")
    
    logger.info("\n" + "="*60)
    logger.info("Feature importance analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
