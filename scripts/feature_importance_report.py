#!/usr/bin/env python3
"""
Feature Importance Report (SHAP Alternative)
============================================
Generates feature importance analysis without SHAP dependency.

Usage:
    python scripts/feature_importance_report.py --config config.yaml

Author: Musab 0988932
Date: November 2025
"""

import sys
import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import GeneExpressionDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if it's a dict with 'pipeline' key
    if isinstance(model_data, dict) and 'pipeline' in model_data:
        return model_data['pipeline']
    return model_data


def get_feature_importance(model, feature_names, top_n=20):
    """Extract feature importance from tree-based model."""
    # Get the classifier from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier')
        if classifier is None:
            # Try to get last step
            steps = list(model.named_steps.values())
            classifier = steps[-1] if steps else model
    else:
        classifier = model
    
    # Get feature importances
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning(f"Model type {type(classifier)} does not have feature_importances_ or coef_")
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def plot_feature_importance(importance_df, output_path, title="Top Feature Importances"):
    """Plot feature importance bar chart."""
    plt.figure(figsize=(10, 8))
    
    # Reverse order for better visualization
    importance_df_plot = importance_df.iloc[::-1]
    
    plt.barh(range(len(importance_df_plot)), importance_df_plot['importance'])
    plt.yticks(range(len(importance_df_plot)), importance_df_plot['feature'])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved figure: {output_path}")


def main(config_path: str):
    """Main execution function."""
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    figures_dir = Path(config['paths']['figures'])
    reports_dir = Path(config['paths']['reports'])
    models_dir = Path('models')
    
    # Create directories
    (figures_dir / 'shap').mkdir(parents=True, exist_ok=True)
    (reports_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading best model")
    logger.info("=" * 60)
    
    # Find best model
    model_files = list(models_dir.glob("final_model_*.pkl"))
    if not model_files:
        logger.error("No trained models found!")
        return
    
    # Use the first model (or you can load the best from results)
    model_path = model_files[0]
    model_name = model_path.stem.replace('final_model_', '')
    logger.info(f"Loading model: {model_name}")
    
    model = load_model(model_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading processed data")
    logger.info("=" * 60)
    
    # Load data
    loader = GeneExpressionDataLoader(
        data_path=config['paths']['processed'],
        expression_file='expression_processed.csv',
        metadata_file='metadata_processed.csv'
    )
    expression, metadata = loader.load_data()
    X = expression.drop('sample_id', axis=1, errors='ignore')
    y = metadata['label'].values
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Extracting feature importance")
    logger.info("=" * 60)
    
    # Transform data through pipeline to get PCA features
    if hasattr(model, 'named_steps'):
        # Transform through all steps except classifier
        X_transformed = X.copy()
        for step_name in model.named_steps.keys():
            if step_name == 'classifier':
                break
            X_transformed = model.named_steps[step_name].transform(X_transformed)
        
        # Get feature names (PCA components)
        if 'pca' in model.named_steps:
            n_components = model.named_steps['pca'].n_components_
            feature_names = [f'PC_{i+1}' for i in range(n_components)]
        else:
            feature_names = [f'Feature_{i+1}' for i in range(X_transformed.shape[1])]
    else:
        feature_names = X.columns.tolist()
    
    # Get importance
    top_n = config.get('shap', {}).get('top_features', 20)
    importance_df = get_feature_importance(model, feature_names, top_n=top_n)
    
    if importance_df is not None:
        logger.info(f"Extracted top {len(importance_df)} features")
        
        # Save to CSV
        importance_file = reports_dir / 'tables' / f'feature_importance_{model_name}.csv'
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"Saved importance scores to {importance_file}")
        
        # Display top features
        logger.info("\nTop 10 Most Important Features:")
        logger.info(importance_df.head(10).to_string(index=False))
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Creating visualizations")
        logger.info("=" * 60)
        
        # Plot feature importance
        plot_path = figures_dir / 'shap' / f'feature_importance_{model_name}.png'
        plot_feature_importance(
            importance_df,
            plot_path,
            title=f"Top {len(importance_df)} Feature Importances - {model_name}"
        )
        
        # Additional plot: cumulative importance
        plt.figure(figsize=(10, 6))
        cumsum = importance_df['importance'].cumsum()
        plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title(f'Cumulative Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        cumsum_path = figures_dir / 'shap' / f'cumulative_importance_{model_name}.png'
        plt.savefig(cumsum_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved figure: {cumsum_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nResults saved to: {reports_dir / 'tables'}")
        logger.info(f"Figures saved to: {figures_dir / 'shap'}")
        
    else:
        logger.warning("Could not extract feature importance from model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate feature importance report'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    main(args.config)
