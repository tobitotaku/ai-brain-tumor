#!/usr/bin/env python3
"""
SHAP Explainability Report Script
==================================
This script generates SHAP (SHapley Additive exPlanations) visualizations
and reports for model explainability.

Usage:
    python scripts/shap_report.py --config config.yaml --model models/final_model.pkl

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
import shap

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import load_pipeline
from src.plots import save_figure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str, model_path: str = None):
    """
    Generate SHAP explainability report.
    
    Parameters:
        config_path: Path to configuration YAML file.
        model_path: Path to trained model file. If None, finds latest model.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    shap_dir = Path(config['paths']['figures']) / 'shap'
    reports_dir = Path(config['paths']['reports']) / 'tables'
    shap_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model if not specified
    if model_path is None:
        models_dir = Path('models')
        model_files = list(models_dir.glob('final_model_*.pkl'))
        
        if not model_files:
            logger.error("No trained models found in models/")
            logger.info("Please run train_cv.py first or specify --model path")
            return
        
        model_path = sorted(model_files)[-1]  # Most recent
        logger.info(f"Using model: {model_path}")
    
    # Step 1: Load model and data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading model and data")
    logger.info("=" * 60)
    
    pipeline, metadata = load_pipeline(model_path)
    logger.info(f"Model: {metadata.get('model_name', 'unknown')}")
    logger.info(f"Feature method: {metadata.get('feature_method', 'unknown')}")
    
    # Load processed data
    processed_dir = Path(config['paths']['processed'])
    expression_df = pd.read_csv(processed_dir / 'expression_processed.csv', index_col=0)
    metadata_df = pd.read_csv(processed_dir / 'metadata_processed.csv', index_col=0)
    
    X = expression_df
    y = metadata_df['label'].values
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes")
    
    # Step 2: Prepare data for SHAP
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preparing SHAP analysis")
    logger.info("=" * 60)
    
    # Transform data through pipeline (except classifier)
    # This gives us the transformed features that go into the classifier
    X_transformed = X.copy()
    
    for step_name, transformer in pipeline.steps[:-1]:  # All except classifier
        logger.info(f"Applying: {step_name}")
        X_transformed = transformer.transform(X_transformed)
    
    logger.info(f"Transformed data shape: {X_transformed.shape}")
    
    # Get feature names
    if hasattr(X_transformed, 'columns'):
        feature_names = X_transformed.columns.tolist()
        X_transformed = X_transformed.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
    
    # Get classifier
    classifier = pipeline.steps[-1][1]
    
    # Step 3: Compute SHAP values
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Computing SHAP values")
    logger.info("=" * 60)
    
    n_background = config.get('shap', {}).get('n_samples', 100)
    n_background = min(n_background, len(X_transformed))
    
    # Sample background data
    np.random.seed(config['random_state'])
    background_indices = np.random.choice(len(X_transformed), n_background, replace=False)
    background_data = X_transformed[background_indices]
    
    logger.info(f"Using {n_background} samples for SHAP background")
    
    # Choose appropriate explainer based on model type
    model_name = metadata.get('model_name', '')
    
    if 'lr' in model_name or 'logistic' in model_name.lower():
        # Linear models: use LinearExplainer
        logger.info("Using LinearExplainer for linear model")
        explainer = shap.LinearExplainer(classifier, background_data)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Tree-based models: use TreeExplainer
        # For other models: use KernelExplainer (slower)
        try:
            logger.info("Attempting TreeExplainer...")
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            
            # For binary classification, TreeExplainer returns values for both classes
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
        except:
            logger.info("TreeExplainer failed, using KernelExplainer...")
            logger.warning("This may take a while...")
            
            # Create prediction function
            def model_predict(X):
                return classifier.predict_proba(X)[:, 1]
            
            explainer = shap.KernelExplainer(model_predict, background_data)
            shap_values = explainer.shap_values(X_transformed[:200])  # Limit for speed
            X_transformed = X_transformed[:200]
    
    logger.info(f"SHAP values computed: shape {shap_values.shape}")
    
    # Step 4: Create SHAP visualizations
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating SHAP visualizations")
    logger.info("=" * 60)
    
    # Set SHAP plotting style
    shap.initjs()
    
    # 1. Summary Plot (Beeswarm)
    logger.info("Creating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(shap_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {shap_dir / 'shap_beeswarm.png'}")
    
    # 2. Bar Plot (Mean absolute SHAP values)
    logger.info("Creating bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=20
    )
    plt.tight_layout()
    plt.savefig(shap_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {shap_dir / 'shap_bar.png'}")
    
    # 3. Waterfall plot (single sample)
    logger.info("Creating waterfall plot for sample...")
    # Select a GBM sample (label = 1)
    gbm_indices = np.where(y == 1)[0]
    if len(gbm_indices) > 0:
        sample_idx = gbm_indices[0]
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                data=X_transformed[sample_idx],
                feature_names=feature_names
            ),
            show=False,
            max_display=15
        )
        plt.tight_layout()
        plt.savefig(shap_dir / 'shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {shap_dir / 'shap_waterfall_sample.png'}")
    
    # 4. Dependence plots for top features
    logger.info("Creating dependence plots...")
    
    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    for i, feat_idx in enumerate(top_feature_indices):
        feature_name = feature_names[feat_idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            shap_dir / f'shap_dependence_{i+1}_{feature_name.replace("/", "_")}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
    
    logger.info(f"Created {len(top_feature_indices)} dependence plots")
    
    # Step 5: Create SHAP summary table
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Creating SHAP summary table")
    logger.info("=" * 60)
    
    # Calculate feature importance metrics
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    
    shap_summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'std_shap': std_shap,
        'rank': np.arange(1, len(feature_names) + 1)
    })
    
    # Sort by importance
    shap_summary = shap_summary.sort_values('mean_abs_shap', ascending=False)
    shap_summary['rank'] = np.arange(1, len(shap_summary) + 1)
    
    # Save summary
    shap_summary_file = reports_dir / 'shap_summary.csv'
    shap_summary.to_csv(shap_summary_file, index=False)
    logger.info(f"Saved SHAP summary to {shap_summary_file}")
    
    # Display top 10 features
    logger.info("\nTop 10 Most Important Features (by mean |SHAP|):")
    logger.info(f"\n{shap_summary.head(10).to_string(index=False)}")
    
    # Step 6: Save SHAP values for future use
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving SHAP values")
    logger.info("=" * 60)
    
    # Save as numpy arrays
    shap_output = {
        'shap_values': shap_values,
        'feature_names': feature_names,
        'data': X_transformed,
        'y_true': y[:len(X_transformed)] if len(y) > len(X_transformed) else y
    }
    
    shap_file = shap_dir / 'shap_values.npz'
    np.savez(shap_file, **shap_output)
    logger.info(f"Saved SHAP values to {shap_file}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nSHAP plots saved to: {shap_dir}/")
    logger.info(f"SHAP summary saved to: {shap_summary_file}")
    logger.info(f"SHAP values saved to: {shap_file}")
    logger.info(f"\nTop feature: {shap_summary.iloc[0]['feature']}")
    logger.info(f"Mean |SHAP|: {shap_summary.iloc[0]['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate SHAP explainability report for GBM classification model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model file (default: latest in models/)'
    )
    
    args = parser.parse_args()
    
    main(args.config, args.model)
