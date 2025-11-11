#!/usr/bin/env python3
"""
Nested Cross-Validation Training Script
========================================
This script trains and evaluates multiple models using nested cross-validation
for unbiased performance estimation.

Usage:
    python scripts/train_cv.py --config config.yaml

"""

import sys
import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

# Disable joblib memory mapping for this data size (18K+ features causes I/O bottleneck)
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import GeneExpressionDataLoader
from src.pipeline import (
    build_pipeline,
    nested_cross_validation,
    train_final_model,
    save_pipeline
)
from src.models import get_model_config, get_all_models
from src.eval import (
    calculate_metrics,
    calculate_metrics_with_ci,
    calibration_metrics,
    decision_curve_analysis
)
from src.plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    plot_decision_curve
)

# Configure enhanced logging with timestamps and real-time visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Also log to file with timestamp
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(fh)


def main(config_path: str):
    """
    Main training pipeline with nested cross-validation.
    
    Parameters:
        config_path: Path to configuration YAML file.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    models_dir = Path('models')
    figures_dir = Path(config['paths']['figures'])
    reports_dir = Path(config['paths']['reports'])
    
    for dir_path in [models_dir, figures_dir / 'modeling', 
                      figures_dir / 'calibration', reports_dir / 'tables']:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading processed data")
    logger.info("=" * 60)
    
    processed_dir = Path(config['paths']['processed'])
    
    expression_df = pd.read_csv(processed_dir / 'expression_processed.csv', index_col=0)
    metadata_df = pd.read_csv(processed_dir / 'metadata_processed.csv', index_col=0)
    
    X = expression_df
    y = metadata_df['label'].values
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Get CV configuration
    outer_cv = config['cv']['outer_folds']
    inner_cv = config['cv']['inner_folds']
    random_state = config['random_state']
    
    # Get models and feature selection methods
    feature_methods = config['features']['routes']
    models_dict = get_all_models(config)
    
    logger.info(f"\nFeature methods: {feature_methods}")
    logger.info(f"Models: {list(models_dict.keys())}")
    logger.info(f"Nested CV: outer={outer_cv}, inner={inner_cv}")
    
    # Storage for all results
    all_results = []
    all_predictions = {}
    
    # Step 2: Train and evaluate all combinations
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Nested Cross-Validation")
    logger.info("=" * 60)
    
    total_combinations = len(feature_methods) * len(models_dict)
    current = 0
    
    # Create progress bar for all combinations
    pbar = tqdm(
        total=total_combinations,
        desc="Training models",
        unit="model",
        position=1,
        leave=True
    )
    
    for feature_method in feature_methods:
        for model_name, (_, param_grid) in models_dict.items():
            current += 1
            
            pbar.set_description(f"[{current}/{total_combinations}] {feature_method} + {model_name}")
            logger.info(f"\n[{current}/{total_combinations}] {feature_method} + {model_name}")
            logger.info("-" * 60)
            
            # Build pipeline
            pipeline = build_pipeline(
                feature_method=feature_method,
                model_name=model_name,
                config=config,
                include_batch_correction=True
            )
            
            # Run nested CV
            cv_results = nested_cross_validation(
                X=X,
                y=y,
                pipeline=pipeline,
                param_grid=param_grid,
                outer_cv=outer_cv,
                inner_cv=inner_cv,
                random_state=random_state,
                scoring='roc_auc'
            )
            
            # Calculate comprehensive metrics
            y_true = cv_results['y_true_all']
            y_pred = cv_results['y_pred_all']
            y_prob = cv_results['y_prob_all']
            
            metrics = calculate_metrics(y_true, y_pred, y_prob)
            
            # Store results
            result_entry = {
                'feature_method': feature_method,
                'model': model_name,
                'pipeline_name': f"{feature_method}_{model_name}",
                **metrics,
                'cv_mean_train': cv_results['mean_train_score'],
                'cv_std_train': cv_results['std_train_score'],
                'cv_mean_val': cv_results['mean_val_score'],
                'cv_std_val': cv_results['std_val_score']
            }
            
            all_results.append(result_entry)
            
            # Store predictions
            pipeline_name = f"{feature_method}_{model_name}"
            all_predictions[pipeline_name] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            logger.info(f"Results:")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            logger.info(f"  F1 Score: {metrics['f1']:.3f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            
            # Update progress bar
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Step 3: Save results
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Saving results")
    logger.info("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    # Calculate mean ± SD across outer folds per pipeline
    # Group by pipeline_name and calculate statistics
    summary_stats = []
    
    for pipeline_name in results_df['pipeline_name'].unique():
        pipeline_results = results_df[results_df['pipeline_name'] == pipeline_name]
        
        stats = {
            'pipeline': pipeline_name,
            'n_folds': len(pipeline_results)
        }
        
        for metric in ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']:
            if metric in pipeline_results.columns:
                values = pipeline_results[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_mean'] = values.mean()
                    stats[f'{metric}_std'] = values.std()
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('roc_auc_mean', ascending=False)
    
    # Save summary with SD
    summary_file = reports_dir / 'tables' / 'summary_metrics.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary metrics to {summary_file}")
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Comparison (Nested CV)}")
    latex_lines.append("\\label{tab:model_performance}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Model} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{F1} & \\textbf{Accuracy} \\\\")
    latex_lines.append("\\hline")
    
    for _, row in summary_df.iterrows():
        pipeline = row['pipeline'].replace('_', '\\_')
        
        roc_mean = row.get('roc_auc_mean', 0)
        roc_std = row.get('roc_auc_std', 0)
        
        pr_mean = row.get('pr_auc_mean', 0)
        pr_std = row.get('pr_auc_std', 0)
        
        f1_mean = row.get('f1_mean', 0)
        f1_std = row.get('f1_std', 0)
        
        acc_mean = row.get('accuracy_mean', 0)
        acc_std = row.get('accuracy_std', 0)
        
        latex_lines.append(
            f"{pipeline} & "
            f"${roc_mean:.3f} \\pm {roc_std:.3f}$ & "
            f"${pr_mean:.3f} \\pm {pr_std:.3f}$ & "
            f"${f1_mean:.3f} \\pm {f1_std:.3f}$ & "
            f"${acc_mean:.3f} \\pm {acc_std:.3f}$ \\\\"
        )
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_file = reports_dir / 'tables' / 'summary_metrics.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Save main results table (per fold)
    results_file = reports_dir / 'tables' / 'nested_cv_results.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to {results_file}")
    
    # Display top models
    n_top = min(5, len(results_df))
    logger.info(f"\nTop {n_top} Models by ROC-AUC:")
    logger.info(results_df[['pipeline_name', 'roc_auc', 'f1', 'accuracy']].head(n_top).to_string(index=False))
    
    # Step 4: Create visualizations
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Creating visualizations")
    logger.info("=" * 60)
    
    # Prepare data for plotting (use top 5 models or fewer if not available)
    top_models = results_df.head(min(5, len(results_df)))['pipeline_name'].tolist()
    y_prob_dict = {name: all_predictions[name]['y_prob'] for name in top_models}
    y_true_plot = all_predictions[top_models[0]]['y_true']
    
    # ROC curves
    n_models = len(y_prob_dict)
    title = f'ROC Curves - Top {n_models} Models' if n_models > 1 else 'ROC Curves'
    plot_roc_curve(
        y_true_plot,
        y_prob_dict,
        title=title,
        save_path=figures_dir / 'modeling' / 'roc_curves.png'
    )
    logger.info("Created ROC curves")
    
    # PR curves
    n_models = len(y_prob_dict)
    title = f'Precision-Recall Curves - Top {n_models} Models' if n_models > 1 else 'Precision-Recall Curves'
    plot_precision_recall_curve(
        y_true_plot,
        y_prob_dict,
        title=title,
        save_path=figures_dir / 'modeling' / 'pr_curves.png'
    )
    logger.info("Created PR curves")
    
    # Confusion matrix for best model
    best_model_name = top_models[0]
    best_preds = all_predictions[best_model_name]
    
    plot_confusion_matrix(
        best_preds['y_true'],
        best_preds['y_pred'],
        labels=['Control', 'GBM'],
        title=f'Confusion Matrix - {best_model_name}',
        save_path=figures_dir / 'modeling' / 'confusion_matrix_best.png'
    )
    logger.info("Created confusion matrix")
    
    # Calibration curves
    n_models_calib = len(y_prob_dict)
    title_calib = f'Calibration Curves - Top {n_models_calib} Models' if n_models_calib > 1 else 'Calibration Curves'
    plot_calibration_curve(
        y_true_plot,
        y_prob_dict,
        n_bins=10,
        title=title_calib,
        save_path=figures_dir / 'calibration' / 'calibration_curves.png'
    )
    logger.info("Created calibration curves")
    
    # Decision curve for best model
    dca_df = decision_curve_analysis(
        best_preds['y_true'],
        best_preds['y_prob']
    )
    
    plot_decision_curve(
        dca_df,
        title=f'Decision Curve Analysis - {best_model_name}',
        save_path=figures_dir / 'calibration' / 'decision_curve_best.png'
    )
    logger.info("Created decision curve")
    
    # Step 5: Calculate metrics with confidence intervals
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Bootstrap confidence intervals")
    logger.info("=" * 60)
    
    n_bootstrap = config['evaluation'].get('bootstrap_ci', {}).get('n_bootstrap', 1000)
    
    metrics_ci_df = calculate_metrics_with_ci(
        best_preds['y_true'],
        best_preds['y_pred'],
        best_preds['y_prob'],
        n_bootstrap=n_bootstrap
    )
    
    # Save metrics with CI
    metrics_ci_file = reports_dir / 'tables' / f'metrics_ci_{best_model_name}.csv'
    metrics_ci_df.to_csv(metrics_ci_file, index=False)
    logger.info(f"Saved metrics with CI to {metrics_ci_file}")
    
    logger.info(f"\nMetrics for {best_model_name}:")
    logger.info(f"\n{metrics_ci_df.to_string(index=False)}")
    
    # Step 6: Train final model on all data
    if config['output'].get('save_models', True):
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Training final model on all data")
        logger.info("=" * 60)
        
        # Use best performing configuration
        best_row = results_df.iloc[0]
        best_feature = best_row['feature_method']
        best_model = best_row['model']
        
        logger.info(f"Best configuration: {best_feature} + {best_model}")
        
        # Build pipeline
        final_pipeline = build_pipeline(
            feature_method=best_feature,
            model_name=best_model,
            config=config,
            include_batch_correction=True
        )
        
        # Get param grid
        _, param_grid = get_model_config(best_model, config)
        
        # Train on all data
        fitted_pipeline, train_results = train_final_model(
            X=X,
            y=y,
            pipeline=final_pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            random_state=random_state
        )
        
        # Save model
        model_metadata = {
            'feature_method': best_feature,
            'model_name': best_model,
            'training_date': datetime.now().isoformat(),
            'config': config,
            'cv_score': train_results['best_score'],
            'best_params': train_results['best_params'],
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        model_file = models_dir / f'final_model_{best_feature}_{best_model}.pkl'
        save_pipeline(fitted_pipeline, str(model_file), metadata=model_metadata)
        
        logger.info(f"Saved final model to {model_file}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.3f}")
    logger.info(f"F1 Score: {results_df.iloc[0]['f1']:.3f}")
    logger.info(f"\nResults saved to: {reports_dir / 'tables'}/")
    logger.info(f"Figures saved to: {figures_dir}/")
    if config['output'].get('save_models', True):
        logger.info(f"Model saved to: {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train GBM classification models with nested cross-validation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    main(args.config)
