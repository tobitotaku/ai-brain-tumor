"""
SHAP Analysis on Compact Gene Panel
====================================
Performs SHAP analysis on top-N most stable genes from filter_l1 selection.

NOTE: This script requires SHAP, which is incompatible with Python 3.14.
As a workaround, we use sklearn's feature_importances_ for tree models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import logging
from typing import List

from src.data import GeneExpressionDataLoader
from src.pipeline import build_pipeline, train_final_model
from src.eval import calculate_metrics, bootstrap_confidence_intervals
from src.plots import plot_feature_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_stability_panel(stability_path: Path, top_n: int = 30) -> List[str]:
    """
    Load top-N most stable genes from stability analysis.
    
    Parameters:
        stability_path: Path to stability_panel.csv
        top_n: Number of top genes to select
        
    Returns:
        List of gene names
    """
    if not stability_path.exists():
        raise FileNotFoundError(
            f"Stability panel not found at {stability_path}. "
            "Run stability_analysis.py first."
        )
    
    df = pd.read_csv(stability_path)
    top_genes = df.head(top_n)['gene'].tolist()
    
    logger.info(f"Loaded top {top_n} genes from stability panel")
    logger.info(f"Selection frequency range: {df.iloc[top_n-1]['selection_frequency']:.2f} - {df.iloc[0]['selection_frequency']:.2f}")
    
    return top_genes


def train_on_compact_panel(
    X: pd.DataFrame,
    y: np.ndarray,
    gene_panel: List[str],
    config: dict
) -> tuple:
    """
    Train best model on compact gene panel.
    
    Parameters:
        X: Full feature matrix
        y: Target labels
        gene_panel: List of genes to use
        config: Configuration dict
        
    Returns:
        Tuple of (fitted_pipeline, predictions_dict)
    """
    # Subset to panel genes only
    missing_genes = [g for g in gene_panel if g not in X.columns]
    if missing_genes:
        logger.warning(f"{len(missing_genes)} genes not found in data: {missing_genes[:5]}")
        gene_panel = [g for g in gene_panel if g in X.columns]
    
    X_panel = X[gene_panel]
    logger.info(f"Training on compact panel: {X_panel.shape[1]} genes")
    
    # Build pipeline (without feature selection, directly to classifier)
    # Use filter_l1 route config but with pre-selected genes
    from src.preprocess import BatchCorrector, ExpressionScaler
    from src.models import get_model_config
    from sklearn.pipeline import Pipeline
    
    batch_corrector = BatchCorrector(method='combat')
    scaler = ExpressionScaler(method='standard')
    
    # Use Random Forest (typically best performing)
    model, param_grid = get_model_config('random_forest', config)
    
    pipeline = Pipeline([
        ('batch_correction', batch_corrector),
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    logger.info("Training final model on full data (compact panel)...")
    
    # Train with hyperparameter tuning
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    
    cv = StratifiedKFold(
        n_splits=config['cv']['inner_folds'],
        shuffle=True,
        random_state=config['random_state']
    )
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_panel, y)
    
    logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
    logger.info(f"Best params: {grid_search.best_params_}")
    
    # Get predictions on full data
    y_pred = grid_search.predict(X_panel)
    y_prob = grid_search.predict_proba(X_panel)[:, 1]
    
    predictions = {
        'y_true': y,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return grid_search.best_estimator_, predictions


def extract_compact_feature_importance(
    pipeline,
    gene_panel: List[str],
    save_dir: Path
):
    """
    Extract feature importance from compact panel model.
    
    Parameters:
        pipeline: Fitted sklearn pipeline
        gene_panel: List of gene names
        save_dir: Directory to save results
    """
    # Get classifier from pipeline
    classifier = pipeline.named_steps['classifier']
    
    # Extract importance
    if hasattr(classifier, 'feature_importances_'):
        importance = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importance = np.abs(classifier.coef_[0])
    else:
        logger.warning("Classifier has no feature_importances_ or coef_")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'gene': gene_panel,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Save CSV
    csv_path = save_dir / 'feature_importance_compact_panel.csv'
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Saved compact panel importance to {csv_path}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 12))
    
    top_features = importance_df.head(30)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['gene'], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature Importance: Compact Gene Panel (Top-30 Stable Genes)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = save_dir / 'feature_importance_compact_panel.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved compact panel importance plot to {fig_path}")


def main():
    """Main execution function."""
    # Load config
    config_path = Path("config/config_ultrafast_pca.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Paths
    stability_path = Path("reports/tables/stability_panel.csv")
    output_dir = Path("reports/tables")
    fig_dir = Path("figures/shap")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load stability panel
    logger.info("=" * 60)
    logger.info("STEP 1: Load compact gene panel")
    logger.info("=" * 60)
    
    top_n = 30
    gene_panel = load_stability_panel(stability_path, top_n=top_n)
    
    logger.info(f"Top 10 genes: {gene_panel[:10]}")
    
    # Step 2: Load data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Load processed data")
    logger.info("=" * 60)
    
    loader = GeneExpressionDataLoader()
    X, y, metadata = loader.load_data(
        expression_path="data/processed/expression_processed.csv",
        metadata_path="data/processed/metadata_processed.csv"
    )
    
    logger.info(f"Loaded: {X.shape[0]} samples × {X.shape[1]} genes")
    
    # Step 3: Train on compact panel
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Train model on compact panel")
    logger.info("=" * 60)
    
    pipeline, predictions = train_on_compact_panel(X, y, gene_panel, config)
    
    # Step 4: Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Evaluate compact panel model")
    logger.info("=" * 60)
    
    metrics = calculate_metrics(
        predictions['y_true'],
        predictions['y_pred'],
        predictions['y_prob']
    )
    
    logger.info("\nPerformance on Compact Panel ({} genes):".format(len(gene_panel)))
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Bootstrap CI
    ci_df = bootstrap_confidence_intervals(
        predictions['y_true'],
        predictions['y_prob'],
        n_bootstrap=1000
    )
    
    ci_path = output_dir / 'metrics_ci_compact_panel.csv'
    ci_df.to_csv(ci_path, index=False)
    logger.info(f"\nSaved metrics with CI to {ci_path}")
    
    # Step 5: Extract feature importance
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Extract feature importance")
    logger.info("=" * 60)
    
    extract_compact_feature_importance(pipeline, gene_panel, fig_dir)
    
    # Step 6: Save model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Save compact panel model")
    logger.info("=" * 60)
    
    model_path = Path("models/final_model_compact_panel.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'pipeline': pipeline, 'gene_panel': gene_panel}, f)
    
    logger.info(f"Saved model to {model_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ COMPACT PANEL ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
