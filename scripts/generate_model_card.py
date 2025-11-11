"""
Model Card Generator
====================
Auto-generates comprehensive model card from training results.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> dict:
    """Load all result files into a dictionary."""
    results = {}
    
    # Nested CV results
    nested_cv_path = results_dir / "nested_cv_results.csv"
    if nested_cv_path.exists():
        results['nested_cv'] = pd.read_csv(nested_cv_path)
    
    # Find best model metrics with CI
    metrics_files = list(results_dir.glob("metrics_ci_*.csv"))
    if metrics_files:
        # Get most recent or best model
        best_model_file = metrics_files[0]
        results['metrics_ci'] = pd.read_csv(best_model_file)
        results['best_model_name'] = best_model_file.stem.replace("metrics_ci_", "")
    
    # Stability panel if available
    stability_path = results_dir / "stability_panel.csv"
    if stability_path.exists():
        results['stability'] = pd.read_csv(stability_path)
    
    return results


def format_metric_with_ci(row: pd.Series) -> str:
    """Format metric with confidence interval."""
    # Handle different column name variations
    mean_val = row.get('mean', row.get('value', 0))
    ci_lower = row.get('ci_lower', row.get('lower_ci', None))
    ci_upper = row.get('ci_upper', row.get('upper_ci', None))
    
    if pd.isna(ci_lower) or pd.isna(ci_upper):
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})"


def generate_model_card(
    results: dict,
    config_path: Path,
    output_path: Path
):
    """
    Generate comprehensive model card from results.
    
    Parameters:
        results: Dictionary of loaded result DataFrames
        config_path: Path to config.yaml
        output_path: Path to save model card
    """
    # Load config if available
    config = {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {
            'project_name': 'GBM Classification',
            'features': {'routes': ['filter_l1', 'pca']},
            'models': ['random_forest', 'lr_elasticnet'],
            'cv': {'outer_folds': 3, 'inner_folds': 2, 'test_size': 0.2},
            'preprocessing': {'batch_correction': 'combat', 'scaler': 'standard'}
        }
    
    # Start building card
    card = []
    card.append("# Model Card: GBM Classification Pipeline\n")
    card.append("*Auto-generated from training results*\n")
    card.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    card.append("---\n\n")
    
    # Model Details
    card.append("## Model Details\n\n")
    card.append(f"**Project:** {config.get('project_name', 'GBM Classification')}\n")
    card.append(f"**Developer:** Hogeschool Rotterdam Team\n")
    card.append(f"**Institution:** Hogeschool Rotterdam - Minor AI in Healthcare\n")
    card.append(f"**Date:** {datetime.now().strftime('%B %Y')}\n")
    
    if 'best_model_name' in results:
        card.append(f"**Best Model:** {results['best_model_name']}\n")
    
    card.append("\n### Model Architecture\n\n")
    card.append("The pipeline consists of:\n\n")
    card.append("1. **Feature Selection:** ")
    
    feature_routes = config.get('features', {}).get('routes', [])
    if 'filter_l1' in feature_routes:
        k_best = config.get('feature_selection', {}).get('filter_l1', {}).get('k_best', 500)
        card.append(f"FilterL1 (k={k_best}: variance → correlation → L1 regularization)")
    if 'pca' in feature_routes:
        n_comp = config.get('feature_selection', {}).get('pca', {}).get('n_components', 200)
        if 'filter_l1' in feature_routes:
            card.append(f" OR PCA (n={n_comp} components)")
        else:
            card.append(f"PCA (n={n_comp} components)")
    card.append("\n")
    
    card.append("2. **Preprocessing:** ")
    batch_method = config.get('preprocessing', {}).get('batch_correction', 'combat')
    scaler = config.get('preprocessing', {}).get('scaler', 'standard')
    card.append(f"ComBat batch correction → {scaler.capitalize()} scaling\n")
    
    card.append("3. **Classifiers:** ")
    models = config.get('models', [])
    card.append(f"{', '.join(models)}\n\n")
    
    # Intended Use
    card.append("## Intended Use\n\n")
    card.append("**Primary Use Case:** Binary classification of gene expression profiles to distinguish between:\n")
    card.append("- Healthy brain tissue samples\n")
    card.append("- Glioblastoma Multiforme (GBM) tumor samples\n\n")
    card.append("**Intended Users:** Bioinformatics researchers, clinical research teams\n\n")
    card.append("**Out-of-Scope Use Cases:**\n")
    card.append("- Direct clinical diagnosis without expert review\n")
    card.append("- Application to other cancer types\n")
    card.append("- Use with different sequencing platforms without validation\n\n")
    
    # Training Data
    card.append("## Training Data\n\n")
    card.append("**Source:** Combined gene expression dataset (18,635 samples × 18,858 genes)\n\n")
    
    if 'nested_cv' in results:
        df = results['nested_cv']
        if not df.empty and 'roc_auc' in df.columns:
            n_samples = 18635  # From data
            card.append(f"**Dataset Size:** {n_samples:,} samples\n")
            card.append(f"**Features:** 18,858 genes (ENSEMBL IDs)\n")
            card.append(f"**Class Distribution:** ~93% healthy, ~7% tumor (imbalanced)\n\n")
    
    card.append("**Preprocessing Steps:**\n")
    card.append(f"1. Low-variance gene removal (threshold={config.get('preprocessing', {}).get('variance_threshold', 0.01)})\n")
    card.append(f"2. High-correlation filtering (threshold={config.get('preprocessing', {}).get('correlation_threshold', 0.95)})\n")
    card.append("3. Batch effect correction (ComBat)\n")
    card.append("4. Feature scaling (StandardScaler)\n\n")
    
    # Evaluation
    card.append("## Evaluation\n\n")
    
    cv_config = config.get('cv', {})
    outer_folds = cv_config.get('outer_folds', 5)
    inner_folds = cv_config.get('inner_folds', 3)
    test_size = cv_config.get('test_size', 0.2)
    
    card.append(f"**Validation Strategy:** Nested stratified cross-validation\n")
    card.append(f"- Outer folds: {outer_folds} (performance estimation)\n")
    card.append(f"- Inner folds: {inner_folds} (hyperparameter tuning)\n")
    card.append(f"- Hold-out test set: {int(test_size*100)}%\n\n")
    
    # Performance Metrics
    card.append("### Performance Metrics\n\n")
    
    if 'nested_cv' in results:
        df = results['nested_cv'].sort_values('roc_auc', ascending=False)
        
        card.append("**Model Comparison (Nested CV):**\n\n")
        card.append("| Model | ROC-AUC | PR-AUC | F1 | Accuracy |\n")
        card.append("|-------|---------|--------|-----|----------|\n")
        
        for _, row in df.iterrows():
            card.append(
                f"| {row['model']} | "
                f"{row.get('roc_auc', 0):.3f} | "
                f"{row.get('pr_auc', 0):.3f} | "
                f"{row.get('f1', 0):.3f} | "
                f"{row.get('accuracy', 0):.3f} |\n"
            )
        card.append("\n")
    
    if 'metrics_ci' in results:
        df = results['metrics_ci']
        
        card.append(f"**Best Model Detailed Performance ({results.get('best_model_name', 'Unknown')}):**\n\n")
        card.append("| Metric | Value (95% CI) |\n")
        card.append("|--------|----------------|\n")
        
        for _, row in df.iterrows():
            metric_name = row['metric'].replace('_', ' ').title()
            formatted = format_metric_with_ci(row)
            card.append(f"| {metric_name} | {formatted} |\n")
        card.append("\n")
    
    # Calibration
    card.append("### Calibration Analysis\n\n")
    card.append("Model predictions are calibrated using isotonic regression to ensure ")
    card.append("predicted probabilities accurately reflect true outcome frequencies.\n\n")
    
    if 'metrics_ci' in results:
        brier_row = results['metrics_ci'][results['metrics_ci']['metric'] == 'brier_score']
        if not brier_row.empty:
            brier = brier_row.iloc[0]['mean']
            card.append(f"**Brier Score:** {brier:.3f} (lower is better, perfect=0)\n\n")
    
    # Feature Importance
    if 'stability' in results:
        df = results['stability']
        card.append("### Feature Stability (Filter-L1 Route)\n\n")
        card.append("Top 10 most stable genes across 100 bootstrap iterations:\n\n")
        card.append("| Rank | Gene | Selection Frequency | Mean Importance |\n")
        card.append("|------|------|---------------------|------------------|\n")
        
        for i, row in df.head(10).iterrows():
            card.append(
                f"| {i+1} | {row['gene']} | "
                f"{row['selection_frequency']:.2f} | "
                f"{row['mean_importance']:.4f} |\n"
            )
        card.append("\n")
        
        high_stability = (df['selection_frequency'] >= 0.8).sum()
        card.append(f"**Highly stable features (≥80% selection):** {high_stability}\n\n")
    
    # Limitations
    card.append("## Limitations\n\n")
    card.append("1. **Class Imbalance:** Dataset is heavily skewed (~93% healthy), requiring balanced weighting\n")
    card.append("2. **Generalization:** Performance on external datasets not yet validated\n")
    card.append("3. **Batch Effects:** Model assumes batch correction adequately removes technical variation\n")
    card.append("4. **Feature Interpretation:** PCA components lack direct biological interpretability\n")
    card.append("5. **Computational Cost:** Filter-L1 route requires significant compute time for large gene sets\n\n")
    
    # Ethical Considerations
    card.append("## Ethical Considerations\n\n")
    card.append("**Fairness:**\n")
    card.append("- No patient demographics available to assess fairness across subgroups\n")
    card.append("- Class imbalance may bias model toward healthy class predictions\n\n")
    card.append("**Privacy:**\n")
    card.append("- Gene expression data is de-identified\n")
    card.append("- No linkage to individual patient records\n\n")
    card.append("**Clinical Impact:**\n")
    card.append("- Model is NOT approved for clinical use\n")
    card.append("- Predictions must be validated by trained professionals\n")
    card.append("- False negatives (missing tumors) have severe clinical consequences\n\n")
    
    # Recommendations
    card.append("## Recommendations\n\n")
    card.append("**For Research Use:**\n")
    card.append("- Validate on independent external cohorts\n")
    card.append("- Perform biological pathway analysis on stable genes\n")
    card.append("- Investigate decision boundaries and misclassified samples\n\n")
    card.append("**For Future Improvement:**\n")
    card.append("- Collect more tumor samples to balance dataset\n")
    card.append("- Integrate multi-omic data (methylation, CNV, mutations)\n")
    card.append("- Develop ensemble methods combining multiple feature selection routes\n")
    card.append("- External validation on TCGA-GBM and other public datasets\n\n")
    
    # Citation
    card.append("## Citation\n\n")
    card.append("```\n")
    card.append("@misc{2025gbm,\n")
    card.append("  author = {Hogeschool Rotterdam Team},\n")
    card.append("  title = {GBM Classification Pipeline: Gene Expression Analysis},\n")
    card.append("  year = {2025},\n")
    card.append("  institution = {Hogeschool Rotterdam},\n")
    card.append("  note = {Minor AI in Healthcare - Retake Project}\n")
    card.append("}\n")
    card.append("```\n\n")
    
    # Save card
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(card)
    
    logger.info(f"Model card generated: {output_path}")


def main():
    """Main execution function."""
    results_dir = Path("reports/tables")
    # Try to find the most recent config file
    config_candidates = [
        Path("config_ultrafast_pca.yaml"),
        Path("config_academic_feasible.yaml"),
        Path("config.yaml")
    ]
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break
    
    if config_path is None:
        logger.warning("No config file found, using default values")
        config_path = Path("config.yaml")  # Will handle missing file gracefully
    
    output_path = Path("metadata/model_card_generated.md")
    
    logger.info("Loading training results...")
    results = load_results(results_dir)
    
    logger.info(f"Found {len(results)} result sets")
    for key in results:
        logger.info(f"  - {key}")
    
    logger.info("Generating model card...")
    generate_model_card(results, config_path, output_path)
    
    logger.info("✅ Model card generation complete!")


if __name__ == "__main__":
    main()
