"""
Pipeline Orchestration Module
==============================
This module implements the complete machine learning pipeline with
proper cross-validation and no data leakage.

Author: Musab 0988932
Date: November 2025
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
from pathlib import Path
from tqdm import tqdm

from .preprocess import BatchCorrector, ExpressionScaler
from .features import create_feature_selector
from .models import get_model_config

logger = logging.getLogger(__name__)


def build_pipeline(
    feature_method: str,
    model_name: str,
    config: dict,
    include_batch_correction: bool = True
) -> Pipeline:
    """
    Build a complete sklearn pipeline.
    
    Pipeline steps:
    1. Feature selection (filter_l1, pca, or bio_panel)
    2. Batch correction (optional)
    3. Scaling
    4. Classifier
    
    Parameters:
        feature_method: Feature selection method.
        model_name: Model name.
        config: Configuration dictionary.
        include_batch_correction: Whether to include batch correction.
        
    Returns:
        Sklearn Pipeline object.
    """
    steps = []
    
    # Step 1: Feature selection
    feature_selector = create_feature_selector(feature_method, config)
    steps.append(('feature_selector', feature_selector))
    
    # Step 2: Batch correction (optional)
    if include_batch_correction:
        batch_method = config.get('preprocessing', {}).get('batch_correction', 'combat')
        batch_corrector = BatchCorrector(method=batch_method)
        steps.append(('batch_correction', batch_corrector))
    
    # Step 3: Scaling
    scale_method = config.get('preprocessing', {}).get('scaler', 'standard')
    scaler = ExpressionScaler(method=scale_method)
    steps.append(('scaler', scaler))
    
    # Step 4: Classifier
    model, _ = get_model_config(model_name, config)
    steps.append(('classifier', model))
    
    pipeline = Pipeline(steps)
    
    logger.info(
        f"Built pipeline: {feature_method} -> "
        f"{'batch_corr ->' if include_batch_correction else ''} "
        f"scaler -> {model_name}"
    )
    
    return pipeline


def nested_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    pipeline: Pipeline,
    param_grid: Dict[str, List],
    outer_cv: int = 5,
    inner_cv: int = 5,
    random_state: int = 42,
    scoring: str = 'roc_auc'
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for unbiased performance estimation.
    
    Outer loop: Estimates model performance
    Inner loop: Hyperparameter tuning
    
    Parameters:
        X: Feature matrix.
        y: Target labels.
        pipeline: Sklearn pipeline.
        param_grid: Parameter grid for hyperparameter tuning.
        outer_cv: Number of outer CV folds.
        inner_cv: Number of inner CV folds.
        random_state: Random seed.
        scoring: Scoring metric for optimization.
        
    Returns:
        Dictionary with CV results.
    """
    logger.info(
        f"Starting nested CV: outer={outer_cv}, inner={inner_cv}, "
        f"scoring={scoring}"
    )
    
    # Outer CV for performance estimation
    outer_cv_splitter = StratifiedKFold(
        n_splits=outer_cv,
        shuffle=True,
        random_state=random_state
    )
    
    # Inner CV for hyperparameter tuning
    inner_cv_splitter = StratifiedKFold(
        n_splits=inner_cv,
        shuffle=True,
        random_state=random_state
    )
    
    outer_results = {
        'fold': [],
        'train_score': [],
        'val_score': [],
        'best_params': [],
        'y_true': [],
        'y_pred': [],
        'y_prob': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(
            list(outer_cv_splitter.split(X, y)),
            desc="Outer CV",
            unit="fold",
            position=0,
            leave=True
        )
    ):
        logger.info(f"Outer fold {fold_idx + 1}/{outer_cv}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Inner CV: Hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,  # Show grid search progress
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model from inner CV
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation fold
        y_pred = best_model.predict(X_val)
        y_prob = best_model.predict_proba(X_val)[:, 1]
        
        # Store results
        outer_results['fold'].append(fold_idx + 1)
        outer_results['train_score'].append(grid_search.best_score_)
        outer_results['val_score'].append(best_model.score(X_val, y_val))
        outer_results['best_params'].append(grid_search.best_params_)
        outer_results['y_true'].append(y_val)
        outer_results['y_pred'].append(y_pred)
        outer_results['y_prob'].append(y_prob)
        
        logger.info(
            f"  Best params: {grid_search.best_params_}"
        )
        logger.info(
            f"  Train score: {grid_search.best_score_:.3f}, "
            f"Val score: {outer_results['val_score'][-1]:.3f}"
        )
    
    # Concatenate predictions from all folds
    outer_results['y_true_all'] = np.concatenate(outer_results['y_true'])
    outer_results['y_pred_all'] = np.concatenate(outer_results['y_pred'])
    outer_results['y_prob_all'] = np.concatenate(outer_results['y_prob'])
    
    # Summary statistics
    outer_results['mean_train_score'] = np.mean(outer_results['train_score'])
    outer_results['std_train_score'] = np.std(outer_results['train_score'])
    outer_results['mean_val_score'] = np.mean(outer_results['val_score'])
    outer_results['std_val_score'] = np.std(outer_results['val_score'])
    
    logger.info(
        f"Nested CV complete: "
        f"Val score = {outer_results['mean_val_score']:.3f} "
        f"± {outer_results['std_val_score']:.3f}"
    )
    
    return outer_results


def train_final_model(
    X: pd.DataFrame,
    y: np.ndarray,
    pipeline: Pipeline,
    param_grid: Dict[str, List],
    cv: int = 5,
    random_state: int = 42,
    scoring: str = 'roc_auc'
) -> Tuple[Pipeline, Dict]:
    """
    Train final model on all data with hyperparameter tuning.
    
    This should only be done after nested CV for performance estimation.
    
    Parameters:
        X: Feature matrix (all data).
        y: Target labels.
        pipeline: Sklearn pipeline.
        param_grid: Parameter grid.
        cv: Number of CV folds for tuning.
        random_state: Random seed.
        scoring: Scoring metric.
        
    Returns:
        Tuple of (fitted_pipeline, results_dict).
    """
    logger.info("Training final model on all data")
    
    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best CV score: {results['best_score']:.3f}")
    
    return grid_search.best_estimator_, results


def save_pipeline(
    pipeline: Pipeline,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save fitted pipeline to disk.
    
    Parameters:
        pipeline: Fitted pipeline.
        filepath: Path to save file.
        metadata: Optional metadata to save with pipeline.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    save_obj = {
        'pipeline': pipeline,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_obj, f)
    
    logger.info(f"Saved pipeline to {filepath}")


def load_pipeline(filepath: str) -> Tuple[Pipeline, Dict]:
    """
    Load fitted pipeline from disk.
    
    Parameters:
        filepath: Path to saved pipeline file.
        
    Returns:
        Tuple of (pipeline, metadata).
    """
    with open(filepath, 'rb') as f:
        save_obj = pickle.load(f)
    
    logger.info(f"Loaded pipeline from {filepath}")
    
    return save_obj['pipeline'], save_obj.get('metadata', {})


def run_ablation_study(
    X: pd.DataFrame,
    y: np.ndarray,
    base_config: dict,
    ablation_tests: List[Dict],
    outer_cv: int = 5,
    inner_cv: int = 5
) -> pd.DataFrame:
    """
    Run ablation study to assess component importance.
    
    Parameters:
        X: Feature matrix.
        y: Target labels.
        base_config: Base configuration.
        ablation_tests: List of configuration modifications to test.
        outer_cv: Outer CV folds.
        inner_cv: Inner CV folds.
        
    Returns:
        DataFrame with ablation study results.
    """
    logger.info(f"Starting ablation study with {len(ablation_tests)} tests")
    
    results = []
    
    for test_config in ablation_tests:
        test_name = test_config.get('name', 'unnamed_test')
        logger.info(f"\nAblation test: {test_name}")
        
        # Merge with base config
        config = base_config.copy()
        config.update(test_config)
        
        # Get feature method and model
        feature_method = config.get('features', {}).get('routes', ['filter_l1'])[0]
        model_name = list(config.get('models', {}).keys())[0]
        
        # Build pipeline
        include_batch = config.get('preprocessing', {}).get('batch_correction') is not None
        pipeline = build_pipeline(feature_method, model_name, config, include_batch)
        
        # Get param grid
        _, param_grid = get_model_config(model_name, config)
        
        # Run nested CV
        cv_results = nested_cross_validation(
            X, y, pipeline, param_grid,
            outer_cv=outer_cv,
            inner_cv=inner_cv
        )
        
        results.append({
            'test_name': test_name,
            'mean_score': cv_results['mean_val_score'],
            'std_score': cv_results['std_val_score'],
            'config': test_config
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_score', ascending=False)
    
    logger.info("\nAblation study complete:")
    logger.info(f"\n{results_df[['test_name', 'mean_score', 'std_score']]}")
    
    return results_df


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import yaml
    
    # Load example config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate example data
    X, y = make_classification(
        n_samples=200,
        n_features=1000,
        n_informative=50,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(X.shape[1])])
    
    print("Testing pipeline...")
    
    # Build pipeline
    pipeline = build_pipeline(
        feature_method='filter_l1',
        model_name='lr_elasticnet',
        config=config
    )
    
    print(f"\nPipeline steps: {[name for name, _ in pipeline.steps]}")
    
    # Test nested CV (with small folds for speed)
    _, param_grid = get_model_config('lr_elasticnet', config)
    
    # Simplify param grid for testing
    param_grid = {
        'classifier__C': [0.1, 1.0],
        'classifier__l1_ratio': [0.5]
    }
    
    print("\nRunning nested CV (2x2 for testing)...")
    results = nested_cross_validation(
        X_df, y, pipeline, param_grid,
        outer_cv=2, inner_cv=2
    )
    
    print(f"\nCV Results:")
    print(f"  Mean validation score: {results['mean_val_score']:.3f}")
    print(f"  Std validation score: {results['std_val_score']:.3f}")
