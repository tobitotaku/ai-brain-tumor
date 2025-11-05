"""
Model Definitions Module
=======================
This module defines machine learning models for GBM classification,
including Logistic Regression, Random Forest, and Gradient Boosting.

Author: Musab 0988932
Date: November 2025
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def get_model_config(model_name: str, config: dict) -> Tuple[Any, Dict]:
    """
    Get model instance and parameter grid from configuration.
    
    Parameters:
        model_name: Name of the model ('lr_elasticnet', 'random_forest', 'lightgbm', 'xgboost').
        config: Configuration dictionary.
        
    Returns:
        Tuple of (model_instance, param_grid).
    """
    models_config = config.get('models', {})
    
    if model_name not in models_config:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    model_config = models_config[model_name]
    
    if not model_config.get('enabled', True):
        logger.warning(f"Model {model_name} is disabled in configuration")
        return None, None
    
    # Get parameter grid
    param_grid = model_config.get('param_grid', {})
    
    # Create model instance
    if model_name == 'lr_elasticnet':
        model = create_logistic_regression()
    elif model_name == 'random_forest':
        model = create_random_forest()
    elif model_name == 'lightgbm':
        model = create_lightgbm()
    elif model_name == 'xgboost':
        model = create_xgboost()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Created {model_name} with {len(param_grid)} hyperparameters to tune")
    
    return model, param_grid


def create_logistic_regression() -> LogisticRegression:
    """
    Create Logistic Regression with ElasticNet regularization.
    
    ElasticNet combines L1 (feature selection) and L2 (regularization)
    penalties, controlled by the l1_ratio parameter.
    
    Returns:
        LogisticRegression instance.
    """
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',  # Supports elasticnet
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    return model


def create_random_forest() -> RandomForestClassifier:
    """
    Create Random Forest classifier.
    
    Random Forest is an ensemble of decision trees that provides
    good performance and feature importance estimates.
    
    Returns:
        RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=False  # Will use CV for evaluation
    )
    
    return model


def create_lightgbm() -> LGBMClassifier:
    """
    Create LightGBM classifier.
    
    LightGBM is a gradient boosting framework that uses tree-based
    learning algorithms, optimized for speed and efficiency.
    
    Returns:
        LGBMClassifier instance.
    """
    model = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1  # Suppress warnings
    )
    
    return model


def create_xgboost() -> XGBClassifier:
    """
    Create XGBoost classifier.
    
    XGBoost is a powerful gradient boosting implementation with
    regularization and advanced optimization.
    
    Returns:
        XGBClassifier instance.
    """
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0  # Suppress warnings
    )
    
    return model


def get_all_models(config: dict) -> Dict[str, Tuple[Any, Dict]]:
    """
    Get all enabled models and their parameter grids.
    
    Parameters:
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping model names to (model, param_grid) tuples.
    """
    model_names = ['lr_elasticnet', 'random_forest', 'lightgbm', 'xgboost']
    models = {}
    
    for name in model_names:
        try:
            model, param_grid = get_model_config(name, config)
            if model is not None:
                models[name] = (model, param_grid)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping model {name}: {e}")
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    return models


class ModelWrapper:
    """
    Wrapper for sklearn models with additional metadata.
    
    This class stores a fitted model along with training information,
    feature names, and performance metrics.
    
    Attributes:
        model: Fitted sklearn model.
        model_name: Name of the model type.
        feature_names: Names of features used for training.
        training_info: Dictionary with training metadata.
        performance: Dictionary with performance metrics.
    """
    
    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        self.feature_names = None
        self.training_info = {}
        self.performance = {}
    
    def fit(self, X, y, **kwargs):
        """Fit the wrapped model."""
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def set_training_info(self, **kwargs):
        """Store training metadata."""
        self.training_info.update(kwargs)
    
    def set_performance(self, **kwargs):
        """Store performance metrics."""
        self.performance.update(kwargs)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.feature_names is None:
            logger.warning("No feature names available")
            return {}
        
        # Get importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (RF, LightGBM, XGBoost)
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models (Logistic Regression)
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning(f"Model {self.model_name} does not support feature importance")
            return {}
        
        return dict(zip(self.feature_names, importance))
    
    def __repr__(self):
        return f"ModelWrapper({self.model_name}, features={len(self.feature_names) if self.feature_names else 'unknown'})"


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Example configuration
    example_config = {
        'models': {
            'lr_elasticnet': {
                'enabled': True,
                'param_grid': {
                    'C': [0.1, 1.0],
                    'l1_ratio': [0.5]
                }
            },
            'random_forest': {
                'enabled': True,
                'param_grid': {
                    'n_estimators': [100],
                    'max_depth': [10]
                }
            },
            'lightgbm': {
                'enabled': True,
                'param_grid': {
                    'n_estimators': [100],
                    'learning_rate': [0.1]
                }
            }
        }
    }
    
    # Test model creation
    print("Testing model creation...")
    models = get_all_models(example_config)
    
    for name, (model, param_grid) in models.items():
        print(f"\n{name}:")
        print(f"  Model: {type(model).__name__}")
        print(f"  Params to tune: {list(param_grid.keys())}")
    
    # Test model wrapper
    print("\n\nTesting ModelWrapper...")
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    lr_model, _ = get_model_config('lr_elasticnet', example_config)
    wrapper = ModelWrapper(lr_model, 'lr_elasticnet')
    wrapper.fit(X, y)
    
    predictions = wrapper.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")
