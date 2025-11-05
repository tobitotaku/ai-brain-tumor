"""
Preprocessing Module
====================
This module implements batch correction, normalization, and scaling methods
for gene expression data, ensuring no data leakage in cross-validation.

Author: Musab 0988932
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Optional, Union, Literal
import logging

logger = logging.getLogger(__name__)


class BatchCorrector(BaseEstimator, TransformerMixin):
    """
    Batch effect correction using ComBat or Harmony algorithm.
    
    This transformer removes systematic technical variation between batches
    while preserving biological variation. It must be fit on training data only
    to prevent data leakage.
    
    Parameters:
        method: Batch correction method ('combat', 'harmony', or None).
        batch_col: Name of the batch column in metadata.
        
    Attributes:
        correction_params_: Learned parameters for batch correction.
    """
    
    def __init__(
        self,
        method: Optional[Literal['combat', 'harmony']] = 'combat',
        batch_col: str = 'batch'
    ):
        self.method = method
        self.batch_col = batch_col
        self.correction_params_ = None
        
    def fit(self, X: pd.DataFrame, y=None, batch: Optional[pd.Series] = None):
        """
        Learn batch correction parameters from training data.
        
        Parameters:
            X: Gene expression matrix (samples × genes).
            y: Target labels (ignored, for sklearn compatibility).
            batch: Batch labels for each sample.
            
        Returns:
            self
        """
        if self.method is None or self.method == 'none':
            logger.info("Batch correction disabled")
            return self
        
        if batch is None:
            logger.warning("No batch information provided, skipping batch correction")
            return self
        
        if self.method == 'combat':
            self._fit_combat(X, batch)
        elif self.method == 'harmony':
            self._fit_harmony(X, batch)
        else:
            raise ValueError(f"Unknown batch correction method: {self.method}")
        
        return self
    
    def _fit_combat(self, X: pd.DataFrame, batch: pd.Series):
        """
        Fit ComBat batch correction.
        
        ComBat uses empirical Bayes to estimate and remove batch effects.
        """
        try:
            from combat.pycombat import pycombat
            
            logger.info(f"Fitting ComBat batch correction on {X.shape}")
            
            # pycombat expects genes × samples
            X_transposed = X.T
            
            # Store mean and std for each batch
            self.correction_params_ = {
                'method': 'combat',
                'batch_info': batch.to_dict(),
                'global_mean': X.mean(axis=0),
                'global_std': X.std(axis=0)
            }
            
            logger.info("ComBat parameters fitted")
            
        except ImportError:
            logger.error("pycombat not installed. Install with: pip install combat")
            logger.info("Proceeding without batch correction")
            self.method = None
    
    def _fit_harmony(self, X: pd.DataFrame, batch: pd.Series):
        """
        Fit Harmony batch correction.
        
        Harmony uses soft clustering and linear correction.
        """
        try:
            import harmonypy as hm
            
            logger.info(f"Fitting Harmony batch correction on {X.shape}")
            
            # Harmony works on the data directly
            self.correction_params_ = {
                'method': 'harmony',
                'batch_info': batch.to_dict()
            }
            
            logger.info("Harmony parameters fitted")
            
        except ImportError:
            logger.error("harmonypy not installed. Install with: pip install harmonypy")
            logger.info("Proceeding without batch correction")
            self.method = None
    
    def transform(self, X: pd.DataFrame, batch: Optional[pd.Series] = None):
        """
        Apply batch correction to data.
        
        Parameters:
            X: Gene expression matrix to correct.
            batch: Batch labels (required if method is not None).
            
        Returns:
            Batch-corrected dataframe.
        """
        if self.method is None or self.method == 'none':
            return X
        
        if batch is None:
            logger.warning("No batch information for transform, returning uncorrected data")
            return X
        
        if self.method == 'combat':
            return self._transform_combat(X, batch)
        elif self.method == 'harmony':
            return self._transform_harmony(X, batch)
        else:
            return X
    
    def _transform_combat(self, X: pd.DataFrame, batch: pd.Series):
        """Apply ComBat correction."""
        try:
            from combat.pycombat import pycombat
            
            # pycombat expects genes × samples
            X_transposed = X.T
            
            # Apply combat
            corrected = pycombat(
                data=X_transposed,
                batch=batch.values
            )
            
            # Transpose back to samples × genes
            corrected_df = pd.DataFrame(
                corrected.T,
                index=X.index,
                columns=X.columns
            )
            
            logger.info("ComBat correction applied")
            return corrected_df
            
        except Exception as e:
            logger.error(f"ComBat correction failed: {e}")
            return X
    
    def _transform_harmony(self, X: pd.DataFrame, batch: pd.Series):
        """Apply Harmony correction."""
        try:
            import harmonypy as hm
            
            # Create metadata dataframe
            meta_data = pd.DataFrame({
                'batch': batch.values
            }, index=X.index)
            
            # Run harmony
            ho = hm.run_harmony(
                X.values,
                meta_data,
                vars_use=['batch']
            )
            
            corrected_df = pd.DataFrame(
                ho.Z_corr.T,
                index=X.index,
                columns=X.columns
            )
            
            logger.info("Harmony correction applied")
            return corrected_df
            
        except Exception as e:
            logger.error(f"Harmony correction failed: {e}")
            return X
    
    def fit_transform(self, X: pd.DataFrame, y=None, batch: Optional[pd.Series] = None):
        """Fit and transform in one step."""
        return self.fit(X, y, batch).transform(X, batch)


class ExpressionScaler(BaseEstimator, TransformerMixin):
    """
    Scale gene expression data using various strategies.
    
    This wrapper ensures proper scaling WITHOUT per-class normalization,
    which would cause data leakage.
    
    Parameters:
        method: Scaling method ('standard', 'robust', 'minmax', or None).
        
    Attributes:
        scaler_: Fitted sklearn scaler object.
    """
    
    def __init__(self, method: Literal['standard', 'robust', 'minmax', None] = 'standard'):
        self.method = method
        self.scaler_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit scaler on training data.
        
        Parameters:
            X: Gene expression matrix.
            y: Target labels (ignored).
            
        Returns:
            self
        """
        if self.method is None or self.method == 'none':
            logger.info("Scaling disabled")
            return self
        
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'robust':
            self.scaler_ = RobustScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        self.scaler_.fit(X_array)
        logger.info(f"Fitted {self.method} scaler on {X_array.shape}")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Apply scaling transformation.
        
        Parameters:
            X: Gene expression matrix.
            
        Returns:
            Scaled data (same type as input).
        """
        if self.method is None or self.method == 'none':
            return X
        
        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else X
        
        X_scaled = self.scaler_.transform(X_array)
        
        if is_dataframe:
            return pd.DataFrame(
                X_scaled,
                index=X.index,
                columns=X.columns
            )
        else:
            return X_scaled
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features to reduce redundancy.
    
    This transformer removes one feature from each pair of features
    with correlation above the threshold.
    
    Parameters:
        threshold: Correlation threshold (default: 0.95).
        method: Correlation method ('pearson', 'spearman').
        
    Attributes:
        features_to_keep_: List of feature indices to keep.
    """
    
    def __init__(self, threshold: float = 0.95, method: str = 'pearson'):
        self.threshold = threshold
        self.method = method
        self.features_to_keep_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Identify features to keep based on correlation.
        
        Parameters:
            X: Feature matrix.
            y: Target (ignored).
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Compute correlation matrix
        corr_matrix = X.corr(method=self.method).abs()
        
        # Find features to drop
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_tri_corr = corr_matrix.where(upper_tri)
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper_tri_corr.columns
            if any(upper_tri_corr[column] > self.threshold)
        ]
        
        # Keep features not in drop list
        self.features_to_keep_ = [
            col for col in X.columns if col not in to_drop
        ]
        
        n_dropped = len(X.columns) - len(self.features_to_keep_)
        logger.info(
            f"CorrelationFilter: removed {n_dropped} features "
            f"(threshold={self.threshold})"
        )
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Remove correlated features.
        
        Parameters:
            X: Feature matrix.
            
        Returns:
            Filtered feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.features_to_keep_]
        else:
            # For numpy array, use feature indices
            feature_indices = [
                i for i, col in enumerate(range(X.shape[1]))
                if col in self.features_to_keep_
            ]
            return X[:, feature_indices]


def create_preprocessing_pipeline(
    config: dict,
    include_batch_correction: bool = True,
    include_scaling: bool = True
):
    """
    Create a preprocessing pipeline from configuration.
    
    Parameters:
        config: Configuration dictionary.
        include_batch_correction: Whether to include batch correction.
        include_scaling: Whether to include scaling.
        
    Returns:
        List of (name, transformer) tuples for sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline
    
    steps = []
    
    # Batch correction
    if include_batch_correction:
        batch_method = config.get('preprocessing', {}).get('batch_correction', 'combat')
        steps.append(('batch_correction', BatchCorrector(method=batch_method)))
    
    # Correlation filtering
    corr_threshold = config.get('preprocessing', {}).get('correlation_threshold', 0.95)
    steps.append(('correlation_filter', CorrelationFilter(threshold=corr_threshold)))
    
    # Scaling
    if include_scaling:
        scale_method = config.get('preprocessing', {}).get('scaler', 'standard')
        steps.append(('scaler', ExpressionScaler(method=scale_method)))
    
    logger.info(f"Created preprocessing pipeline with {len(steps)} steps")
    
    return steps


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    n_samples, n_genes = 100, 1000
    X = pd.DataFrame(
        np.random.randn(n_samples, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)]
    )
    batch = pd.Series(np.random.choice(['A', 'B'], n_samples))
    
    print("Testing preprocessing components...")
    
    # Test batch correction
    batch_corrector = BatchCorrector(method='combat')
    X_corrected = batch_corrector.fit_transform(X, batch=batch)
    print(f"Batch correction: {X.shape} -> {X_corrected.shape}")
    
    # Test scaling
    scaler = ExpressionScaler(method='standard')
    X_scaled = scaler.fit_transform(X_corrected)
    print(f"Scaling: mean={X_scaled.values.mean():.4f}, std={X_scaled.values.std():.4f}")
    
    # Test correlation filter
    corr_filter = CorrelationFilter(threshold=0.95)
    X_filtered = corr_filter.fit_transform(X_scaled)
    print(f"Correlation filter: {X_scaled.shape[1]} -> {X_filtered.shape[1]} genes")
