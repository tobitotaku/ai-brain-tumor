"""
Feature Selection Module
========================
This module implements multiple feature selection strategies including
filter methods, L1 regularization, PCA, and stability selection.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import resample
from typing import List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class FilterL1Selector(BaseEstimator, TransformerMixin):
    """
    Feature selection using variance filter + correlation removal + L1 regularization.
    
    This multi-stage selector:
    1. Removes low-variance features
    2. Removes highly correlated features
    3. Applies L1-penalized logistic regression for final selection
    
    Parameters:
        k_best: Number of top features to select.
        variance_threshold: Minimum variance threshold.
        correlation_threshold: Maximum correlation threshold.
        C: Inverse regularization strength for L1.
        random_state: Random seed for reproducibility.
        
    Attributes:
        selected_features_: Names/indices of selected features.
        feature_importance_: Importance scores for selected features.
    """
    
    def __init__(
        self,
        k_best: int = 200,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        C: float = 1.0,
        random_state: int = 42
    ):
        self.k_best = k_best
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.C = C
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importance_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        """
        Fit the feature selector on training data.
        
        Parameters:
            X: Feature matrix (samples × genes).
            y: Target labels.
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if not is_dataframe:
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        logger.info(f"Starting FilterL1 selection from {X.shape[1]} features")
        
        # Step 1: Variance filtering
        var_filter = VarianceThreshold(threshold=self.variance_threshold)
        X_var = var_filter.fit_transform(X)
        var_features = X.columns[var_filter.get_support()].tolist()
        
        logger.info(f"After variance filter: {len(var_features)} features")
        
        # Step 2: Correlation filtering
        X_var_df = pd.DataFrame(X_var, columns=var_features, index=X.index)
        corr_matrix = X_var_df.corr().abs()
        
        # Remove one feature from each highly correlated pair
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        to_drop = [
            column for column in corr_matrix.columns
            if any(corr_matrix[column].where(upper_tri[:, list(corr_matrix.columns).index(column)]) 
                   > self.correlation_threshold)
        ]
        
        corr_features = [f for f in var_features if f not in to_drop]
        X_corr = X_var_df[corr_features]
        
        logger.info(f"After correlation filter: {len(corr_features)} features")
        
        # Step 3: L1 regularization
        l1_model = LogisticRegression(
            penalty='l1',
            C=self.C,
            solver='liblinear',
            max_iter=1000,
            random_state=self.random_state
        )
        
        l1_model.fit(X_corr, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(l1_model.coef_[0])
        
        # Select top k features
        k_actual = min(self.k_best, len(corr_features))
        top_indices = np.argsort(importance)[-k_actual:][::-1]
        
        self.selected_features_ = [corr_features[i] for i in top_indices]
        self.feature_importance_ = {
            corr_features[i]: importance[i] for i in top_indices
        }
        
        logger.info(f"Final L1 selection: {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Transform data by selecting features.
        
        Parameters:
            X: Feature matrix.
            
        Returns:
            Transformed matrix with selected features only.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # For numpy array, assume same column order as fit
            feature_indices = [i for i, col in enumerate(self.selected_features_)]
            return X[:, feature_indices]
    
    def get_feature_names_out(self, input_features=None):
        """Get names of selected features (sklearn compatibility)."""
        return self.selected_features_


class OptimizedFilterL1Selector(BaseEstimator, TransformerMixin):
    """
    Optimized Filter+L1 with ANOVA pre-filtering for computational efficiency.
    
    Multi-stage pipeline for high-dimensional feature selection:
    1. Variance threshold: Remove low-variance features
    2. ANOVA F-test: Select top-k univariate features (KEY OPTIMIZATION)
    3. Correlation pruning: Remove redundant features
    4. L1 regularization: Final multivariate selection
    
    This approach reduces correlation computation from O(p²) to O(k²) where k << p,
    enabling efficient processing of high-dimensional genomic data (18K+ features).
    
    Parameters:
        k_best: Number of final features to select.
        k_prefilter: Number of features after ANOVA pre-filtering (reduces corr. cost).
        variance_threshold: Minimum variance threshold.
        correlation_threshold: Maximum correlation threshold.
        C: Inverse regularization strength for L1.
        random_state: Random seed for reproducibility.
        
    Attributes:
        selected_features_: Names/indices of selected features.
        feature_importance_: Importance scores for selected features.
        
    References:
        - Saeys et al. (2007). Bioinformatics 23(19):2507-17
        - Guyon & Elisseeff (2003). JMLR 3:1157-82
    """
    
    def __init__(
        self,
        k_best: int = 200,
        k_prefilter: int = 2000,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        C: float = 1.0,
        random_state: int = 42
    ):
        self.k_best = k_best
        self.k_prefilter = k_prefilter
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.C = C
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importance_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        """
        Fit the optimized feature selector on training data.
        
        Parameters:
            X: Feature matrix (samples × genes).
            y: Target labels.
            
        Returns:
            self
        """
        # Convert to DataFrame if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        if not is_dataframe:
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        logger.info(f"OptimizedFilterL1: Starting from {X.shape[1]} features")
        
        # Step 1: Variance filtering
        var_filter = VarianceThreshold(threshold=self.variance_threshold)
        X_var = var_filter.fit_transform(X)
        var_features = X.columns[var_filter.get_support()].tolist()
        
        logger.info(f"  → After variance filter: {len(var_features)} features")
        
        # Step 2: ANOVA F-test pre-filtering (KEY OPTIMIZATION)
        X_var_df = pd.DataFrame(X_var, columns=var_features, index=X.index)
        k_prefilter_actual = min(self.k_prefilter, len(var_features))
        f_selector = SelectKBest(f_classif, k=k_prefilter_actual)
        X_prefiltered = f_selector.fit_transform(X_var_df, y)
        prefilter_mask = f_selector.get_support()
        prefiltered_features = [var_features[i] for i, selected in enumerate(prefilter_mask) if selected]
        
        logger.info(f"  → After ANOVA F-test: {len(prefiltered_features)} features (reduces corr. from {len(var_features)}² to {len(prefiltered_features)}²)")
        
        # Step 3: Correlation filtering (OPTIMIZED - vectorized, memory-efficient)
        X_prefiltered_array = X_prefiltered
        n_features = X_prefiltered_array.shape[1]
        
        # Use numpy for faster correlation computation
        # Normalize columns for correlation computation
        X_norm = (X_prefiltered_array - X_prefiltered_array.mean(axis=0)) / X_prefiltered_array.std(axis=0)
        
        # Compute correlation matrix efficiently
        corr_matrix = np.abs(np.corrcoef(X_norm.T))
        
        # Identify features to drop using upper triangle
        to_drop_indices = set()
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if corr_matrix[i, j] > self.correlation_threshold:
                    # Drop the feature with lower variance (keep more informative one)
                    if X_prefiltered_array[:, i].var() < X_prefiltered_array[:, j].var():
                        to_drop_indices.add(i)
                    else:
                        to_drop_indices.add(j)
        
        keep_indices = [i for i in range(n_features) if i not in to_drop_indices]
        corr_features = [prefiltered_features[i] for i in keep_indices]
        X_corr = X_prefiltered[:, keep_indices]
        
        logger.info(f"  → After correlation pruning: {len(corr_features)} features")
        
        # Step 4: L1 regularization
        l1_model = LogisticRegression(
            penalty='l1',
            C=self.C,
            solver='liblinear',
            max_iter=5000,  # Increased for convergence on high-dimensional data
            random_state=self.random_state
        )
        
        l1_model.fit(X_corr, y)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(l1_model.coef_[0])
        
        # Select top k features
        k_actual = min(self.k_best, len(corr_features))
        top_indices = np.argsort(importance)[-k_actual:][::-1]
        
        self.selected_features_ = [corr_features[i] for i in top_indices]
        self.feature_importance_ = {
            corr_features[i]: importance[i] for i in top_indices
        }
        
        logger.info(f"  → Final L1 selection: {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Transform data by selecting features.
        
        Parameters:
            X: Feature matrix.
            
        Returns:
            Transformed matrix with selected features only.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # For numpy array, assume same column order as fit
            raise NotImplementedError("OptimizedFilterL1Selector requires DataFrame input for safety")
    
    def get_feature_names_out(self, input_features=None):
        """Get names of selected features (sklearn compatibility)."""
        return self.selected_features_


class PCASelector(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Principal Component Analysis.
    
    This unsupervised method projects data onto principal components
    that capture the most variance.
    
    Parameters:
        n_components: Number of components to keep.
        random_state: Random seed for reproducibility.
        
    Attributes:
        pca_: Fitted PCA object.
        explained_variance_ratio_: Variance explained by each component.
    """
    
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit PCA on training data.
        
        Parameters:
            X: Feature matrix.
            y: Target (ignored, for compatibility).
            
        Returns:
            self
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(X_array)
        
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        total_variance = self.explained_variance_ratio_.sum()
        
        logger.info(
            f"PCA fitted: {self.n_components} components explain "
            f"{total_variance*100:.2f}% variance"
        )
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Transform data to PCA space.
        
        Parameters:
            X: Feature matrix.
            
        Returns:
            Transformed matrix with PCA components.
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_pca = self.pca_.transform(X_array)
        
        # Return as DataFrame with PC names
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_pca,
                index=X.index,
                columns=[f"PC{i+1}" for i in range(self.n_components)]
            )
        else:
            return X_pca
    
    def get_feature_names_out(self, input_features=None):
        """Get names of PCA components."""
        return [f"PC{i+1}" for i in range(self.n_components)]


class BioPanelSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection using a curated biological panel of genes.
    
    This selector uses domain knowledge to select pre-defined genes
    relevant to the biological process (e.g., GBM-related genes).
    
    Parameters:
        gene_list: List of gene symbols to select.
        
    Attributes:
        selected_features_: Names of genes found in data.
    """
    
    def __init__(self, gene_list: Optional[List[str]] = None):
        self.gene_list = gene_list or []
        self.selected_features_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Identify which genes from the panel are present in the data.
        
        Parameters:
            X: Feature matrix.
            y: Target (ignored).
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            available_genes = X.columns.tolist()
        else:
            # For numpy, assume gene_list corresponds to column indices
            available_genes = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Find intersection of panel genes and available genes
        self.selected_features_ = [
            gene for gene in self.gene_list if gene in available_genes
        ]
        
        missing_genes = len(self.gene_list) - len(self.selected_features_)
        
        logger.info(
            f"BioPanelSelector: {len(self.selected_features_)} genes found, "
            f"{missing_genes} missing from data"
        )
        
        if len(self.selected_features_) == 0:
            logger.warning("No genes from bio panel found in data!")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Select genes from biological panel.
        
        Parameters:
            X: Feature matrix.
            
        Returns:
            Matrix with panel genes only.
        """
        if len(self.selected_features_) == 0:
            raise ValueError("No features selected - check gene panel")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # For numpy, would need index mapping
            raise NotImplementedError("BioPanelSelector requires DataFrame input")
    
    def get_feature_names_out(self, input_features=None):
        """Get names of selected genes."""
        return self.selected_features_


class StabilitySelector:
    """
    Stability selection using bootstrap resampling.
    
    This method runs feature selection multiple times on bootstrap samples
    and identifies features that are consistently selected (stable features).
    
    Parameters:
        base_selector: Feature selector to use (e.g., FilterL1Selector).
        n_bootstrap: Number of bootstrap iterations.
        threshold: Minimum selection frequency to consider feature stable.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        base_selector: BaseEstimator,
        n_bootstrap: int = 100,
        threshold: float = 0.7,
        random_state: int = 42
    ):
        self.base_selector = base_selector
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.random_state = random_state
        self.stability_scores_ = None
        self.stable_features_ = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StabilitySelector':
        """
        Perform stability selection.
        
        Parameters:
            X: Feature matrix.
            y: Target labels.
            
        Returns:
            self
        """
        logger.info(f"Starting stability selection with {self.n_bootstrap} iterations")
        
        feature_counts = {col: 0 for col in X.columns}
        
        np.random.seed(self.random_state)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = resample(
                np.arange(len(X)),
                replace=True,
                n_samples=len(X),
                random_state=self.random_state + i
            )
            
            X_boot = X.iloc[indices]
            y_boot = y[indices]
            
            # Fit selector
            selector = clone_selector(self.base_selector)
            selector.fit(X_boot, y_boot)
            
            # Count selected features
            selected = selector.selected_features_
            for feature in selected:
                if feature in feature_counts:
                    feature_counts[feature] += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i+1}/{self.n_bootstrap} iterations")
        
        # Calculate stability scores
        self.stability_scores_ = {
            feature: count / self.n_bootstrap
            for feature, count in feature_counts.items()
        }
        
        # Select stable features
        self.stable_features_ = [
            feature for feature, score in self.stability_scores_.items()
            if score >= self.threshold
        ]
        
        logger.info(
            f"Stability selection complete: {len(self.stable_features_)} stable features "
            f"(threshold={self.threshold})"
        )
        
        return self
    
    def get_stable_features(self) -> List[str]:
        """Get list of stable features."""
        return self.stable_features_
    
    def get_stability_scores(self) -> pd.DataFrame:
        """
        Get stability scores as a DataFrame.
        
        Returns:
            DataFrame with features and their stability scores.
        """
        df = pd.DataFrame([
            {'feature': feature, 'stability_score': score}
            for feature, score in self.stability_scores_.items()
        ]).sort_values('stability_score', ascending=False)
        
        return df


def clone_selector(selector: BaseEstimator) -> BaseEstimator:
    """
    Create a copy of a selector with same parameters.
    
    Parameters:
        selector: Selector to clone.
        
    Returns:
        New selector instance.
    """
    from sklearn.base import clone
    return clone(selector)


def create_feature_selector(
    method: str,
    config: dict
) -> BaseEstimator:
    """
    Create a feature selector based on method name and configuration.
    
    Parameters:
        method: Feature selection method name.
        config: Configuration dictionary.
        
    Returns:
        Feature selector instance.
    """
    if method == "filter_l1":
        return FilterL1Selector(
            k_best=config['features'].get('k_best', 200),
            variance_threshold=config['preprocessing'].get('variance_threshold', 0.01),
            correlation_threshold=config['preprocessing'].get('correlation_threshold', 0.95),
            random_state=config.get('random_state', 42)
        )
    
    elif method == "filter_l1_optimized":
        return OptimizedFilterL1Selector(
            k_best=config['features'].get('k_best', 200),
            k_prefilter=config['features'].get('k_prefilter', 2000),
            variance_threshold=config['preprocessing'].get('variance_threshold', 0.01),
            correlation_threshold=config['preprocessing'].get('correlation_threshold', 0.95),
            C=config['features'].get('l1_C', 1.0),
            random_state=config.get('random_state', 42)
        )
    
    elif method == "pca":
        return PCASelector(
            n_components=config['features'].get('pca_components', 50),
            random_state=config.get('random_state', 42)
        )
    
    elif method == "bio_panel":
        # Load gene list
        from .data import load_gene_list
        gene_list_path = config['features'].get('bio_panel_path', 'metadata/biopanel.csv')
        
        try:
            genes = load_gene_list(gene_list_path)
            return BioPanelSelector(gene_list=genes)
        except FileNotFoundError:
            logger.warning(f"Bio panel file not found: {gene_list_path}")
            return BioPanelSelector(gene_list=[])
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate example data
    X, y = make_classification(
        n_samples=200,
        n_features=1000,
        n_informative=50,
        n_redundant=100,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f"gene_{i}" for i in range(X.shape[1])])
    
    print("\nTesting feature selection methods...")
    
    # Test Filter + L1
    print("\n1. FilterL1Selector:")
    filter_l1 = FilterL1Selector(k_best=100)
    filter_l1.fit(X_df, y)
    X_l1 = filter_l1.transform(X_df)
    print(f"   Selected {X_l1.shape[1]} features")
    print(f"   Top 5 features: {list(filter_l1.feature_importance_.keys())[:5]}")
    
    # Test PCA
    print("\n2. PCASelector:")
    pca_sel = PCASelector(n_components=50)
    pca_sel.fit(X_df, y)
    X_pca = pca_sel.transform(X_df)
    print(f"   {X_pca.shape[1]} components")
    print(f"   Explained variance: {pca_sel.explained_variance_ratio_[:5]}")
    
    # Test Stability Selection
    print("\n3. StabilitySelector:")
    stability = StabilitySelector(
        base_selector=FilterL1Selector(k_best=100),
        n_bootstrap=50,
        threshold=0.7
    )
    stability.fit(X_df, y)
    print(f"   {len(stability.stable_features_)} stable features")
    
    scores_df = stability.get_stability_scores()
    print(f"   Top 5 stable features:")
    print(scores_df.head())
