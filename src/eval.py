"""
Evaluation Module
=================
This module implements comprehensive evaluation metrics, calibration analysis,
and decision curve analysis for model assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.
        
    Returns:
        Dictionary of metric names and values.
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # NPV and PPV
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Probability-based metrics (if probabilities available)
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = np.nan
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['pr_auc'] = np.nan
        
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        metrics['log_loss'] = log_loss(y_true, y_prob)
    
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a metric using bootstrap.
    
    Parameters:
        y_true: True labels.
        y_prob: Predicted probabilities.
        metric_func: Function to calculate metric (e.g., roc_auc_score).
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (default: 0.95).
        random_state: Random seed.
        
    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci).
    """
    np.random.seed(random_state)
    
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Ensure both classes are present
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            score = metric_func(y_true[indices], y_prob[indices])
            bootstrap_scores.append(score)
        except:
            continue
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    point_estimate = metric_func(y_true, y_prob)
    lower_ci = np.percentile(bootstrap_scores, lower_percentile)
    upper_ci = np.percentile(bootstrap_scores, upper_percentile)
    
    return point_estimate, lower_ci, upper_ci


def calculate_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Calculate metrics with confidence intervals.
    
    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level.
        
    Returns:
        DataFrame with metrics and confidence intervals.
    """
    results = []
    
    # ROC-AUC
    if y_prob is not None:
        auc_point, auc_lower, auc_upper = bootstrap_ci(
            y_true, y_prob, roc_auc_score, n_bootstrap, confidence
        )
        results.append({
            'metric': 'ROC-AUC',
            'value': auc_point,
            'ci_lower': auc_lower,
            'ci_upper': auc_upper,
            'ci': f"{auc_point:.3f} ({auc_lower:.3f}-{auc_upper:.3f})"
        })
        
        # PR-AUC
        pr_point, pr_lower, pr_upper = bootstrap_ci(
            y_true, y_prob, average_precision_score, n_bootstrap, confidence
        )
        results.append({
            'metric': 'PR-AUC',
            'value': pr_point,
            'ci_lower': pr_lower,
            'ci_upper': pr_upper,
            'ci': f"{pr_point:.3f} ({pr_lower:.3f}-{pr_upper:.3f})"
        })
    
    # Basic metrics (without CI for simplicity)
    basic_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        value = basic_metrics[metric_name]
        results.append({
            'metric': metric_name.capitalize(),
            'value': value,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci': f"{value:.3f}"
        })
    
    return pd.DataFrame(results)


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Calculate calibration metrics and curve.
    
    Calibration measures how well predicted probabilities match observed frequencies.
    
    Parameters:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.
        
    Returns:
        Dictionary with calibration metrics and curve data.
    """
    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_prob)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='quantile'
    )
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))
    
    return {
        'brier_score': brier,
        'ece': ece,
        'mce': mce,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Perform decision curve analysis.
    
    Decision curves show the net benefit of using a model at different
    probability thresholds compared to treating all or no patients.
    
    Parameters:
        y_true: True labels.
        y_prob: Predicted probabilities.
        thresholds: Array of threshold values to evaluate.
        
    Returns:
        DataFrame with net benefit at each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    results = []
    n = len(y_true)
    prevalence = y_true.sum() / n
    
    for threshold in thresholds:
        # Classify based on threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Net benefit of the model
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        
        # Net benefit of treating all
        net_benefit_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        
        # Net benefit of treating none
        net_benefit_none = 0
        
        results.append({
            'threshold': threshold,
            'net_benefit_model': net_benefit,
            'net_benefit_all': max(0, net_benefit_all),
            'net_benefit_none': net_benefit_none
        })
    
    return pd.DataFrame(results)


def compare_models(
    results_dict: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Parameters:
        results_dict: Dictionary mapping model names to their metrics.
        
    Returns:
        DataFrame comparing models across metrics.
    """
    comparison = pd.DataFrame(results_dict).T
    
    # Highlight best model for each metric
    comparison_styled = comparison.copy()
    
    # Metrics where higher is better
    higher_better = ['accuracy', 'precision', 'recall', 'f1', 
                     'roc_auc', 'pr_auc', 'specificity', 'sensitivity']
    
    # Metrics where lower is better
    lower_better = ['brier_score', 'log_loss', 'ece', 'mce']
    
    for metric in higher_better:
        if metric in comparison.columns:
            best_idx = comparison[metric].idxmax()
            comparison_styled.loc[best_idx, metric] = f"**{comparison.loc[best_idx, metric]:.3f}**"
    
    for metric in lower_better:
        if metric in comparison.columns:
            best_idx = comparison[metric].idxmin()
            comparison_styled.loc[best_idx, metric] = f"**{comparison.loc[best_idx, metric]:.3f}**"
    
    return comparison


def statistical_comparison(
    y_true: np.ndarray,
    y_prob1: np.ndarray,
    y_prob2: np.ndarray,
    metric: str = 'roc_auc',
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict:
    """
    Statistically compare two models using bootstrap.
    
    Parameters:
        y_true: True labels.
        y_prob1: Probabilities from model 1.
        y_prob2: Probabilities from model 2.
        metric: Metric to compare ('roc_auc' or 'pr_auc').
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed.
        
    Returns:
        Dictionary with comparison results.
    """
    np.random.seed(random_state)
    
    if metric == 'roc_auc':
        metric_func = roc_auc_score
    elif metric == 'pr_auc':
        metric_func = average_precision_score
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Point estimates
    score1 = metric_func(y_true, y_prob1)
    score2 = metric_func(y_true, y_prob2)
    
    # Bootstrap
    differences = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            s1 = metric_func(y_true[indices], y_prob1[indices])
            s2 = metric_func(y_true[indices], y_prob2[indices])
            differences.append(s1 - s2)
        except:
            continue
    
    differences = np.array(differences)
    
    # Calculate p-value (two-tailed)
    p_value = (np.sum(differences <= 0) / len(differences))
    p_value = 2 * min(p_value, 1 - p_value)
    
    # Confidence interval of difference
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        'model1_score': score1,
        'model2_score': score2,
        'difference': score1 - score2,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = lr.predict(X_test)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    
    print("=== Evaluation Examples ===\n")
    
    # Basic metrics
    print("1. Basic Metrics (Logistic Regression):")
    metrics = calculate_metrics(y_test, y_pred_lr, y_prob_lr)
    for k, v in list(metrics.items())[:5]:
        print(f"   {k}: {v:.3f}")
    
    # Metrics with CI
    print("\n2. Metrics with Confidence Intervals:")
    metrics_ci = calculate_metrics_with_ci(y_test, y_pred_lr, y_prob_lr, n_bootstrap=100)
    print(metrics_ci.head())
    
    # Calibration
    print("\n3. Calibration Metrics:")
    cal_metrics = calibration_metrics(y_test, y_prob_lr)
    print(f"   Brier Score: {cal_metrics['brier_score']:.3f}")
    print(f"   ECE: {cal_metrics['ece']:.3f}")
    
    # Model comparison
    print("\n4. Statistical Comparison (LR vs RF):")
    comparison = statistical_comparison(y_test, y_prob_lr, y_prob_rf, n_bootstrap=100)
    print(f"   LR AUC: {comparison['model1_score']:.3f}")
    print(f"   RF AUC: {comparison['model2_score']:.3f}")
    print(f"   Difference: {comparison['difference']:.3f}")
    print(f"   P-value: {comparison['p_value']:.3f}")
