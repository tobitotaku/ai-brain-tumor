"""
Visualization Module
====================
This module provides publication-ready plotting functions for EDA,
model evaluation, calibration, and explainability analysis.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def save_figure(fig, filepath: str, dpi: int = 300):
    """
    Save figure to file.
    
    Parameters:
        fig: Matplotlib figure object.
        filepath: Path to save figure.
        dpi: Resolution (dots per inch).
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved figure: {filepath}")


def plot_class_distribution(
    y: np.ndarray,
    labels: Optional[Dict[int, str]] = None,
    title: str = "Class Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution as bar chart.
    
    Parameters:
        y: Target labels.
        labels: Dictionary mapping class indices to names.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / counts.sum() * 100
    
    if labels is None:
        labels = {i: f"Class {i}" for i in unique}
    
    bars = ax.bar(
        range(len(unique)),
        counts,
        tick_label=[labels[i] for i in unique],
        color=['#3498db', '#e74c3c']
    )
    
    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{count}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    n_components: int = 50,
    title: str = "PCA Explained Variance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative explained variance for PCA.
    
    Parameters:
        explained_variance_ratio: Variance explained by each component.
        n_components: Number of components to show.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Individual variance
    ax1.bar(range(1, n_components + 1), explained_variance_ratio[:n_components])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Variance')
    ax1.grid(axis='y', alpha=0.3)
    
    # Cumulative variance
    cumsum = np.cumsum(explained_variance_ratio[:n_components])
    ax2.plot(range(1, n_components + 1), cumsum, marker='o', markersize=3)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    title: str = "ROC Curves",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Parameters:
        y_true: True labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = np.trapz(tpr, fpr)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    title: str = "Precision-Recall Curves",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curves for multiple models.
    
    Parameters:
        y_true: True labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    baseline = y_true.sum() / len(y_true)
    
    for model_name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = np.trapz(precision, recall)
        
        ax.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})', linewidth=2)
    
    # Baseline
    ax.axhline(y=baseline, color='k', linestyle='--', 
               label=f'Baseline ({baseline:.3f})', linewidth=1)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels for display.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)")
        annotations.append(row)
    
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    n_bins: int = 10,
    title: str = "Calibration Curves",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration (reliability) curves for multiple models.
    
    Parameters:
        y_true: True labels.
        y_prob_dict: Dictionary mapping model names to predicted probabilities.
        n_bins: Number of bins for calibration.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, y_prob in y_prob_dict.items():
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='quantile'
        )
        
        ax.plot(prob_pred, prob_true, marker='o', label=model_name, linewidth=2)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_decision_curve(
    dca_df: pd.DataFrame,
    title: str = "Decision Curve Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot decision curve showing net benefit.
    
    Parameters:
        dca_df: DataFrame from decision_curve_analysis() function.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        dca_df['threshold'],
        dca_df['net_benefit_model'],
        label='Model',
        linewidth=2,
        color='#3498db'
    )
    
    ax.plot(
        dca_df['threshold'],
        dca_df['net_benefit_all'],
        label='Treat All',
        linewidth=2,
        color='#e74c3c',
        linestyle='--'
    )
    
    ax.axhline(
        y=0,
        label='Treat None',
        linewidth=2,
        color='#95a5a6',
        linestyle=':'
    )
    
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters:
        importance_dict: Dictionary mapping feature names to importance scores.
        top_n: Number of top features to show.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, scores = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_stability_scores(
    stability_df: pd.DataFrame,
    top_n: int = 30,
    threshold: float = 0.7,
    title: str = "Feature Stability Scores",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature stability scores from bootstrap selection.
    
    Parameters:
        stability_df: DataFrame with 'feature' and 'stability_score' columns.
        top_n: Number of top features to show.
        threshold: Stability threshold to highlight.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    # Get top N features
    top_features = stability_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.25)))
    
    # Color bars based on threshold
    colors = ['#2ecc71' if score >= threshold else '#e74c3c' 
              for score in top_features['stability_score']]
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['stability_score'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Stability Score (Selection Frequency)')
    ax.set_title(title)
    ax.axvline(x=threshold, color='k', linestyle='--', linewidth=1, 
               label=f'Threshold ({threshold})')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: np.ndarray,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curve showing model performance vs training set size.
    
    Parameters:
        train_scores: Training scores for each size.
        val_scores: Validation scores for each size.
        train_sizes: Training set sizes.
        title: Plot title.
        save_path: Path to save figure.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    ax.plot(train_sizes, train_mean, label='Training Score', 
            color='#3498db', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#3498db')
    
    ax.plot(train_sizes, val_mean, label='Validation Score', 
            color='#e74c3c', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Testing visualization functions...")
    
    # Generate example data
    np.random.seed(42)
    y_true = np.random.binint(0, 2, 200)
    y_prob = np.random.rand(200)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Test plots
    print("\n1. Class distribution...")
    fig = plot_class_distribution(y_true, labels={0: 'Control', 1: 'GBM'})
    plt.close(fig)
    
    print("2. ROC curve...")
    fig = plot_roc_curve(y_true, {'Model': y_prob})
    plt.close(fig)
    
    print("3. Confusion matrix...")
    fig = plot_confusion_matrix(y_true, y_pred, labels=['Control', 'GBM'])
    plt.close(fig)
    
    print("\nAll visualization tests passed!")
