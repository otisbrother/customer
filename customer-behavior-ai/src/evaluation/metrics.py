"""
Evaluation metrics for model assessment
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, average_precision_score
)
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import FIGURES_DIR, create_directories


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: np.ndarray = None,
                            verbose: bool = True) -> dict:
    """
    Comprehensive evaluation of classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for class 1)
        verbose: Whether to print results
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    if verbose:
        print("\n" + "=" * 50)
        print("Classification Metrics")
        print("=" * 50)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        if y_proba is not None:
            print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
            print(f"PR AUC:      {metrics['pr_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  TN: {tn:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TP: {tp:5d}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list = None, 
                          save_path: Path = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if labels is None:
        labels = ['No Churn', 'Churn']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'confusion_matrix.png', dpi=150)
        print(f"Saved confusion matrix to {save_path / 'confusion_matrix.png'}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                    save_path: Path = None) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'roc_curve.png', dpi=150)
        print(f"Saved ROC curve to {save_path / 'roc_curve.png'}")
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                 save_path: Path = None) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='green')
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'precision_recall_curve.png', dpi=150)
        print(f"Saved PR curve to {save_path / 'precision_recall_curve.png'}")
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                             save_path: Path = None) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    df = importance_df.head(top_n).sort_values('importance')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df['feature'], df['importance'], color='steelblue')
    
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'feature_importance.png', dpi=150)
        print(f"Saved feature importance to {save_path / 'feature_importance.png'}")
    
    return fig


def plot_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                             save_path: Path = None) -> plt.Figure:
    """
    Plot metrics at different classification thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        metrics['threshold'].append(thresh)
        metrics['precision'].append(precision_score(y_true, y_pred_thresh, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred_thresh, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred_thresh, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['threshold'], metrics['precision'], 'b-', label='Precision', lw=2)
    ax.plot(metrics['threshold'], metrics['recall'], 'g-', label='Recall', lw=2)
    ax.plot(metrics['threshold'], metrics['f1'], 'r-', label='F1 Score', lw=2)
    
    # Find optimal threshold (max F1)
    best_idx = np.argmax(metrics['f1'])
    best_thresh = metrics['threshold'][best_idx]
    ax.axvline(x=best_thresh, color='gray', linestyle='--', 
               label=f'Optimal threshold ({best_thresh:.2f})')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Classification Threshold')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'threshold_analysis.png', dpi=150)
        print(f"Saved threshold analysis to {save_path / 'threshold_analysis.png'}")
    
    return fig


def plot_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: np.ndarray = None,
                            save_path: Path = None):
    """
    Generate all evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        save_path: Path to save figures
    """
    create_directories()
    
    if save_path is None:
        save_path = FIGURES_DIR
    
    print("\nGenerating evaluation plots...")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=save_path)
    
    if y_proba is not None:
        # ROC curve
        plot_roc_curve(y_true, y_proba, save_path=save_path)
        
        # Precision-Recall curve
        plot_precision_recall_curve(y_true, y_proba, save_path=save_path)
        
        # Threshold analysis
        plot_threshold_analysis(y_true, y_proba, save_path=save_path)
    
    print(f"\nAll plots saved to {save_path}")


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                            metric: str = 'f1') -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Optimal threshold value
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    
    func = metric_funcs.get(metric, f1_score)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = func(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print(f"Optimal threshold for {metric}: {best_threshold:.2f} (score: {best_score:.4f})")
    return best_threshold


def get_classification_summary(metrics: dict) -> pd.DataFrame:
    """
    Convert metrics dict to summary DataFrame.
    
    Args:
        metrics: Dictionary with evaluation metrics
    
    Returns:
        Summary DataFrame
    """
    summary_metrics = {k: v for k, v in metrics.items() 
                       if not isinstance(v, np.ndarray)}
    
    df = pd.DataFrame([summary_metrics]).T
    df.columns = ['Value']
    df = df.round(4)
    
    return df
