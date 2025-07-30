"""
Performance Metrics Module

Evaluate binary classification models for credit risk/PD modeling.

- Input: y_true (data/raw/application_train.csv), y_score (models/l1_stacking/l1_xgb_oof_predictions.csv)
- Output: metrics, plots in validation_results/metrics_results/

Example:
    y_true = pd.read_csv("data/raw/application_train.csv")['TARGET']
    y_score = pd.read_csv("models/l1_stacking/l1_xgb_oof_predictions.csv")['oof_preds']
    report = get_perf_report(y_true, y_score)
    plot_pr_curve(y_true, y_score, filename="validation_results/metrics_results/l1_xgb_oof_pr_curve.png")

Note: For calibration analysis including ECE (Expected Calibration Error), confidence intervals 
and group-wise analysis, use calibration_analysis.py module instead.

For advanced PSI analysis with detailed debugging and drift analysis, use stability.py module.

Metrics explained:
- AUC: Area Under ROC Curve (ranking quality, 0.5 = random, 1.0 = perfect)
- Gini: 2*AUC-1 (alternative to AUC, 0 = random, 1 = perfect)
- KS: Kolmogorov–Smirnov statistic (max separation between positive/negative CDFs)
- Brier: Brier score (mean squared error for probabilities, lower = better)
- HL_chi2, HL_p_value: Hosmer-Lemeshow goodness-of-fit test (calibration)
- F1_Score: Harmonic mean of precision and recall at 0.5 cutoff
- Precision, Recall: At 0.5 cutoff
- PR_AUC: Area under Precision-Recall curve
- Optimal_Cutoff, Max_F1: Threshold and F1 maximizing F1 score

Note: For Population Stability Index (PSI) and drift analysis, use stability.py module.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_fscore_support, auc, precision_recall_curve, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import chi2
import warnings
import os
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from pathlib import Path

__all__ = [
    "auc_gini",
    "ks_stat",
    "brier_score",
    "hosmer_lemeshow",
    "get_perf_report",
    "find_optimal_cutoff",
    "plot_pr_curve",
    "plot_roc_curve",
    "plot_probability_histogram", 
    "plot_confusion_matrix"
]

def _prepare_series(y_true: pd.Series, y_score: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure numpy array inputs and drop NaNs consistently."""
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)
    mask = y_true.notna() & y_score.notna()
    return y_true[mask].to_numpy(), y_score[mask].to_numpy()

def auc_gini(y_true: pd.Series, y_score: pd.Series) -> Tuple[float, float]:
    """Return AUC and Gini (2*AUC-1)."""
    y, p = _prepare_series(y_true, y_score)
    
    # Handle single class case
    if len(np.unique(y)) == 1:
        return 0.5, 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_val = roc_auc_score(y, p)
    gini = 2 * auc_val - 1
    return auc_val, gini

def ks_stat(y_true: pd.Series, y_score: pd.Series, n_bins: int = 10) -> float:
    """Kolmogorov–Smirnov statistic for binary classification."""
    y, p = _prepare_series(y_true, y_score)
    
    # Use numpy for faster quantile calculation
    percentiles = np.linspace(0, 100, n_bins + 1)
    cuts = np.percentile(p, percentiles)
    cuts = np.unique(cuts)  # Ensure unique bins
    
    if len(cuts) < 2:
        return 0.0  # Not enough unique values
    
    # Create bins using numpy
    bucket = np.digitize(p, bins=cuts[1:-1], right=True)
    data = pd.DataFrame({"y": y, "bucket": bucket})
    
    grouped = data.groupby("bucket", observed=True)
    stats = grouped.agg(total=("y", "count"), bad=("y", "sum"))
    stats["good"] = stats["total"] - stats["bad"]
    
    if stats["bad"].sum() == 0 or stats["good"].sum() == 0:
        return 0.0
    
    stats = stats.sort_index(ascending=False)
    stats["cum_bad_pct"] = stats["bad"].cumsum() / stats["bad"].sum()
    stats["cum_good_pct"] = stats["good"].cumsum() / stats["good"].sum()
    
    ks = (stats["cum_bad_pct"] - stats["cum_good_pct"]).abs().max()
    return float(ks)

def brier_score(y_true: pd.Series, y_score: pd.Series) -> float:
    """Brier score loss (mean squared error for probabilities)."""
    y, p = _prepare_series(y_true, y_score)
    return float(brier_score_loss(y, p))

def hosmer_lemeshow(y_true: pd.Series, y_score: pd.Series, n_groups: int = 10) -> Tuple[float, float]:
    """Hosmer–Lemeshow goodness‑of‑fit test."""
    y, p = _prepare_series(y_true, y_score)
    
    # Create bins using numpy for efficiency
    percentiles = np.linspace(0, 100, n_groups + 1)
    cuts = np.percentile(p, percentiles)
    cuts = np.unique(cuts)
    
    if len(cuts) < 2:
        return 0.0, 1.0  # Not enough bins
    
    bucket = np.digitize(p, bins=cuts[1:-1], right=True)
    df = pd.DataFrame({"y": y, "p": p, "bucket": bucket})
    
    grouped = df.groupby("bucket", observed=True)
    obs = grouped["y"].sum().to_numpy()
    total = grouped.size().to_numpy()
    exp = grouped["p"].mean().to_numpy() * total
    
    # Avoid division by zero
    epsilon = 1e-10
    denominator = exp * (1 - np.clip(exp / (total + epsilon), 0, 1)) + epsilon
    chi_sq = np.nansum((obs - exp) ** 2 / denominator)
    
    dof = max(0, n_groups - 2)  # Ensure non-negative degrees of freedom
    p_value = 1 - chi2.cdf(chi_sq, dof) if dof > 0 else 1.0
    
    return float(chi_sq), float(p_value)

def find_optimal_cutoff(y_true: pd.Series, y_score: pd.Series, n_thresholds: int = 100) -> Tuple[float, float]:
    """Find optimal threshold and max F1 score."""
    y, p = _prepare_series(y_true, y_score)
    if len(y) == 0 or len(np.unique(y)) != 2:
        return 0.5, 0.0
    
    # Vectorized implementation for better performance
    thresholds = np.linspace(0, 1, n_thresholds)
    preds = p[:, None] > thresholds
    f1_scores = np.array([
        f1_score(y, preds[:, i], zero_division=0) for i in range(n_thresholds)
    ])
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def plot_pr_curve(y_true: pd.Series, y_score: pd.Series, filename: str = 'pr_curve.png'):
    """Plot and save Precision-Recall curve."""
    y, p = _prepare_series(y_true, y_score)
    if len(y) == 0:
        return
    
    precision, recall, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(y_true, y_score, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_probability_histogram(y_score, filename="probability_histogram.png", bins=20):
    plt.figure(figsize=(8, 6))
    plt.hist(y_score, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_true, y_score, threshold=None, filename="confusion_matrix.png"):
    """Plot confusion matrix with automatic threshold selection if needed."""
    # Prepare data
    y, p = _prepare_series(y_true, y_score)
    
    # Auto-select optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_cutoff(y, p)
        print(f"[INFO] Using optimal threshold: {threshold:.3f}")
    
    # Debug info
    print(f"[DEBUG] y_score range: [{p.min():.6f}, {p.max():.6f}]")
    print(f"[DEBUG] Threshold: {threshold}")
    print(f"[DEBUG] Number of predictions >= threshold: {(p >= threshold).sum()}")
    print(f"[DEBUG] Number of predictions < threshold: {(p < threshold).sum()}")
    
    y_pred = (p >= threshold).astype(int)
    
    # More debug info
    print(f"[DEBUG] y_true distribution: {np.bincount(y)}")
    print(f"[DEBUG] y_pred distribution: {np.bincount(y_pred)}")
    
    # Check if all predictions are same class
    if len(np.unique(y_pred)) == 1:
        print(f"[WARNING] All predictions are class {y_pred[0]}! Consider adjusting threshold.")
    
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.savefig(filename)
    plt.close()
    
    return cm

def get_perf_report(
    y_true: pd.Series,
    y_score: pd.Series,
    n_bins: int = 10,
    n_groups_hl: int = 10,
    n_thresholds: int = 100
) -> Dict[str, float]:
    """Return comprehensive performance report for binary classifier."""
    y, p = _prepare_series(y_true, y_score)
    if len(y) == 0:
        return {
            "AUC": 0.5,
            "Gini": 0.0,
            "KS": 0.0,
            "Brier": 0.25,
            "HL_chi2": 0.0,
            "HL_p_value": 1.0,
            "F1_Score": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "PR_AUC": 0.0,
            "Optimal_Cutoff": 0.5,
            "Max_F1": 0.0
        }
    
    # Calculate basic metrics
    auc_val, gini = auc_gini(y, p)
    ks = ks_stat(y, p, n_bins=n_bins)
    brier = brier_score(y, p)
    hl_chi, hl_p = hosmer_lemeshow(y, p, n_groups=n_groups_hl)
    
    # Calculate precision-recall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, p > 0.5, average='binary', zero_division=0
    )
    precisions, recalls, _ = precision_recall_curve(y, p)
    pr_auc = auc(recalls, precisions)
    
    # Find optimal cutoff
    opt_thresh, max_f1 = find_optimal_cutoff(y, p, n_thresholds)
    
    return {
        "AUC": round(auc_val, 5),
        "Gini": round(gini, 5),
        "KS": round(ks, 5),
        "Brier": round(brier, 5),
        "HL_chi2": round(hl_chi, 3),
        "HL_p_value": round(hl_p, 3),
        "F1_Score": round(f1, 5),
        "Precision": round(precision, 5),
        "Recall": round(recall, 5),
        "PR_AUC": round(pr_auc, 5),
        "Optimal_Cutoff": round(opt_thresh, 3),
        "Max_F1": round(max_f1, 5)
    }
    

# For quick test
if __name__ == "__main__":
    y_true = pd.read_csv("data/raw/application_train.csv")["TARGET"]
    y_score = pd.read_csv("models/l3_stacking/l3_extratree_oof_predictions.csv")["oof_preds"]

    auc_val, gini = auc_gini(y_true, y_score)
    ks = ks_stat(y_true, y_score)
    brier = brier_score(y_true, y_score)
    # For PSI and drift analysis, use stability.py module
    report = get_perf_report(y_true, y_score)

    print(f"AUC: {auc_val:.4f}, Gini: {gini:.4f}, KS: {ks:.4f}, Brier: {brier:.4f}")
    print("Main metrics:", report)

    Path("validation_results/metrics_results").mkdir(parents=True, exist_ok=True)
    plot_pr_curve(y_true, y_score, filename="validation_results/metrics_results/l3_extratree_oof_pr_curve.png")
    plot_roc_curve(y_true, y_score, filename="validation_results/metrics_results/l3_extratree_oof_roc_curve.png")
    plot_probability_histogram(y_score, filename="validation_results/metrics_results/l3_extratree_oof_histogram.png")
    
    # Plot confusion matrix with optimal threshold
    print("\n=== Confusion Matrix with Optimal Threshold ===")
    cm_optimal = plot_confusion_matrix(y_true, y_score, threshold=None, filename="validation_results/metrics_results/l3_extratree_oof_confusion_matrix_optimal.png")
    
    print("All plots saved.")

    with open("validation_results/metrics_results/l3_extratree_oof_metrics.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for k, v in report.items():
            writer.writerow([k, v])