"""
calibration_analysis.py

This module provides advanced tools for calibration analysis and visualization of binary classification models in credit risk modeling or similar machine learning tasks.

Workflow:
- Supports plotting calibration curves for a single model, group-wise analysis, and multi-model comparison.
- Computes key calibration metrics: Brier score, ROC AUC, Expected Calibration Error (ECE), Maximum Calibration Error (MCE).
- Allows saving plots and metrics for reporting and further analysis.
- Handles group-based calibration analysis (e.g., by customer segment or feature value).
- Designed to work with OOF (out-of-fold) prediction files and ground truth label files, as typically produced in ML pipelines.

Main Functions:
- plot_calibration_curve: Plots a calibration curve for a single model, with options for confidence intervals and histogram.
- run_calibration_analysis: Runs calibration analysis for a model, optionally by group, and saves plots/metrics.
- plot_multiple_calibration_curves: Plots calibration curves for multiple models on the same chart for comparison.

Inputs:
- OOF prediction CSV file (with column 'oof_preds') for L3 stacking (in models/l3_stacking/l3_extratree_oof_predictions.csv).
- Target/label CSV file (with column 'TARGET') (in data/raw/).

Outputs:
- Calibration plots (PNG) and metrics (dict or DataFrame) in validation_results/calibration_results/.

Example usage:
    import pandas as pd
    y_true = pd.read_csv("data/raw/application_train.csv")["TARGET"].values
    y_pred_l3 = pd.read_csv("models/l3_stacking/l3_extratree_oof_predictions.csv")["oof_preds"].values
    models = {
        "L3_ExtraTree": (y_true, y_pred_l3)
    }
    plot_multiple_calibration_curves(
        results=models,
        save_path="validation_results/calibration_results/l3_model_comparison.png"
    )

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import seaborn as sns
from scipy.stats import binomtest
import os
import re
import time
from tqdm import tqdm

# === CONFIG: Just change the group variable name here ===
GROUP_COL = "NAME_EDUCATION_TYPE"  # Change the group variable name here

def sanitize_filename(s):
    import re
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(s))

GROUP_COL_FRIENDLY = sanitize_filename(GROUP_COL)

def plot_calibration_curve(
    y_true, 
    y_pred_proba, 
    n_bins=10, 
    model_name="Model",
    strategy='quantile',
    show_histogram=True,
    confidence_intervals=True,
    save_path=None,
    show_plot=True
):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true and y_pred_proba must have the same length")
    
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy=strategy
    )
    
    brier = brier_score_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    bin_counts = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_pred_proba)
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)
    
    mce = np.max(np.abs(prob_true - prob_pred))
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(prob_pred, prob_true, 's-', label=f'{model_name}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Ideal')
    
    if confidence_intervals:
        for i in range(len(prob_true)):
            n = bin_counts[i]
            p = prob_true[i]
            if n > 0:
                ci = binomtest(int(p * n), n).proportion_ci(confidence_level=0.95)
                lower = ci.low
                upper = ci.high
                ax1.vlines(prob_pred[i], lower, upper, color='red', alpha=0.5)
    
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Actual positive rate')
    ax1.set_title(f'Calibration Curve\n{model_name} (Brier: {brier:.4f}, AUC: {auc:.4f}, ECE: {ece:.4f}, MCE: {mce:.4f})')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    if show_histogram:
        ax2 = fig.add_subplot(gs[1])
        ax2.hist(y_pred_proba, bins=n_bins, range=(0, 1), 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Sample count')
        ax2.set_title('Distribution of predicted probabilities')
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Calibration plot saved at: {save_path}")
    
    if show_plot:
        plt.show()
    
    return {
        'brier_score': brier,
        'roc_auc': auc,
        'ece': ece,
        'mce': mce,
        'calibration_curve': (prob_true, prob_pred)
    }

def run_calibration_analysis(
    oof_path: str, 
    target_path: str, 
    model_name="Model",
    group_col=None,
    group_values=None,
    save_dir=None,
    show_plot=True
):
    pred_df = pd.read_csv(oof_path)
    target_df = pd.read_csv(target_path)
    
    if "oof_preds" not in pred_df.columns:
        raise ValueError("Column 'oof_preds' not found in prediction file.")
    if "TARGET" not in target_df.columns:
        raise ValueError("Column 'TARGET' not found in target file.")

    if group_col:
        if group_col not in target_df.columns:
            raise ValueError(f"Column {group_col} not found in target file")
        merged_df = pred_df.merge(target_df, left_index=True, right_index=True)
    else:
        y_pred = pred_df["oof_preds"]
        y_true = target_df["TARGET"]
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n{'='*50}")
    print(f"Calibration analysis for {model_name}")
    print(f"{'='*50}")
    
    if not group_col:
        save_path = os.path.join(save_dir, f"calibration_{model_name}.png") if save_dir else None
        metrics = plot_calibration_curve(
            y_true, y_pred, 
            model_name=model_name,
            save_path=save_path,
            show_plot=show_plot
        )
        metrics_df = pd.DataFrame([metrics])
        return metrics_df
    
    group_metrics = {}
    
    if group_values:
        groups = group_values
    else:
        groups = merged_df[group_col].unique()
    
    groups = list(groups)
    total_groups = len(groups)
    for idx, group_val in enumerate(tqdm(groups, desc=f"Processing groups ({group_col})"), 1):
        group_df = merged_df[merged_df[group_col] == group_val]
        if len(group_df) == 0:
            continue

        print(f"\n[{idx}/{total_groups}] Processing group {group_col} = {group_val} ({len(group_df)} samples)")
        start_time = time.time()

        y_true_group = group_df["TARGET"]
        y_pred_group = group_df["oof_preds"]

        safe_group_val = sanitize_filename(group_val)
        save_path = os.path.join(
            save_dir, f"calibration_{model_name}_{group_col}_{safe_group_val}.png"
        ) if save_dir else None
        metrics = plot_calibration_curve(
            y_true_group, y_pred_group, 
            model_name=f"{model_name} - {group_col}={group_val}",
            save_path=save_path,
            show_plot=show_plot
        )
        group_metrics[group_val] = metrics

        elapsed = time.time() - start_time
        print(f"    Done group {group_col} = {group_val} in {elapsed:.1f} seconds.")
    
    return group_metrics

def plot_multiple_calibration_curves(
    results: dict, 
    save_path=None,
    show_histogram=False,
    show_plot=True
):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (model_name, (y_true, y_pred_proba)) in enumerate(results.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy='quantile')
        plt.plot(prob_pred, prob_true, 'o-', color=colors[i], label=model_name)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Actual positive rate')
    plt.title('Comparison of Calibration Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Comparison plot saved at: {save_path}")
    
    if show_plot:
        plt.show()

# Quick test

if __name__ == "__main__":
    # Overall calibration
    metrics_df = run_calibration_analysis(
        oof_path="models/l3_stacking/l3_extratree_oof_predictions.csv",
        target_path="data/raw/application_train.csv",
        model_name="L3_ExtraTree",
        save_dir="validation_results/calibration_results",
        show_plot=False
    )
    metrics_df.to_csv(f"validation_results/calibration_results/l3_extratree_metrics.csv", index=False)

    # Grouped calibration (automatically by GROUP_COL)
    group_metrics = run_calibration_analysis(
        oof_path="models/l3_stacking/l3_extratree_oof_predictions.csv",
        target_path="data/raw/application_train.csv",
        model_name="L3_ExtraTree",
        group_col=GROUP_COL,
        save_dir="validation_results/calibration_results",
        show_plot=False
    )

    group_metrics_df = pd.DataFrame.from_dict(group_metrics, orient="index")
    group_metrics_df.index.name = "group_value"
    group_metrics_df.to_csv(f"validation_results/calibration_results/l3_extratree_metrics_by_{GROUP_COL_FRIENDLY}.csv")

    # Plot calibration curve for L3
    y_true = pd.read_csv("data/raw/application_train.csv")['TARGET'].values
    y_pred_l3 = pd.read_csv("models/l3_stacking/l3_extratree_oof_predictions.csv")['oof_preds'].values
    models = {
        "L3_ExtraTree": (y_true, y_pred_l3)
    }
    plot_multiple_calibration_curves(
        results=models,
        save_path=f"validation_results/calibration_results/l3_model_comparison.png",
        show_plot=False
    )