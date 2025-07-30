"""
Data Drift Analysis Module
=========================

This module provides functions for feature drift analysis, population stability index (PSI), and batch stability overview for credit risk modeling or similar machine learning tasks.

Main features:
- Calculate PSI between two distributions (e.g., train/test, time splits)
- Compute drift metrics for numerical and categorical features (KS, Chi2, JS, Wasserstein, mean diff, ...)
- Summarize batch stability and drift severity
- Plot PSI histograms, feature drift (numerical/categorical), and top drifted features
- Designed for use in model validation, monitoring, and data quality checks

How to use:
-----------
- Place this file in your project (e.g., src/validation/).
- Run the file directly from the project root or the module path:
      python -m src.validation.stability
- The script will automatically:
    * Load train and test data (from data/raw/application_train.csv and data/raw/application_test.csv)
    * Calculate PSI for model score (if available)
    * Analyze feature drift between train and test
    * Summarize drift results and save to CSV
    * Plot and save drift visualizations for top features
    * Save all results to validation_results/stability_results/

Notes:
------
- If the test set does not have labels (TARGET column), only feature drift will be checked (no target or model performance drift).
- This script is intended for batch drift analysis between two datasets. For time series drift monitoring, see advanced modules or extend this script.

Output:
-------
- CSV files with drift summary and feature drift details
- PNG images with drift plots for top features and PSI histogram
- All outputs are saved in validation_results/stability_results/

"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional
import os

# 1. Population Stability Index

def calculate_psi(expected, actual, buckets=10, epsilon=1e-6, return_details=False):
    """Calculate PSI between two distributions. Optionally return bin details for debugging."""
    # Use quantile-based binning
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicate values

    if len(breakpoints) < 2:
        # Not enough bins to calculate PSI
        if return_details:
            return 0.0, breakpoints, None, None
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Replace 0 with epsilon to avoid log(0)
    expected_percents = np.clip(expected_percents, epsilon, None)
    actual_percents = np.clip(actual_percents, epsilon, None)
    
    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_score = np.sum(psi_values)

    if return_details:
        # Return bin details for debugging
        return psi_score, breakpoints, expected_percents, actual_percents
    return psi_score

# 2. Feature Drift Metrics

def calculate_feature_drift(train_df, test_df, numerical_cols=[], categorical_cols=[], sample_size=10000, max_unique_cats=100, verbose=False):
    """Calculate drift metrics for features between two datasets.
    Returns drift report and log of checked/skipped features.
    """
    # Sample for large datasets
    if len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42)
    if len(test_df) > sample_size:
        test_df = test_df.sample(sample_size, random_state=42)
    
    drift_report = {}
    checked_features = 0
    skipped_features = 0
    skipped_features_list = []
    
    # Auto-detect feature types if not specified
    if not numerical_cols and not categorical_cols:
        numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = train_df.select_dtypes(exclude=np.number).columns.tolist()

    for col in numerical_cols:
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()
        
        # Skip if not enough data
        if len(train_col) < 10 or len(test_col) < 10:
            skipped_features += 1
            skipped_features_list.append(col)
            continue
            
        ks_stat_val, ks_pval = ks_2samp(train_col, test_col)
        mean_diff_pct = abs(train_col.mean() - test_col.mean()) / (abs(train_col.mean()) + 1e-6)
        w_dist = wasserstein_distance(train_col, test_col)
        
        drift_report[col] = {
            'type': 'numerical',
            'train_mean': train_col.mean(),
            'test_mean': test_col.mean(),
            'mean_diff_pct': mean_diff_pct,
            'KS_stat': ks_stat_val,
            'KS_pvalue': ks_pval,
            'Wasserstein_dist': w_dist
        }
        checked_features += 1

    for col in categorical_cols:
        # Get all categories
        all_categories = set(train_df[col].dropna().unique()) | set(test_df[col].dropna().unique())
        
        # Skip feature with too many unique values
        if len(all_categories) > max_unique_cats:
            skipped_features += 1
            skipped_features_list.append(col)
            if verbose:
                print(f"[Drift] Skip feature '{col}' (number of unique values = {len(all_categories)})")
            continue
        
        # Create contingency table
        train_counts = train_df[col].value_counts().reindex(all_categories, fill_value=0)
        test_counts = test_df[col].value_counts().reindex(all_categories, fill_value=0)
        
        # Skip if not enough data
        if train_counts.sum() == 0 or test_counts.sum() == 0:
            skipped_features += 1
            skipped_features_list.append(col)
            continue
            
        contingency_table = pd.DataFrame({'train': train_counts, 'test': test_counts}).T
        
        try:
            chi2_stat, chi2_pval, _, _ = chi2_contingency(contingency_table)
        except ValueError:
            chi2_stat, chi2_pval = np.nan, np.nan
        
        # Calculate JS Divergence
        train_dist = train_counts / train_counts.sum()
        test_dist = test_counts / test_counts.sum()
        try:
            js_div = jensenshannon(train_dist, test_dist, base=2)
        except Exception:
            js_div = np.nan
        
        drift_report[col] = {
            'type': 'categorical',
            'train_dist': train_dist.to_dict(),
            'test_dist': test_dist.to_dict(),
            'Chi2_stat': chi2_stat,
            'Chi2_pvalue': chi2_pval,
            'JS_divergence': js_div
        }
        checked_features += 1
    
    # Return log of checked/skipped features
    return {
        'drift_report': drift_report,
        'checked_features': checked_features,
        'skipped_features': skipped_features,
        'skipped_features_list': skipped_features_list
    }

# 3. Batch Stability Overview

def stability_summary(score_train, score_test, train_df, test_df, numerical_cols=[], categorical_cols=[]):
    """Summarize batch stability and feature drift between two datasets."""
    psi_score = calculate_psi(score_train, score_test)
    feature_drift = calculate_feature_drift(train_df, test_df, numerical_cols, categorical_cols)
    
    total_drift_score = 0
    drift_flags = []
    
    for col, metrics in feature_drift.items():
        if not isinstance(metrics, dict) or 'type' not in metrics:
            # Skip invalid feature
            continue
        if metrics['type'] == 'numerical' and metrics.get('KS_pvalue', 1) < 0.05:
            total_drift_score += 1
            drift_flags.append(f"{col} (KS: {metrics.get('KS_stat', 0):.3f})")
        elif metrics['type'] == 'categorical' and metrics.get('Chi2_pvalue', 1) < 0.05:
            total_drift_score += 1
            drift_flags.append(f"{col} (Chi2: {metrics.get('Chi2_stat', 0):.3f})")
    
    severity = "Low"
    n_features = len(feature_drift)
    drift_score_pct = round(total_drift_score / n_features * 100, 1) if n_features > 0 else 0.0

    if n_features > 0:
        if total_drift_score > n_features * 0.3 or psi_score > 0.25:
            severity = "High"
        elif total_drift_score > n_features * 0.1 or psi_score > 0.1:
            severity = "Medium"
    else:
        severity = "N/A"

    return {
        'Score_PSI': round(psi_score, 4),
        'PSI_Severity': severity,
        'Total_Drift_Features': total_drift_score,
        'Drift_Flag_Features': drift_flags,
        'Drift_Score_Percentage': drift_score_pct,
        'Feature_Drift': feature_drift
    }
    
def plot_psi_histogram(expected, actual, breakpoints=None, bins=10, title=None):
    """Plot histogram comparing expected/actual distributions for PSI."""
    if breakpoints is None:
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints)
    plt.figure(figsize=(8, 4))
    plt.hist(expected, bins=breakpoints, alpha=0.5, label='Expected', color='blue', density=True)
    plt.hist(actual, bins=breakpoints, alpha=0.5, label='Actual', color='orange', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(title or 'PSI Distribution Comparison')
    plt.legend()
    plt.tight_layout()
    # plt.show()  # Đã tắt popup

def plot_categorical_drift(train_counts, test_counts, feature_name, top_n=20):
    """Plot bar chart comparing categorical distributions between train/test."""
    # Only take top_n most frequent categories
    all_cats = set(train_counts.keys()) | set(test_counts.keys())
    cats_sorted = sorted(all_cats, key=lambda x: train_counts.get(x, 0) + test_counts.get(x, 0), reverse=True)[:top_n]
    train_vals = [train_counts.get(cat, 0) for cat in cats_sorted]
    test_vals = [test_counts.get(cat, 0) for cat in cats_sorted]
    x = np.arange(len(cats_sorted))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, train_vals, width, label='Train')
    plt.bar(x + width/2, test_vals, width, label='Test')
    plt.xticks(x, cats_sorted, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title(f'Categorical Drift: {feature_name}')
    plt.legend()
    plt.tight_layout()
    # plt.show()  # Đã tắt popup

def plot_numerical_drift(train_col, test_col, feature_name, bins=30):
    """Plot histogram comparing numerical distributions between train/test."""
    plt.figure(figsize=(8, 4))
    sns.histplot(train_col, bins=bins, color='blue', label='Train', stat='density', kde=True, alpha=0.5)
    sns.histplot(test_col, bins=bins, color='orange', label='Test', stat='density', kde=True, alpha=0.5)
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.title(f'Numerical Drift: {feature_name}')
    plt.legend()
    plt.tight_layout()
    # plt.show()  # Đã tắt popup
    
def plot_top_feature_drifts(train_df, test_df, drift_report, top_n=3):
    """Automatically plot the top N most drifted numerical and categorical features."""
    # Classify features
    numerical_feats = []
    categorical_feats = []
    for feat, metrics in drift_report.items():
        if metrics['type'] == 'numerical':
            numerical_feats.append((feat, abs(metrics.get('KS_stat', 0))))
        elif metrics['type'] == 'categorical':
            js = metrics.get('JS_divergence', 0)
            if js is not None and not np.isnan(js):
                categorical_feats.append((feat, js))
    # Select top N
    top_num = sorted(numerical_feats, key=lambda x: -x[1])[:top_n]
    top_cat = sorted(categorical_feats, key=lambda x: -x[1])[:top_n]
    # Plot numerical
    for feat, ks in top_num:
        print(f"Numerical drift: {feat} (KS={ks:.3f})")
        plot_numerical_drift(train_df[feat].dropna(), test_df[feat].dropna(), feat)

    # Plot categorical
    for feat, js in top_cat:
        print(f"Categorical drift: {feat} (JS={js:.3f})")
        train_counts = train_df[feat].value_counts().to_dict()
        test_counts = test_df[feat].value_counts().to_dict()
        plot_categorical_drift(train_counts, test_counts, feat)


if __name__ == "__main__":

    # 1. Tạo thư mục lưu kết quả
    os.makedirs("validation_results/stability_results", exist_ok=True)

    # 2. Load dữ liệu
    y_train = pd.read_csv("data/raw/application_train.csv")
    y_test = pd.read_csv("data/raw/application_test.csv") if os.path.exists("data/raw/application_test.csv") else None
    score_train = pd.read_csv("models/l3_stacking/l3_extratree_oof_predictions.csv")['oof_preds']
    score_test = pd.read_csv("models/l3_stacking/l3_extratree_test_predictions.csv")['test_preds'] if y_test is not None else None

    # 3. Xác định các cột chung
    feature_cols = [col for col in y_train.columns if y_test is not None and col in y_test.columns] if y_test is not None else y_train.columns.tolist()
    numerical_cols = y_train[feature_cols].select_dtypes(include='number').columns.tolist()
    categorical_cols = y_train[feature_cols].select_dtypes(exclude='number').columns.tolist()

    # 4. Phân tích drift cho feature nếu có test set
    if y_test is not None:
        print("\n=== 1. Calculate PSI for model score ===")
        psi_score = calculate_psi(score_train, score_test)
        print(f"PSI (score): {psi_score:.4f}")
        plot_psi_histogram(score_train, score_test, title="PSI Histogram for Model Score")
        plt.savefig("validation_results/stability_results/psi_histogram_score.png"); plt.close()

        print("\n=== 2. Calculate feature drift ===")
        drift_result = calculate_feature_drift(y_train, y_test, numerical_cols, categorical_cols)
        drift_report = drift_result['drift_report']
        print(f"Checked features: {drift_result['checked_features']}, Skipped: {drift_result['skipped_features']}")

        print("\n=== 3. Stability summary ===")
        summary = stability_summary(score_train, score_test, y_train, y_test, numerical_cols, categorical_cols)
        summary_simple = summary.copy()
        feature_drift = summary_simple.pop('Feature_Drift')
        pd.DataFrame([summary_simple]).to_csv("validation_results/stability_results/l3_stacking_stability_summary.csv", index=False)
        # Save feature drift details
        feature_drift_df = pd.DataFrame.from_dict(feature_drift['drift_report'], orient='index')
        feature_drift_df.reset_index(inplace=True)
        feature_drift_df.rename(columns={'index': 'feature'}, inplace=True)
        feature_drift_df.to_csv("validation_results/stability_results/l3_stacking_feature_drift.csv", index=False)

        print("\n=== 4. Plot drift for top N features ===")
        top_n = 3
        plot_top_feature_drifts(y_train, y_test, drift_report, top_n=top_n)
        # Lưu hình cho top N numerical
        numerical_feats = [(feat, abs(metrics.get('KS_stat', 0))) for feat, metrics in drift_report.items() if metrics['type'] == 'numerical']
        top_num = sorted(numerical_feats, key=lambda x: -x[1])[:top_n]
        for feat, ks in top_num:
            plt.figure()
            plot_numerical_drift(y_train[feat].dropna(), y_test[feat].dropna(), feat)
            plt.savefig(f"validation_results/stability_results/numerical_drift_{feat}.png"); plt.close()
        # Lưu hình cho top N categorical
        categorical_feats = [(feat, metrics.get('JS_divergence', 0)) for feat, metrics in drift_report.items() if metrics['type'] == 'categorical' and metrics.get('JS_divergence', 0) is not None]
        top_cat = sorted(categorical_feats, key=lambda x: -x[1])[:top_n]
        for feat, js in top_cat:
            plt.figure()
            train_counts = y_train[feat].value_counts().to_dict()
            test_counts = y_test[feat].value_counts().to_dict()
            plot_categorical_drift(train_counts, test_counts, feat)
            plt.savefig(f"validation_results/stability_results/categorical_drift_{feat}.png"); plt.close()

        print("\n=== Done! All results saved in validation_results/stability_results ===")
    else:
        print("No application_test.csv found. Only feature drift in train set can be checked.")