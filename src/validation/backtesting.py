"""
backtesting.py

Rolling window backtesting for credit risk models with drift analysis, model persistence, and performance visualization.

WARNING: Backtesting is currently DISABLED because the dataset does not contain a suitable time series variable (APP_DATE, derived from DAYS_DECISION in previous_application) for rolling window splits.
To enable backtesting, your data must include a valid APP_DATE column, created by merging DAYS_DECISION from previous_application into your application data.

Features:
- Rolling splits with customizable window/horizon
- LightGBM training with warm start
- Drift analysis and performance metrics
- Artifact saving (model, metrics, drift, plots, predictions, config)
- CLI for flexible experiment control

Inputs:
- Application train data (with TARGET, APP_DATE, features)
- CLI arguments: window, horizon, output dir, features, seed

Outputs:
- Model files, metrics, drift reports, prediction files, plots, config

Example usage:
    # Merge DAYS_DECISION from previous_application to create APP_DATE
    prev_app = pd.read_csv('previous_application.csv')
    app = pd.read_csv('application_train.csv')
    prev_dates = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].max().reset_index()
    app = app.merge(prev_dates, on='SK_ID_CURR', how='left')
    SNAPSHOT = pd.Timestamp("2018-01-01")
    app['APP_DATE'] = SNAPSHOT + pd.to_timedelta(app['DAYS_DECISION'], unit='D')
    # Now run backtesting
    python backtesting.py --window 18 --horizon 3 --out output/backtest --features ... --seed 42
"""
from __future__ import annotations
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import logging
from tqdm import tqdm
from src.validation.performance_metrics import get_perf_report  # type: ignore
from src.validation.stability import stability_summary  # type: ignore
from pd_modeling_project.src.utils import utils
from src.data_pipeline.loaders import load_application_train  # type: ignore
from src.data_pipeline.processor import DataProcessor
from src.processing.encoding import TargetEncoder
from src.processing.imputation import SimpleImputer
from src.modeling.feature_selector import FeatureSelector

import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import glob

SNAPSHOT = pd.Timestamp("2018-01-01")
DEFAULT_FEATURES = pd.read_csv("output/src_results/modeling_results/top_shap_features.csv")["feature"].tolist()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# INSTRUCTION: To use backtesting, merge a time-based column (e.g., APP_DATE) into your application data.
# Example for merging DAYS_DECISION from previous_application:
#
# prev_app = pd.read_csv('previous_application.csv')
# app = pd.read_csv('application_train.csv')
# prev_dates = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].max().reset_index()  # or min(), depending on your goal
# app = app.merge(prev_dates, on='SK_ID_CURR', how='left')
# app['APP_DATE'] = SNAPSHOT + pd.to_timedelta(app['DAYS_DECISION'], unit='D')
#
# After merging, the pipeline will use app['APP_DATE'] for rolling window splits.

def _build_app_date(df: pd.DataFrame) -> pd.Series:
    """Return application date using DAYS_DECISION from previous_application"""
    if "DAYS_DECISION" not in df.columns:
        raise KeyError("DAYS_DECISION column not found. Please merge DAYS_DECISION from previous_application into your application data.")
    if df["DAYS_DECISION"].isnull().any():
        raise ValueError("Missing values in DAYS_DECISION column.")
    return SNAPSHOT + pd.to_timedelta(df["DAYS_DECISION"], unit="D")

def _generate_rolling_splits(
    dates: pd.Series,
    window_train_months: int = 18,
    test_horizon_months: int = 3,
    min_obs: int = 5000,
):
    """Yield (train_idx, test_idx, as_of_date) for each rolling split."""
    df_dates = dates.sort_values()
    min_date = df_dates.min().to_period("M").to_timestamp()
    max_date = df_dates.max().to_period("M").to_timestamp()

    current_end = min_date + pd.DateOffset(months=window_train_months)
    while current_end + pd.DateOffset(months=test_horizon_months) <= max_date:
        train_start = current_end - pd.DateOffset(months=window_train_months)
        test_end = current_end + pd.DateOffset(months=test_horizon_months)

        train_idx = dates[(dates >= train_start) & (dates < current_end)].index
        test_idx = dates[(dates >= current_end) & (dates < test_end)].index

        if len(test_idx) >= min_obs:
            yield train_idx, test_idx, current_end
        current_end += pd.DateOffset(months=test_horizon_months)


# --- Stacking helpers ---
def train_l1_stacking(X_train, y_train, X_test, seed=42, n_splits=5):
    base_models = [
        ("xgb", xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1, random_state=seed, n_jobs=-1, use_label_encoder=False, verbosity=0)),
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)),
        ("catboost", CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6, l2_leaf_reg=3.0, random_seed=seed, loss_function='Logloss', eval_metric='AUC', verbose=False))
    ]
    oof_preds = {}
    test_preds = {}
    for name, model in base_models:
        oof = np.zeros(len(X_train))
        test_pred = np.zeros(len(X_test))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits
        oof_preds[name] = oof
        test_preds[name] = test_pred
    return oof_preds, test_preds

def train_l2_stacking(y_train, oof_preds_l1, test_preds_l1, seed=42, n_splits=5):
    meta_models = [
        ("extratree", ExtraTreesClassifier(max_depth=4, min_samples_leaf=1000, random_state=seed)),
        ("logistic", LogisticRegression(max_iter=2000, solver='liblinear'))
    ]
    l1_names = list(oof_preds_l1.keys())
    X_l2 = np.column_stack([oof_preds_l1[name] for name in l1_names])
    X_test_l2 = np.column_stack([test_preds_l1[name] for name in l1_names])
    oof_preds = {}
    test_preds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for name, model in meta_models:
        oof = np.zeros(len(X_l2))
        test_pred = np.zeros(len(X_test_l2))
        for train_idx, val_idx in skf.split(X_l2, y_train):
            X_tr, X_val = X_l2[train_idx], X_l2[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test_pred += model.predict_proba(X_test_l2)[:, 1] / n_splits
        oof_preds[name] = oof
        test_preds[name] = test_pred
    return oof_preds, test_preds

def train_l3_stacking(y_train, test_df, oof_preds_l2, test_preds_l2, X_train_selected, X_test_selected, raw_feature_names=None, seed=42, n_splits=5):
    l2_names = list(oof_preds_l2.keys())
    X_l3 = np.column_stack([oof_preds_l2[name] for name in l2_names])
    X_test_l3 = np.column_stack([test_preds_l2[name] for name in l2_names])
    # Add original features if available
    if raw_feature_names is not None and len(raw_feature_names) > 0:
        for feat in raw_feature_names:
            X_l3 = np.column_stack([X_l3, X_train_selected[feat].reset_index(drop=True)])
            X_test_l3 = np.column_stack([X_test_l3, X_test_selected[feat].reset_index(drop=True)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_preds_l3 = np.zeros(len(X_test_l3))
    for train_idx, val_idx in skf.split(X_l3, y_train.reset_index(drop=True)):
        X_tr, X_val = X_l3[train_idx], X_l3[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model_l3 = ExtraTreesClassifier(max_depth=4, min_samples_leaf=1000, random_state=seed)
        model_l3.fit(X_tr, y_tr)
        test_preds_l3 += model_l3.predict_proba(X_test_l3)[:, 1] / n_splits
    return test_preds_l3

def run_backtesting(
    window_train_months: int = 18,
    test_horizon_months: int = 3,
    output_dir: str | Path = "output/backtesting",
    feature_set: list = None,
    verbose: bool = True,
    seed: int = 42,
):
    """
    WARNING: This function is currently DISABLED because the dataset does not contain a suitable time series variable (e.g., APP_DATE) for rolling window splits.
    To enable backtesting, ensure your data has a valid time-based column (e.g., APP_DATE) merged from previous_application or similar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    # Step 1: Load and process data using the full pipeline
    processor = DataProcessor(debug=False, seed=seed, force_reload=False)
    app = processor.load_data()
    app = processor.auto_feature_engineering(app)
    app = processor.handle_missing_values(app)

    # Step 2: Encoding categorical features
    categorical_cols = app.select_dtypes(include=["object", "category"]).columns.tolist()
    target = "TARGET"
    train_df = app[app[target].notnull()].copy()
    test_df = app[app[target].isnull()].copy()
    encoder = TargetEncoder()
    train_encoded = encoder.fit_transform(train_df, target, categorical_cols)
    test_encoded = encoder.transform(test_df, categorical_cols)

    # Step 3: Feature selection (SHAP only)
    shap_path = Path("output/top_shap_features.csv")
    if shap_path.exists():
        feature_cols = pd.read_csv(shap_path)["feature"].tolist()
    else:
        X_train = train_encoded.drop(columns=[target, "SK_ID_CURR"], errors="ignore")
        y_train = train_encoded[target]
        # Reduce features for SHAP if too many
        if X_train.shape[1] > 100:
            from sklearn.feature_selection import VarianceThreshold
            selector_var = VarianceThreshold()
            selector_var.fit(X_train.fillna(0))
            variances = selector_var.variances_
            top_idx = np.argsort(variances)[-100:]
            X_train = X_train.iloc[:, top_idx]
        selector = FeatureSelector(top_k=50, seed=seed, shap_sample=5000, full_shap=False)
        feature_cols = selector.fit(X_train, y_train)
        pd.Series(feature_cols).to_csv(shap_path, index=False, header=["feature"])
        print(f"Top SHAP features saved to {shap_path}")

    # Step 4: Imputation (fit on train, apply to both train/test)
    X_train = train_encoded[feature_cols]
    y_train = train_encoded[target]
    X_test = test_encoded[feature_cols] if not test_df.empty else None
    imputer = SimpleImputer()
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    if X_test is not None:
        X_test = imputer.transform(X_test)

    # Step 5: Add APP_DATE back to app for rolling window splitting
    # APP_DATE must be created from DAYS_DECISION in previous_application (see instructions at the top of this file)
    if 'APP_DATE' not in app.columns:
        raise ValueError('APP_DATE is missing. Please merge DAYS_DECISION from previous_application and create APP_DATE before running backtesting.')
    app = app.dropna(subset=["TARGET", "APP_DATE"]).reset_index(drop=True)
    # Ensure X_train index matches app
    X_train = pd.DataFrame(X_train, columns=feature_cols, index=train_encoded.index)
    app.loc[train_encoded.index, feature_cols] = X_train.values
    app = app.copy()

    # Step 6: Save config/params
    config = {
        "window_train_months": window_train_months,
        "test_horizon_months": test_horizon_months,
        "output_dir": str(output_dir),
        "feature_set": feature_cols,
        "seed": seed
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Step 7: Initialize result tracking
    results = []
    feature_drift_reports = []
    previous_model = None

    splits = list(_generate_rolling_splits(app["APP_DATE"], window_train_months, test_horizon_months))
    for split_no, (train_idx, test_idx, as_of) in enumerate(tqdm(splits, desc="Backtest Splits"), start=1):
        # Prepare data for this split
        X_train_split = app.loc[train_idx, feature_cols]
        y_train_split = app.loc[train_idx, "TARGET"]
        X_test_split = app.loc[test_idx, feature_cols]
        y_test_split = app.loc[test_idx, "TARGET"]

        # L1-L2-L3 stacking pipeline
        oof_preds_l1, test_preds_l1 = train_l1_stacking(X_train_split, y_train_split, X_test_split, seed=seed, n_splits=5)
        oof_preds_l2, test_preds_l2 = train_l2_stacking(y_train_split, oof_preds_l1, test_preds_l1, seed=seed, n_splits=5)
        stacking_pred = train_l3_stacking(y_train_split, app.loc[test_idx], oof_preds_l2, test_preds_l2, X_train_split, X_test_split, raw_feature_names=None, seed=seed, n_splits=5)
        proba_test = stacking_pred
        proba_train = None
        # --- END STACKING ---

        # Performance report for this split
        perf_report = get_perf_report(y_test_split, proba_test)

        # Drift report for this split
        drift_report = stability_summary(
            score_train=np.zeros(len(y_train_split)),  # placeholder, nếu cần có thể tính stacking train
            score_test=proba_test,
            train_df=app.loc[train_idx],
            test_df=app.loc[test_idx],
            numerical_cols=feature_cols,
            categorical_cols=[]
        )

        results.append({
            "split": split_no,
            "as_of": as_of.date(),
            "train_obs": len(train_idx),
            "test_obs": len(test_idx),
            **perf_report,
            "drift_score": drift_report['Drift_Score_Percentage']
        })

        feature_drift_reports.append({
            "split": split_no,
            "as_of": as_of.date(),
            **drift_report
        })

        # Save predictions and artifacts for this split
        split_pred_df = pd.DataFrame({
            "split": split_no,
            "as_of": as_of.date(),
            "y_true": y_test_split,
            "y_pred": proba_test
        })
        split_pred_df.to_csv(predictions_dir / f"predictions_split_{split_no}.csv", index=False)

        logger.info(f"[Split {split_no}] as_of={as_of.date()}")
        logger.info(f"  AUC: {perf_report['AUC']:.4f}, KS: {perf_report['KS']:.4f}")
        logger.info(f"  Drift: {drift_report['Drift_Score_Percentage']:.1f}%")

    # Save overall results and generate plots
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "backtest_metrics.csv", index=False)

    drift_df = pd.DataFrame(feature_drift_reports)
    drift_df.to_csv(output_dir / "feature_drift.csv", index=False)

    _generate_performance_plots(res_df, drift_df, output_dir)

    return res_df, drift_df

def _generate_performance_plots(results_df: pd.DataFrame, drift_df: pd.DataFrame, output_dir: Path):
    """Generate performance and drift trend visualization"""
    plt.figure(figsize=(18, 6))
    # AUC trend
    plt.subplot(131)
    plt.plot(results_df['as_of'], results_df['AUC'], 'o-', label='AUC')
    plt.xlabel('Test Period')
    plt.ylabel('AUC')
    plt.title('AUC Trend Over Time')
    plt.grid(True)
    # KS trend
    plt.subplot(132)
    plt.plot(results_df['as_of'], results_df['KS'], 's-', color='orange', label='KS')
    plt.xlabel('Test Period')
    plt.ylabel('KS Statistic')
    plt.title('KS Statistic Trend')
    plt.grid(True)
    # Drift trend
    plt.subplot(133)
    plt.plot(results_df['as_of'], results_df['drift_score'], '^-', color='green', label='Drift Score')
    plt.xlabel('Test Period')
    plt.ylabel('Drift Score (%)')
    plt.title('Drift Score Trend')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_trend.png", dpi=300)
    plt.savefig(output_dir / "performance_trend.pdf", dpi=300)
    plt.close()

if __name__ == "__main__":
    feature_list = pd.read_csv("output/src_results/modeling_results/top_shap_features.csv")["feature"].tolist()
    run_backtesting(
        window_train_months=18,
        test_horizon_months=3,
        output_dir="output/backtest",
        feature_set=feature_list,
        seed=42
    )
    
    