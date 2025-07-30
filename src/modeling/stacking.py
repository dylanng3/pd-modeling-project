import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.modeling.base_models import xgboost_pred, lightgbm_pred, catboost_pred

# ============================================================
# L1 STACKING
# ============================================================

def run_l1_stacking(X_train_selected, y_train, X_test_selected, tune_hyperparams=False):
    """
    Generate OOF predictions for each L1 base model, return predictions, models, and metrics.
    """
    L1_MODELS = [
        {"name": "xgb", "func": xgboost_pred, "params": None},
        {"name": "lgbm", "func": lightgbm_pred, "params": None},
        {"name": "catboost", "func": catboost_pred, "params": None},
    ]

    print('\n=== Generating L1 OOF predictions for stacking ===')
    oof_preds_l1 = {}
    test_preds_l1 = {}
    models_l1 = {}

    for model in L1_MODELS:
        print(f'Running {model["name"]} ...')
        oof_preds, test_preds, trained_model = model["func"](
            X_train_selected, y_train, X_test_selected,
            model_params=model["params"],
            model_name=model["name"],
            tune_hyperparams=tune_hyperparams
        )
        oof_preds_l1[model["name"]] = oof_preds
        test_preds_l1[model["name"]] = test_preds
        models_l1[model["name"]] = trained_model

    print('=== L1 OOF predictions generated ===\n')
    aucs_l1 = {}
    y_true = y_train.reset_index(drop=True)

    for name in oof_preds_l1:
        aucs_l1[name] = roc_auc_score(y_true, oof_preds_l1[name])

    metrics_l1 = {"auc": aucs_l1}
    return models_l1, oof_preds_l1, test_preds_l1, metrics_l1

# ============================================================
# L2 STACKING
# ============================================================

def run_l2_stacking(y_train, X_train_selected, X_test_selected):
    """
    Generate OOF predictions for L2 meta-models, return models, predictions, and metrics.
    """
    L2_MODELS = [
        {"name": "extratree", "model": ExtraTreesClassifier(max_depth=4, min_samples_leaf=1000, random_state=42)},
        {"name": "logistic", "model": LogisticRegression(max_iter=2000, solver='liblinear')},
    ]

    l1_names = ['xgb', 'lgbm', 'catboost']
    # Read OOF and test predictions from L1 stacking
    l1_oof = [pd.read_csv(f'models/l1_stacking/l1_{name}_oof_predictions.csv').rename(columns={'oof_preds': f'pred_{name}'}) for name in l1_names]
    l1_test = [pd.read_csv(f'models/l1_stacking/l1_{name}_test_predictions.csv').rename(columns={'test_preds': f'pred_{name}'}) for name in l1_names]
    X_l2 = pd.concat(l1_oof, axis=1)
    X_test_l2 = pd.concat(l1_test, axis=1)
    y_l2 = y_train.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print('\n=== Generating L2 OOF predictions for stacking ===')
    models_l2 = {}
    oof_preds_l2 = {}
    test_preds_l2 = {}
    aucs_l2 = {}

    for meta in L2_MODELS:
        print(f'Running {meta["name"]} ...')
        oof = np.zeros(len(X_l2))
        test = np.zeros(len(X_test_l2))
        for train_idx, val_idx in skf.split(X_l2, y_l2):
            X_train, X_val = X_l2.iloc[train_idx], X_l2.iloc[val_idx]
            y_train_, y_val = y_l2.iloc[train_idx], y_l2.iloc[val_idx]
            model = meta["model"]
            model.fit(X_train, y_train_)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            test += model.predict_proba(X_test_l2)[:, 1] / skf.n_splits
        models_l2[meta["name"]] = model
        oof_preds_l2[meta["name"]] = oof
        test_preds_l2[meta["name"]] = test
        aucs_l2[meta["name"]] = roc_auc_score(y_l2, oof)

    metrics_l2 = {"auc": aucs_l2}
    print('=== L2 OOF predictions generated ===\n')
    return models_l2, oof_preds_l2, test_preds_l2, metrics_l2

# ============================================================
# L3 STACKING
# ============================================================

def run_l3_stacking(y_train, test_df, l2_model_names, X_train_selected, X_test_selected, raw_feature_names=None):
    """
    L3 stacking with ExtraTreesClassifier, input is L2 prediction files and raw features (if any).
    Returns model, oof_preds, test_preds, metrics. Does not save files in this function.
    """
    # Read OOF and test predictions from L2 stacking
    l2_oof_list = [pd.read_csv(f'models/l2_stacking/l2_{name}_oof_predictions.csv').rename(columns={'oof_preds': f'pred_{name}'}) for name in l2_model_names]
    l2_test_list = [pd.read_csv(f'models/l2_stacking/l2_{name}_test_predictions.csv').rename(columns={'test_preds': f'pred_{name}'}) for name in l2_model_names]
    X_l3 = pd.concat(l2_oof_list, axis=1)
    X_test_l3 = pd.concat(l2_test_list, axis=1)
    y_l3 = y_train.reset_index(drop=True)

    # Optionally add raw features to L3 stacking
    if raw_feature_names is not None and len(raw_feature_names) > 0:
        for feat in raw_feature_names:
            X_l3[feat] = X_train_selected[feat].reset_index(drop=True)
            X_test_l3[feat] = X_test_selected[feat].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_l3 = np.zeros(len(X_l3))
    test_preds_l3 = np.zeros(len(X_test_l3))

    for train_idx, val_idx in skf.split(X_l3, y_l3):
        X_train, X_val = X_l3.iloc[train_idx], X_l3.iloc[val_idx]
        y_train_, y_val = y_l3.iloc[train_idx], y_l3.iloc[val_idx]
        model_l3 = ExtraTreesClassifier(max_depth=4, min_samples_leaf=1000, random_state=42)
        model_l3.fit(X_train, y_train_)
        oof_preds_l3[val_idx] = model_l3.predict_proba(X_val)[:, 1]
        test_preds_l3 += model_l3.predict_proba(X_test_l3)[:, 1] / skf.n_splits

    auc_l3 = roc_auc_score(y_l3, oof_preds_l3)
    metrics_l3 = {"auc": auc_l3}
    print(f'Final L3 stacking completed.')
    return model_l3, oof_preds_l3, test_preds_l3, metrics_l3

# ============================================================
# UTILITY
# ============================================================

def print_all_auc(y_train):
    y_true = y_train.reset_index(drop=True)
    # L1
    for name in ['xgb', 'lgbm', 'catboost']:
        oof = pd.read_csv(f'models/l1_stacking/l1_{name}_oof_predictions.csv')['oof_preds']
        print(f'L1 {name.upper()} OOF AUC: {roc_auc_score(y_true, oof):.5f}')
    # L2
    for name in ['extratree', 'logistic']:
        oof = pd.read_csv(f'models/l2_stacking/l2_{name}_oof_predictions.csv')['oof_preds']
        print(f'L2 {name} OOF AUC: {roc_auc_score(y_true, oof):.5f}')
    # L3
    oof = pd.read_csv('models/l3_stacking/l3_extratree_oof_predictions.csv')['oof_preds']
    print(f'L3 extratree OOF AUC: {roc_auc_score(y_true, oof):.5f}')