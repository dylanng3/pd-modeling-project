import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from src.modeling.tuner import lightgbm_tuning, catboost_tuning, xgboost_tuning

def lightgbm_pred(X, y, X_test, model_params=None, model_name="lgbm", n_splits=5, seed=42, tune_hyperparams=False, n_trials=20):
    """Generate OOF and test predictions for LightGBM. Support hyperparameter tuning."""
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1
        }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if tune_hyperparams:
            print(f"[LGBM] Optuna tuning fold {fold+1}/{n_splits}...")
            best_params = lightgbm_tuning(X_train, y_train, seed=seed, n_trials=n_trials)
            model = lgb.LGBMClassifier(**best_params, random_state=seed, n_jobs=-1, verbose=-1)
        else:
            model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        models.append(model)
        print(f"[LGBM] Fold {fold+1}/{n_splits} done.")
    print(f"[LGBM] OOF and test predictions generated for {model_name}.")
    return oof_preds, test_preds, models[-1] if models else None

def catboost_pred(X, y, X_test, model_params=None, model_name="catboost", n_splits=5, seed=42, tune_hyperparams=False, n_trials=20):
    if model_params is None:
        model_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': seed,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False
        }
    model_params['train_dir'] = 'models/l1_stacking/catboost_info'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if tune_hyperparams:
            print(f"[CatBoost] Optuna tuning fold {fold+1}/{n_splits}...")
            best_params = catboost_tuning(X_train, y_train, seed=seed, n_trials=n_trials)
            best_params['train_dir'] = 'models/l1_stacking/catboost_info'
            model = CatBoostClassifier(**best_params)
        else:
            model = CatBoostClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        models.append(model)
        print(f"[CatBoost] Fold {fold+1}/{n_splits} done.")
    print(f"[CatBoost] OOF and test predictions generated for {model_name}.")
    return oof_preds, test_preds, models[-1] if models else None

def xgboost_pred(X, y, X_test, model_params=None, model_name="xgb", n_splits=5, seed=42, tune_hyperparams=False, n_trials=20):
    """Generate OOF and test predictions for XGBoost."""
    if model_params is None:
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': seed,
            'n_jobs': -1,
            'use_label_encoder': False,
            'verbosity': 0
        }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if tune_hyperparams:
            print(f"[XGB] Optuna tuning fold {fold+1}/{n_splits}...")
            best_params = xgboost_tuning(X_train, y_train, seed=seed, n_trials=n_trials)
            model = xgb.XGBClassifier(**best_params)
        else:
            model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        models.append(model)
        print(f"[XGB] Fold {fold+1}/{n_splits} done.")
    print(f"[XGB] OOF and test predictions generated for {model_name}.")
    return oof_preds, test_preds, models[-1] if models else None