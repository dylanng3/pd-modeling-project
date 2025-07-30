import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

def xgboost_tuning(X, y, seed=42, n_trials=30, n_folds=3):
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0, 0.2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
            'random_state': seed,
            'n_jobs': -1,
            'use_label_encoder': False,
            'verbosity': 0
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)
    return study.best_params

def lightgbm_tuning(X, y, seed=42, n_trials=20, n_folds=3):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)
    return study.best_params

def catboost_tuning(X, y, seed=42, n_trials=20, n_folds=3):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': seed,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)
    return study.best_params