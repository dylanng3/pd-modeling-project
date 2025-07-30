"""
main.py - Credit Risk Modeling Project

This script serves as the main entry point for the end-to-end credit risk modeling pipeline. 
It orchestrates the data loading, preprocessing, feature engineering, feature selection, encoding, 
imputation, model training (including stacking), and artifact saving. The script is designed for 
both quick debugging and full production runs, and outputs all necessary intermediate and final 
datasets, trained models, predictions, and model summaries.

Usage:
    python main.py

Arguments (see main function):
    debug: Run in debug mode with a subset of data for fast iteration.
    seed: Random seed for reproducibility.
    force_reload: Force reload of data, ignoring any cached files.
    skip_shap: Skip SHAP-based feature selection for faster runs.
    use_ensemble: Whether to use ensemble models.
    tune_hyperparams: Whether to perform hyperparameter tuning.

Outputs:
    - Intermediate and processed datasets (data/interim/, data/processed/)
    - Trained models, predictions, and summaries (models/)
    - Console logs for pipeline progress and diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.utils.utils import set_seed, timer, create_progress_bar
from src.data_pipeline.processor import DataProcessor
from src.processing.data_validator import validate_data
from src.processing.encoding import TargetEncoder
from src.processing.imputation import SimpleImputer
from src.modeling.feature_selector import FeatureSelector
from src.modeling.stacking import run_l1_stacking, run_l2_stacking, run_l3_stacking, print_all_auc  

import pickle
import json


def main(debug=False, seed=42, force_reload=False, skip_shap=False, use_ensemble=False, tune_hyperparams=True):
    set_seed(seed)
    print(f"Using seed: {seed}")
    print(f"Debug mode: {debug}")
    print(f"Force reload: {force_reload}")
    print(f"Skip SHAP: {skip_shap}")
    print(f"Use Ensemble: {use_ensemble}")
    print(f"Tune Hyperparams: {tune_hyperparams}")
    if debug:
        print("DEMO MODE: Running with first 10,000 rows only!")
    else:
        print("FULL MODE: Running with complete dataset!")
    
    # Create overall progress header for the entire pipeline
    print("\n" + "="*60)
    print("STARTING CREDIT RISK MODELING PIPELINE")
    print("="*60)
    
    with timer("Full pipeline"):
        print('Loading data...')
        processor = DataProcessor(debug=debug, seed=seed, force_reload=force_reload)
        df = processor.load_data()
        print(f'Dataset shape: {df.shape}')
        print('Number of test rows after merge:', df['TARGET'].isnull().sum())
        
        print('Starting auto feature engineering...')
        df = processor.auto_feature_engineering(df)
        print(f'Dataset shape after feature engineering: {df.shape}')
        print('Number of test rows after feature engineering:', df['TARGET'].isnull().sum())
        
        print('Handling missing values...')
        df = processor.handle_missing_values(df)
        print('Number of test rows after handling missing values:', df['TARGET'].isnull().sum())
        
        print('Starting encoding...')
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        target = 'TARGET'
        train_df = df[df[target].notnull()]
        test_df = df[df[target].isnull()]
        print(f'Train set shape: {train_df.shape}')
        print(f'Test set shape: {test_df.shape}')
        
        # Target encoding for categorical variables
        encoder = TargetEncoder()
        train_encoded = encoder.fit_transform(train_df, target, categorical_cols)
        test_encoded = encoder.transform(test_df, categorical_cols)
        
        # === SAVE INTERIM DATA ===
        import os
        os.makedirs('data/interim', exist_ok=True)
        train_encoded.to_csv('data/interim/train_encoded.csv', index=False)
        test_encoded.to_csv('data/interim/test_encoded.csv', index=False)
        
        if skip_shap:
            print('Skipping SHAP - Using variance-based feature selection...')
            X_train = train_encoded.drop(columns=[target, 'SK_ID_CURR'])
            y_train = train_encoded[target]
            print(f'X_train shape: {X_train.shape}')
            
            # Use variance-based selection for faster processing
            selector_var = VarianceThreshold()
            selector_var.fit(X_train.fillna(0))
            variances = selector_var.variances_
            top_idx = np.argsort(variances)[-50:]  # Select top 50 features
            selected_features = X_train.columns[top_idx]
            print(f'Selected {len(selected_features)} features based on variance')
        else:
            print('Starting feature selection (SHAP)...')
            X_train = train_encoded.drop(columns=[target, 'SK_ID_CURR'])
            y_train = train_encoded[target]
            print(f'X_train shape: {X_train.shape}')
            
            # Reduce number of features for SHAP computation
            if X_train.shape[1] > 100:
                print(f'[SHAP] Reducing features from {X_train.shape[1]} to 100 for SHAP...')
                selector_var = VarianceThreshold()
                selector_var.fit(X_train.fillna(0))
                variances = selector_var.variances_
                top_idx = np.argsort(variances)[-100:]
                X_train = X_train.iloc[:, top_idx]
                test_encoded = test_encoded[X_train.columns]
                print(f'[SHAP] After variance selection: {X_train.shape[1]} features')
            
            # SHAP feature selection
            selector = FeatureSelector(
                top_k=50,
                seed=seed, 
                shap_sample=5000,
                full_shap=False
            )
            selected_features = selector.fit(X_train, y_train)
            
            # Additional feature filtering based on correlation
            print('[SHAP] Filtering features based on correlation...')
            X_selected = X_train[selected_features]
            # Ensure X_selected is DataFrame
            if not isinstance(X_selected, pd.DataFrame):
                X_selected = pd.DataFrame(X_selected, columns=selected_features)
            corr_matrix = X_selected.corr().abs()
            
            # Remove features with high correlation (>0.95)
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            
            if to_drop:
                print(f'[SHAP] Removing {len(to_drop)} features with high correlation: {to_drop[:5]}...')
                selected_features = [f for f in selected_features if f not in to_drop]
            
            # Limit back to 50 final features
            if len(selected_features) > 50:
                selected_features = selected_features[:50]
            
            print(f'[SHAP] Finally selected {len(selected_features)} features')
        
        X_train_selected = X_train[selected_features]
        X_test_selected = test_encoded[selected_features]
        
        print('Starting imputer...')
        imputer = SimpleImputer()
        imputer.fit(X_train_selected)
        X_train_selected = imputer.transform(X_train_selected)
        X_test_selected = imputer.transform(X_test_selected)
        print('Imputer completed')
        
        # Ensure X_train_selected and X_test_selected are DataFrames
        if not isinstance(X_train_selected, pd.DataFrame):
            # Ensure selected_features is list
            if isinstance(selected_features, pd.Index):
                selected_features = selected_features.tolist()
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)  # type: ignore
        if not isinstance(X_test_selected, pd.DataFrame):
            # Ensure selected_features is list
            if isinstance(selected_features, pd.Index):
                selected_features = selected_features.tolist()
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)  # type: ignore
        
        # === SAVE FULLY PROCESSED DATA ===
        os.makedirs('data/processed', exist_ok=True)
        X_train_processed = X_train_selected.copy()
        X_train_processed['TARGET'] = y_train.values  # Attach the target column back to train
        X_train_processed.to_csv('data/processed/train_processed.csv', index=False)
        X_test_selected.to_csv('data/processed/test_processed.csv', index=False)
        
        # Check data quality before training
        validate_data(X_train_selected, y_train, "Training Data")
        validate_data(X_test_selected, None, "Test Data")

        # === L1 STACKING: Generate OOF prediction for each base model ===
        models_l1, oof_preds_l1, test_preds_l1, metrics_l1 = run_l1_stacking(X_train_selected, y_train, X_test_selected, tune_hyperparams)
        l1_dir = 'models/l1_stacking'
        os.makedirs(l1_dir, exist_ok=True)
        # List of expected L1 models
        expected_l1_models = ['xgb', 'lgbm', 'catboost']
        # Lưu models, predictions, metrics cho từng model L1
        for name in expected_l1_models:
            # Lưu model (nếu có)
            if name in models_l1 and models_l1[name] is not None:
                with open(f'{l1_dir}/l1_{name}_model.pkl', 'wb') as f:
                    pickle.dump(models_l1[name], f)
            # Lưu OOF predictions
            if name in oof_preds_l1:
                pd.DataFrame({'oof_preds': oof_preds_l1[name]}).to_csv(f'{l1_dir}/l1_{name}_oof_predictions.csv', index=False)
            # Lưu test predictions
            if name in test_preds_l1:
                pd.DataFrame({'test_preds': test_preds_l1[name]}).to_csv(f'{l1_dir}/l1_{name}_test_predictions.csv', index=False)
        # Lưu tổng hợp metrics
        with open(f'{l1_dir}/l1_model_summary.json', 'w') as f:
            json.dump(metrics_l1, f, indent=2)

        # === L2 STACKING: Generate OOF prediction for meta-models ===
        models_l2, oof_preds_l2, test_preds_l2, metrics_l2 = run_l2_stacking(y_train, X_train_selected, X_test_selected)
        l2_dir = 'models/l2_stacking'
        os.makedirs(l2_dir, exist_ok=True)
        expected_l2_models = ['extratree', 'logistic']
        # Save models, predictions, metrics for each L2 model
        for name in expected_l2_models:
            # Save model
            if name in models_l2 and models_l2[name] is not None:
                with open(f'{l2_dir}/l2_{name}_model.pkl', 'wb') as f:
                    pickle.dump(models_l2[name], f)
            # Save OOF predictions
            if name in oof_preds_l2:
                pd.DataFrame({'oof_preds': oof_preds_l2[name]}).to_csv(f'{l2_dir}/l2_{name}_oof_predictions.csv', index=False)
            # Save test predictions
            if name in test_preds_l2:
                pd.DataFrame({'test_preds': test_preds_l2[name]}).to_csv(f'{l2_dir}/l2_{name}_test_predictions.csv', index=False)
        # Save metrics
        with open(f'{l2_dir}/l2_model_summary.json', 'w') as f:
            json.dump(metrics_l2, f, indent=2)
        # Check predictions
        for name in expected_l2_models:
            if name not in oof_preds_l2:
                print(f"[WARNING] L2 model '{name}' did NOT produce OOF predictions!")
            else:
                print(f"[OK] L2 model '{name}' OOF predictions found, length = {len(oof_preds_l2[name])}")
        # BLEND (ENSEMBLE) TEST PREDICTIONS L2
        blended_test_pred_l2 = np.mean([v for v in test_preds_l2.values()], axis=0)
        pd.DataFrame({'blended_test_pred': blended_test_pred_l2}).to_csv(f'{l2_dir}/l2_blended_test_predictions.csv', index=False)

        # === L3 STACKING: ExtraTreesClassifier on the output of L2 models and raw features ===
        l3_dir = 'models/l3_stacking'
        os.makedirs(l3_dir, exist_ok=True)
        l2_model_names = ['extratree', 'logistic']
        raw_feature_names = []
        if 'AMT_INCOME_TOTAL' in X_train_selected.columns:
            raw_feature_names.append('AMT_INCOME_TOTAL')
        # Run L3 stacking
        model_l3, oof_preds_l3, test_preds_l3, metrics_l3 = run_l3_stacking(
            y_train,
            test_df,
            l2_model_names,
            X_train_selected,
            X_test_selected,
            raw_feature_names
        )
        # Save model, predictions, summary for L3
        with open(f'{l3_dir}/l3_extratree_model.pkl', 'wb') as f:
            pickle.dump(model_l3, f)
        pd.DataFrame({'oof_preds': oof_preds_l3}).to_csv(f'{l3_dir}/l3_extratree_oof_predictions.csv', index=False)
        pd.DataFrame({'test_preds': test_preds_l3}).to_csv(f'{l3_dir}/l3_extratree_test_predictions.csv', index=False)
        with open(f'{l3_dir}/l3_model_summary.json', 'w') as f:
            json.dump(metrics_l3, f, indent=2)
        # Check L3 prediction
        if oof_preds_l3 is None or len(oof_preds_l3) == 0:
            print("[WARNING] L3 model did NOT produce OOF predictions!")
        else:
            print(f"[OK] L3 model OOF predictions found, length = {len(oof_preds_l3)}")
        # Save L3 submission
        pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'].reset_index(drop=True), 'TARGET': test_preds_l3}).to_csv(f'{l3_dir}/submission_l3.csv', index=False)
        print(f'Final L3 stacking submission saved to {l3_dir}/submission_l3.csv')

        print_all_auc(y_train)

if __name__ == "__main__":
    # ======  PIPELINE STACKING MODE ======

    # 1. Quick check (debug mode, only 10,000 samples — suitable for testing code and pipeline)
    # main(debug=True, seed=42, force_reload=True, skip_shap=False, tune_hyperparams=False)
    
    # 2. Full run with SHAP feature selection and hyperparameter tuning (recommended for production or final submission)
    main(debug=False, seed=42, force_reload=True, skip_shap=False, tune_hyperparams=True)
