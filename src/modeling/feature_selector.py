import numpy as np
import xgboost as xgb
import shap

class FeatureSelector:
    def __init__(self, top_k=30, seed=42, shap_sample=3000, full_shap=False):
        self.top_k = top_k
        self.selected_features = None
        self.seed = seed
        self.shap_sample = shap_sample
        self.full_shap = full_shap
        self.model = None
    
    def fit(self, X, y):
        print('[SHAP] Starting model training for feature selection...')
        X_filled = X.fillna(X.median())
        
        # Ensure clean column names for XGBoost
        clean_names = {}
        for col in X_filled.columns:
            clean_name = str(col)
            clean_name = clean_name.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            clean_name = clean_name.replace('(', '_').replace(')', '_').replace(' ', '_')
            clean_name = clean_name.replace('-', '_').replace('.', '_').replace(',', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            if clean_name and clean_name[0].isdigit():
                clean_name = 'F_' + clean_name
            if clean_name in clean_names.values():
                i = 1
                while f"{clean_name}_{i}" in clean_names.values():
                    i += 1
                clean_name = f"{clean_name}_{i}"
            clean_names[col] = clean_name
        
        X_filled = X_filled.rename(columns=clean_names)
        
        # Use XGBoost for SHAP
        print('[SHAP] Training XGBoost model for SHAP...')
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric='auc'
        )
        self.model.fit(X_filled, y)
        
        # Optimize SHAP calculation
        if self.full_shap or len(X_filled) <= self.shap_sample:
            max_sample = min(5000, len(X_filled))
            if len(X_filled) > max_sample:
                print(f'[SHAP] Reducing sample size from {len(X_filled)} to {max_sample} for speed...')
                X_sample = X_filled.sample(max_sample, random_state=self.seed)
            else:
                X_sample = X_filled
        else:
            print(f'[SHAP] Using sample size {self.shap_sample}...')
            X_sample = X_filled.sample(self.shap_sample, random_state=self.seed)
        
        print(f'[SHAP] Starting SHAP values calculation on {len(X_sample)} rows...')
        
        # Use TreeExplainer for XGBoost with progress tracking
        explainer = shap.TreeExplainer(self.model)
        # Calculate SHAP values with progress tracking
        shap_values = explainer.shap_values(X_sample)
        
        print('[SHAP] SHAP values calculation completed.')
        
        # Calculate feature importance
        print('[SHAP] Calculating feature importance...')
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Get top_k unique features
        selected_indices = []
        seen_features = set()
        
        print(f'[SHAP] Selecting top {self.top_k} features...')
        for idx in sorted_indices:
            # Convert numpy array to scalar if needed
            if hasattr(idx, 'item') and idx.size == 1:
                idx_scalar = idx.item()
            elif hasattr(idx, 'flatten'):
                idx_scalar = int(idx.flatten()[0])
            else:
                idx_scalar = int(idx)
                
            feature_name = str(X_filled.columns[idx_scalar])
            if feature_name not in seen_features:
                selected_indices.append(idx_scalar)
                seen_features.add(feature_name)
                if len(selected_indices) >= self.top_k:
                    break
        
        # If not enough, add more
        if len(selected_indices) < self.top_k:
            remaining_indices = [i for i in range(len(X_filled.columns)) if i not in selected_indices]
            for idx in remaining_indices[:self.top_k - len(selected_indices)]:
                selected_indices.append(idx)
        
        # Map back to original column names
        self.selected_features = X.columns[selected_indices]
        
        # Print top features with classification
        print(f'[SHAP] Top {len(self.selected_features)} features selected:')
        original_count = 0
        other_count = 0
        
        for i, feature in enumerate(self.selected_features):
            importance = feature_importance[selected_indices[i]]
            # Convert importance to scalar if it's a numpy array
            if hasattr(importance, 'item') and importance.size == 1:
                importance_scalar = importance.item()
            elif hasattr(importance, 'flatten'):
                importance_scalar = float(importance.flatten()[0])
            else:
                importance_scalar = float(importance)
                
            feature_str = str(feature)
            if any(prefix in feature_str for prefix in ['AMT_', 'DAYS_', 'CNT_', 'EXT_', 'FLAG_', 'NAME_', 'CODE_', 'ORGANIZATION_', 'CREDIT_', 'ANNUITY_', 'GOODS_', 'DOWNPAYMENT_', 'CHILDREN_', 'EMPLOYED_', 'REGISTRATION_', 'ID_PUBLISH_', 'NEW_', 'BURO_', 'PREV_']):
                original_count += 1
                feature_type = "ORIG"
            else:
                other_count += 1
                feature_type = "OTHER"
            
            print(f'  {i+1:2d}. [{feature_type}] {feature_str}: {importance_scalar:.4f}')
        
        print(f'[SHAP] Summary: {original_count} original, {other_count} other')
        
        return self.selected_features
    
    def transform(self, X):
        return X[self.selected_features]