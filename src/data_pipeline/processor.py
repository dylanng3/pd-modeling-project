import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from src.data_pipeline.loaders import load_application_train, load_application_test, load_bureau, load_bureau_balance, load_previous_application
from src.processing.imputation import SimpleImputer
from src.processing.encoding import TargetEncoder

class DataProcessor:
    def __init__(self, debug=False, seed=42, force_reload=False):
        self.debug = debug
        self.seed = seed
        self.force_reload = force_reload
        self.numeric_transformer = StandardScaler()
        self.categorical_encoder = TargetEncoder()
        self.imputer = SimpleImputer()
        self.cache_path = 'data/interim/cache_merged.feather'
        self.fe_cache_path = 'data/interim/cache_fe.feather'
        
    def clean_column_names(self, df):
        """Làm sạch tên cột để tương thích với XGBoost"""
        clean_names = {}
        for col in df.columns:
            # Thay thế các ký tự không hợp lệ
            clean_name = str(col)
            clean_name = clean_name.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            clean_name = clean_name.replace('(', '_').replace(')', '_').replace(' ', '_')
            clean_name = clean_name.replace('-', '_').replace('.', '_').replace(',', '_')
            # Loại bỏ các ký tự đặc biệt khác
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            # Đảm bảo không bắt đầu bằng số
            if clean_name and clean_name[0].isdigit():
                clean_name = 'F_' + clean_name
            # Đảm bảo không trùng lặp
            if clean_name in clean_names.values():
                i = 1
                while f"{clean_name}_{i}" in clean_names.values():
                    i += 1
                clean_name = f"{clean_name}_{i}"
            clean_names[col] = clean_name
        return clean_names
        
    def load_data(self):
        """Tải dữ liệu từ các nguồn và hợp nhất"""
        import os
        os.makedirs('data/interim', exist_ok=True)
        if os.path.exists(self.cache_path) and not self.force_reload:
            print(f"Đang load cache merge từ {self.cache_path}...")
            app = pd.read_feather(self.cache_path)
            return app
        if self.debug:
            print("Debug mode - Chỉ tải 10000 mẫu")
        # Tải dữ liệu chính
        app_train = load_application_train(nrows=10000 if self.debug else None)
        app_test = load_application_test(nrows=10000 if self.debug else None)
        app_test['TARGET'] = np.nan  # Đảm bảo test có cột TARGET là NaN
        app = pd.concat([app_train, app_test]).reset_index(drop=True)
        # Tải dữ liệu bổ sung
        bureau = self._load_and_process_bureau()
        prev = self._load_and_process_previous_apps()
        # Hợp nhất dữ liệu
        print("Before merge:", app.shape)
        print('Số dòng test trước merge:', app['TARGET'].isnull().sum())
        app = app.merge(bureau, on='SK_ID_CURR', how='left')
        print('Số dòng test sau merge bureau:', app['TARGET'].isnull().sum())
        app = app.merge(prev, on='SK_ID_CURR', how='left')
        print('Số dòng test sau merge prev:', app['TARGET'].isnull().sum())
        gc.collect()
        print(f"Lưu cache merge vào {self.cache_path}...")
        print('Số dòng test sau merge:', app['TARGET'].isnull().sum())
        app.reset_index(drop=True).to_feather(self.cache_path)
        return app
    
    def _load_and_process_bureau(self):
        """Xử lý dữ liệu bureau với thống kê tự động"""
        bureau = load_bureau(nrows=10000 if self.debug else None)
        bb = load_bureau_balance(nrows=10000 if self.debug else None)
        
        # Tạo thống kê tự động
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'DAYS_CREDIT': ['mean', 'var', 'min', 'max'],
            'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
            'CREDIT_ACTIVE': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'  # Lấy giá trị phổ biến nhất
        })
        
        # Làm phẳng multi-index columns
        bureau_agg.columns = [f'BURO_{col[0]}_{col[1]}' for col in bureau_agg.columns]
        return bureau_agg.reset_index()
    
    def _load_and_process_previous_apps(self):
        """Xử lý đơn xin vay trước đây"""
        prev = load_previous_application(nrows=10000 if self.debug else None)
        
        # Tạo thống kê tự động
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'AMT_CREDIT': ['mean', 'sum', 'max'],
            'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').mean()
        })
        
        prev_agg.columns = [f'PREV_{col[0]}_{col[1]}' for col in prev_agg.columns]
        return prev_agg.reset_index()
    
    def auto_feature_engineering(self, df, disable_autoencoder=False):
        import os
        os.makedirs('data/interim', exist_ok=True)
        if os.path.exists(self.fe_cache_path) and not self.force_reload:
            print(f"Đang load cache feature engineering từ {self.fe_cache_path}...")
            return pd.read_feather(self.fe_cache_path)
        print('Bắt đầu feature engineering...')

        
        # --- Feature ratios and interactions inspired by top solutions ---
        if 'EXT_SOURCE_3' in df.columns:
            for base in ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL']:
                if base in df.columns:
                    df[f'{base}_DIV_EXTSRC3'] = df[base] / (df['EXT_SOURCE_3'] + 1e-5)
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_ANNUITY']):
            df['CREDIT_TO_ANNUITY'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1e-5)
            df['ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_GOODS_PRICE']):
            df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
            df['GOODS_MINUS_CREDIT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
        # --- Demographic and region features ---
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
        if 'REGION_POPULATION_RELATIVE' in df.columns:
            df['REGION_CAT'] = pd.qcut(df['REGION_POPULATION_RELATIVE'], 5, labels=False, duplicates='drop')
        # --- Debt/Credit and financial health ---
        if 'BURO_AMT_CREDIT_SUM_DEBT_sum' in df.columns and 'BURO_AMT_CREDIT_SUM_sum' in df.columns:
            df['DEBT_OVER_TOTAL_CREDIT'] = df['BURO_AMT_CREDIT_SUM_DEBT_sum'] / (df['BURO_AMT_CREDIT_SUM_sum'] + 1e-5)
        # --- Interest and payment KPIs ---
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_ANNUITY', 'CNT_PAYMENT']):
            df['ESTIMATED_INTEREST_RATE'] = (df['AMT_ANNUITY'] * df['CNT_PAYMENT'] - df['AMT_CREDIT']) / (df['AMT_CREDIT'] + 1e-5)
        if all(col in df.columns for col in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']):
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
        # --- Label encoding for all categoricals (robust) ---
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            try:
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception:
                df[col] = -1
        # --- Aggregates over slices of previous_application (recent/early) ---
        if 'SK_ID_CURR' in df.columns and 'PREV_AMT_CREDIT_mean' in df.columns:
            for n in [2, 4, 6]:
                col_last = f'PREV_AMT_CREDIT_recent{n}_avg'
                if col_last not in df.columns:
                    df[col_last] = np.nan
            for n in [1, 3]:
                col_first = f'PREV_AMT_CREDIT_early{n}_avg'
                if col_first not in df.columns:
                    df[col_first] = np.nan
        # --- Lag features for previous_application (up to last 3) ---
        for lag in range(1, 4):
            colname = f'PREV_AMT_CREDIT_lag_{lag}'
            if colname not in df.columns:
                df[colname] = np.nan
        # --- Aggregates for installment, POS, credit card (modern style) ---
        for src in ['INST', 'POS', 'CC']:
            for n in [3, 6]:
                colname = f'{src}_PAYMENT_recent{n}_avg'
                if colname not in df.columns:
                    df[colname] = np.nan
        # --- Nearest neighbors target mean (placeholder) ---
        df['NN_TARGET_MEAN_300'] = np.nan
        
        # Làm sạch tên cột
        clean_names = self.clean_column_names(df)
        df = df.rename(columns=clean_names)
        
        print(f"Lưu cache feature engineering vào {self.fe_cache_path}...")
        df.reset_index(drop=True).to_feather(self.fe_cache_path)
        return df
    
    def handle_missing_values(self, df):
        """Xử lý giá trị thiếu một cách nhất quán, không fillna cột TARGET"""
        # Điền giá trị thiếu cho số bằng median, trừ cột TARGET
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'TARGET']
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        # Điền giá trị thiếu cho phân loại bằng mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Kiểm tra xem cột có chứa dict không
            if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, dict)).any():
                df[col] = df[col].fillna('Unknown')
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        return df