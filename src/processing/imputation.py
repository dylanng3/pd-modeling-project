import numpy as np

class SimpleImputer:
    def __init__(self):
        self.median_values = {}
        self.mode_values = {}
    
    def fit(self, df):
        # Lưu median cho cột số
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            self.median_values[col] = df[col].median()
        
        # Lưu mode cho cột phân loại
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.mode_values[col] = df[col].mode()[0]
    
    def transform(self, df):
        df_filled = df.copy()
        
        # Điền giá trị số bằng median
        for col, value in self.median_values.items():
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(value)
        
        # Điền giá trị phân loại bằng mode
        for col, value in self.mode_values.items():
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(value)
        
        return df_filled