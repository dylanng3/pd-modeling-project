import pandas as pd

class TargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.maps = {}
        
    def fit_transform(self, df, target, categorical_cols):
        """Mã hóa các biến phân loại sử dụng thông tin mục tiêu"""
        encoded_df = df.copy()
        
        for col in categorical_cols:
            # Tính toán mã hóa mục tiêu
            averages = encoded_df.groupby(col)[target].mean()
            counts = encoded_df.groupby(col)[target].count()
            
            # Áp dụng smoothing
            self.maps[col] = (averages * counts + self.smoothing * averages.mean()) / (counts + self.smoothing)
            
            # Áp dụng mã hóa và chuyển sang numeric
            encoded_df[col] = encoded_df[col].map(self.maps[col])
            encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')
        
        return encoded_df
    
    def transform(self, df, categorical_cols):
        """Áp dụng mã hóa cho dữ liệu mới"""
        encoded_df = df.copy()
        
        for col in categorical_cols:
            if col in self.maps and len(self.maps[col]) > 0:
                # Map các giá trị đã biết
                encoded_df[col] = encoded_df[col].map(self.maps[col])
                # Chuyển sang numeric và điền giá trị trung bình cho missing
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')
                encoded_df[col] = encoded_df[col].fillna(self.maps[col].mean())
            else:
                # Nếu không có map, chuyển sang numeric và điền 0
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')
                encoded_df[col] = encoded_df[col].fillna(0)
        
        return encoded_df