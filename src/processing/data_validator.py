import numpy as np

def validate_data(X, y, name="Data"):
    """Check data quality before training"""
    print(f"\n=== {name} Validation ===")
    
    # Kiểm tra NaN
    nan_count = X.isnull().sum().sum()
    print(f"NaN values: {nan_count}")
    
    # Kiểm tra Infinity
    inf_count = np.isinf(X.values).sum()
    print(f"Infinity values: {inf_count}")
    
    # Kiểm tra range
    print(f"Min value: {X.min().min():.6f}")
    print(f"Max value: {X.max().max():.6f}")
    
    # Kiểm tra variance
    zero_var_features = X.var() == 0
    print(f"Zero variance features: {zero_var_features.sum()}")
    
    # Kiểm tra target
    if y is not None:
        print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Cảnh báo nếu có vấn đề
    if nan_count > 0:
        print("WARNING: Found NaN values!")
    if inf_count > 0:
        print("WARNING: Found Infinity values!")
    if zero_var_features.sum() > 0:
        print("WARNING: Found zero variance features!")
    
    print("=" * 30)