import numpy as np
import pandas as pd

# Danh sách các cột object/categorical
CATEGORICAL_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
    'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE'
]

# Danh sách các cột int (có thể điều chỉnh thêm nếu cần)
INT_COLS = [
    'SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
    'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'
] + [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]

# Tạo dtype_dict cho train/test
DTYPE_DICT = {col: 'category' for col in CATEGORICAL_COLS}
DTYPE_DICT.update({col: np.int32 for col in INT_COLS if col != 'TARGET'})
DTYPE_DICT['TARGET'] = np.int8  # TARGET chỉ có 0/1

def load_application_train(path='data/raw/application_train.csv', nrows=None):
    df = pd.read_csv(path, dtype=DTYPE_DICT, nrows=nrows)
    # Chuyển các cột float64 còn lại sang float32
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    return df

def load_application_test(path='data/raw/application_test.csv', nrows=None):
    # application_test không có cột TARGET
    dtype_test = DTYPE_DICT.copy()
    dtype_test.pop('TARGET', None)
    df = pd.read_csv(path, dtype=dtype_test, nrows=nrows)
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    return df

def load_bureau(path='data/raw/bureau.csv', nrows=None):
    dtype_bureau = {
        'SK_ID_CURR': np.int32,
        'SK_ID_BUREAU': np.int32,
        'CREDIT_ACTIVE': 'category',
        'CREDIT_CURRENCY': 'category',
        'DAYS_CREDIT': np.int32,
        'CREDIT_DAY_OVERDUE': np.int32,
        'DAYS_CREDIT_ENDDATE': np.float32,
        'DAYS_ENDDATE_FACT': np.float32,
        'AMT_CREDIT_MAX_OVERDUE': np.float32,
        'CNT_CREDIT_PROLONG': np.int32,
        'AMT_CREDIT_SUM': np.float32,
        'AMT_CREDIT_SUM_DEBT': np.float32,
        'AMT_CREDIT_SUM_LIMIT': np.float32,
        'AMT_CREDIT_SUM_OVERDUE': np.float32,
        'CREDIT_TYPE': 'category',
        'DAYS_CREDIT_UPDATE': np.int32,
        'AMT_ANNUITY': np.float32
    }
    df = pd.read_csv(path, dtype=dtype_bureau, nrows=nrows)
    return df

def load_bureau_balance(path='data/raw/bureau_balance.csv', nrows=None):
    dtype_bureau_balance = {
        'SK_ID_BUREAU': np.int32,
        'MONTHS_BALANCE': np.int16,  # Giá trị âm, nhưng nhỏ, dùng int16 là đủ
        'STATUS': 'category'
    }
    df = pd.read_csv(path, dtype=dtype_bureau_balance, nrows=nrows)
    return df

def load_previous_application(path='data/raw/previous_application.csv', nrows=None):
    dtype_prev_app = {
        'SK_ID_PREV': np.int32,
        'SK_ID_CURR': np.int32,
        'NAME_CONTRACT_TYPE': 'category',
        'AMT_ANNUITY': np.float32,
        'AMT_APPLICATION': np.float32,
        'AMT_CREDIT': np.float32,
        'AMT_DOWN_PAYMENT': np.float32,
        'AMT_GOODS_PRICE': np.float32,
        'WEEKDAY_APPR_PROCESS_START': 'category',
        'HOUR_APPR_PROCESS_START': np.int8,
        'FLAG_LAST_APPL_PER_CONTRACT': 'category',
        'NFLAG_LAST_APPL_IN_DAY': np.int8,
        'RATE_DOWN_PAYMENT': np.float32,
        'RATE_INTEREST_PRIMARY': np.float32,
        'RATE_INTEREST_PRIVILEGED': np.float32,
        'NAME_CASH_LOAN_PURPOSE': 'category',
        'NAME_CONTRACT_STATUS': 'category',
        'DAYS_DECISION': np.int16,
        'NAME_PAYMENT_TYPE': 'category',
        'CODE_REJECT_REASON': 'category',
        'NAME_TYPE_SUITE': 'category',
        'NAME_CLIENT_TYPE': 'category',
        'NAME_GOODS_CATEGORY': 'category',
        'NAME_PORTFOLIO': 'category',
        'NAME_PRODUCT_TYPE': 'category',
        'CHANNEL_TYPE': 'category',
        'SELLERPLACE_AREA': np.int16,
        'NAME_SELLER_INDUSTRY': 'category',
        'CNT_PAYMENT': np.float32,
        'NAME_YIELD_GROUP': 'category',
        'PRODUCT_COMBINATION': 'category',
        'DAYS_FIRST_DRAWING': np.float32,
        'DAYS_FIRST_DUE': np.float32,
        'DAYS_LAST_DUE_1ST_VERSION': np.float32,
        'DAYS_LAST_DUE': np.float32,
        'DAYS_TERMINATION': np.float32,
        'NFLAG_INSURED_ON_APPROVAL': np.float32
    }
    df = pd.read_csv(path, dtype=dtype_prev_app, nrows=nrows)
    return df

def load_installments_payments(path='data/raw/installments_payments.csv', nrows=None):
    dtype_inst = {
        'SK_ID_PREV': np.int32,
        'SK_ID_CURR': np.int32,
        'NUM_INSTALMENT_VERSION': np.float32,
        'NUM_INSTALMENT_NUMBER': np.float32,
        'DAYS_INSTALMENT': np.float32,
        'DAYS_ENTRY_PAYMENT': np.float32,
        'AMT_INSTALMENT': np.float32,
        'AMT_PAYMENT': np.float32
    }
    df = pd.read_csv(path, dtype=dtype_inst, nrows=nrows)
    return df
