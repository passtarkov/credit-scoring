import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует bureau.csv — одна строка на клиента.
    
    Принимает: bureau DataFrame (сырой)
    Возвращает: DataFrame с агрегированными фичами по SK_ID_CURR
    """
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_CREDIT_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_ACTIVE_CREDIT_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        BUREAU_AMT_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
        BUREAU_AMT_OVERDUE_MAX=('AMT_CREDIT_SUM_OVERDUE', 'max'),
        BUREAU_DAYS_OVERDUE_MEAN=('CREDIT_DAY_OVERDUE', 'mean'),
        BUREAU_AMT_CREDIT_MAX=('AMT_CREDIT_SUM', 'max'),
    ).reset_index()
    
    return bureau_agg


def build_features(app: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Полный feature engineering pipeline.
    
    Принимает: app (application_train), bureau (bureau.csv)
    Возвращает: DataFrame готовый для обучения модели
    """
    bureau_agg = build_bureau_features(bureau)
    app = app.merge(bureau_agg, on='SK_ID_CURR', how='left')    
    app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].replace(365243, np.nan)

    new_features = {
        'AGE_YEARS': -app['DAYS_BIRTH'] / 365,
        'YEARS_EMPLOYED': -app['DAYS_EMPLOYED'] / 365,
        'ANNUITY_INCOME_RATIO': app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL'],
        'CREDIT_INCOME_RATIO': app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL'],
        'CREDIT_GOODS_RATIO': app['AMT_CREDIT'] / app['AMT_GOODS_PRICE'],
        'EXT_SOURCE_MEAN': app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1),
        'EXT_SOURCE_PROD': app['EXT_SOURCE_1'] * app['EXT_SOURCE_2'] * app['EXT_SOURCE_3'],
    }
    app = pd.concat([app, pd.DataFrame(new_features)], axis=1)
    
    cat_cols = app.select_dtypes(include=['object', 'str']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        app[col] = le.fit_transform(app[col].fillna('Unknown'))
    
    app = app.copy()
    
    return app

FEATURE_COLS = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
    'AGE_YEARS', 'YEARS_EMPLOYED',
    'ANNUITY_INCOME_RATIO', 'CREDIT_INCOME_RATIO',
    'CREDIT_GOODS_RATIO', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_PROD',
    'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
    'BUREAU_CREDIT_COUNT', 'BUREAU_ACTIVE_CREDIT_COUNT',
    'BUREAU_AMT_DEBT_MEAN', 'BUREAU_AMT_OVERDUE_MAX',
    'BUREAU_DAYS_OVERDUE_MEAN', 'BUREAU_AMT_CREDIT_MAX'
]