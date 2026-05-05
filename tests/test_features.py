import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import build_bureau_features, build_features, FEATURE_COLS


def make_mock_app(n=10):
    """Минимальный датафрейм имитирующий application_train"""
    return pd.DataFrame({
        'SK_ID_CURR': range(1, n + 1),
        'TARGET': [0, 1] * (n // 2),
        'DAYS_BIRTH': [-15000] * n,
        'DAYS_EMPLOYED': [-2000] * n,
        'AMT_CREDIT': [500000.0] * n,
        'AMT_ANNUITY': [25000.0] * n,
        'AMT_INCOME_TOTAL': [150000.0] * n,
        'AMT_GOODS_PRICE': [450000.0] * n,
        'REGION_POPULATION_RELATIVE': [0.02] * n,
        'EXT_SOURCE_1': [0.5] * n,
        'EXT_SOURCE_2': [0.6] * n,
        'EXT_SOURCE_3': [0.7] * n,
        'CODE_GENDER': ['M', 'F'] * (n // 2),
        'NAME_CONTRACT_TYPE': ['Cash loans'] * n,
        'NAME_EDUCATION_TYPE': ['Higher education'] * n,
        'NAME_INCOME_TYPE': ['Working'] * n,
        'OCCUPATION_TYPE': ['Laborers'] * n,
        'ORGANIZATION_TYPE': ['Business Entity Type 3'] * n,
    })


def make_mock_bureau(n=20):
    """Минимальный датафрейм имитирующий bureau.csv"""
    return pd.DataFrame({
        'SK_ID_CURR': list(range(1, 11)) * 2,
        'SK_ID_BUREAU': range(1, n + 1),
        'CREDIT_ACTIVE': ['Active', 'Closed'] * (n // 2),
        'AMT_CREDIT_SUM_DEBT': [50000.0] * n,
        'AMT_CREDIT_SUM_OVERDUE': [0.0] * n,
        'CREDIT_DAY_OVERDUE': [0] * n,
        'AMT_CREDIT_SUM': [200000.0] * n,
    })


def test_build_bureau_features_shape():
    """После агрегации — одна строка на клиента"""
    bureau = make_mock_bureau()
    result = build_bureau_features(bureau)
    assert result.shape[0] == 10


def test_build_bureau_features_columns():
    """Все нужные колонки присутствуют"""
    bureau = make_mock_bureau()
    result = build_bureau_features(bureau)
    expected_cols = [
        'BUREAU_CREDIT_COUNT', 'BUREAU_ACTIVE_CREDIT_COUNT',
        'BUREAU_AMT_DEBT_MEAN', 'BUREAU_AMT_OVERDUE_MAX',
        'BUREAU_DAYS_OVERDUE_MEAN', 'BUREAU_AMT_CREDIT_MAX'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Колонка {col} отсутствует"


def test_build_features_no_missing_in_new_cols():
    """Новые производные фичи не должны быть полностью пустыми"""
    app = make_mock_app()
    bureau = make_mock_bureau()
    result = build_features(app, bureau)
    new_cols = ['AGE_YEARS', 'ANNUITY_INCOME_RATIO', 
                'CREDIT_INCOME_RATIO', 'EXT_SOURCE_MEAN']
    for col in new_cols:
        assert result[col].notna().any(), f"{col} полностью пустая"


def test_feature_cols_present():
    """Все фичи из FEATURE_COLS есть в результате"""
    app = make_mock_app()
    bureau = make_mock_bureau()
    result = build_features(app, bureau)
    for col in FEATURE_COLS:
        assert col in result.columns, f"Фича {col} отсутствует в результате"


def test_days_employed_anomaly_removed():
    """Аномалия 365243 в DAYS_EMPLOYED должна стать NaN"""
    app = make_mock_app()
    app['DAYS_EMPLOYED'] = 365243
    bureau = make_mock_bureau()
    result = build_features(app, bureau)
    assert result['DAYS_EMPLOYED'].isna().all(), "Аномалия 365243 не убрана"