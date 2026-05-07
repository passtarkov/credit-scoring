import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.features import build_features, FEATURE_COLS


def train_model(app_path: str, bureau_path: str, model_path: str) -> dict:
    """
    Полный pipeline обучения модели.
    
    Принимает: пути к данным и куда сохранить модель
    Возвращает: словарь с метриками
    """
    app = pd.read_csv(app_path)
    bureau = pd.read_csv(bureau_path)

    df = build_features(app, bureau)
    X = df[FEATURE_COLS]
    y = df['TARGET']

    best_params = {
        'n_estimators': 790,
        'learning_rate': 0.042732320475249455,
        'num_leaves': 94,
        'max_depth': 5,
        'min_child_samples': 199,
        'subsample': 0.902701082869012,
        'colsample_bytree': 0.9408285784806656,
        'reg_alpha': 9.63883769942733,
        'reg_lambda': 2.1182972896884963e-05,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        lgb.LGBMClassifier(**best_params),
        X, y, cv=cv, scoring='roc_auc', n_jobs=-1
    )

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)

    return {
        'cv_roc_auc_mean': scores.mean(),
        'cv_roc_auc_std': scores.std(),
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
    }


if __name__ == '__main__':
    metrics = train_model(
        app_path='data/raw/application_train.csv',
        bureau_path='data/raw/bureau.csv',
        model_path='data/processed/lgb_model.pkl'
    )