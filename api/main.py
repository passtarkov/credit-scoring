import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FEATURE_COLS
from api.schemas import ClientData, ScoreResponse, HealthResponse

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружаем модель один раз при старте сервера.
    Не при каждом запросе — это было бы медленно.
    """
    global model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'lgb_model.pkl'
    )
    model = joblib.load(model_path)
    print(f"Модель загружена из {model_path}")
    yield
    print("Сервер остановлен")


app = FastAPI(
    title="Credit Scoring API",
    description="ML модель для оценки кредитного риска",
    version="1.0.0",
    lifespan=lifespan
)


def prepare_features(client: ClientData) -> pd.DataFrame:
    """
    Превращает данные клиента в датафрейм с нужными фичами.
    Вычисляет производные фичи и применяет label encoding.
    """
    data = client.model_dump()

    days_birth = data.get('DAYS_BIRTH') or 0
    days_employed = data.get('DAYS_EMPLOYED') or 0
    amt_annuity = data.get('AMT_ANNUITY') or 0
    amt_income = data.get('AMT_INCOME_TOTAL') or 1
    amt_credit = data.get('AMT_CREDIT') or 0
    amt_goods = data.get('AMT_GOODS_PRICE') or 1
    ext1 = data.get('EXT_SOURCE_1')
    ext2 = data.get('EXT_SOURCE_2')
    ext3 = data.get('EXT_SOURCE_3')

    data['AGE_YEARS'] = -days_birth / 365
    data['YEARS_EMPLOYED'] = -days_employed / 365
    data['ANNUITY_INCOME_RATIO'] = amt_annuity / amt_income
    data['CREDIT_INCOME_RATIO'] = amt_credit / amt_income
    data['CREDIT_GOODS_RATIO'] = amt_credit / amt_goods

    ext_vals = [v for v in [ext1, ext2, ext3] if v is not None]
    data['EXT_SOURCE_MEAN'] = np.mean(ext_vals) if ext_vals else np.nan
    data['EXT_SOURCE_PROD'] = (ext1 * ext2 * ext3
                                if all(v is not None for v in [ext1, ext2, ext3])
                                else np.nan)

    gender_map = {'M': 1, 'F': 0, 'XNA': 2}
    contract_map = {'Cash loans': 0, 'Revolving loans': 1}

    data['CODE_GENDER'] = gender_map.get(data.get('CODE_GENDER', ''), 0)
    data['NAME_CONTRACT_TYPE'] = contract_map.get(
        data.get('NAME_CONTRACT_TYPE', ''), 0
    )

    for col in ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE',
                'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
        data[col] = 0

    df = pd.DataFrame([data])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLS].astype(float)
    return df



def get_risk_level(score: float) -> str:
    """Конвертирует скор в категорию риска."""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


@app.get("/health", response_model=HealthResponse)
async def health():
    """Проверка что сервер и модель работают."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/score", response_model=ScoreResponse)
async def score(client: ClientData):
    """
    Основной endpoint — принимает данные клиента,
    возвращает вероятность дефолта.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена"
        )

    try:
        X = prepare_features(client)

        score_value = float(model.predict_proba(X)[:, 1][0])
        risk_level = get_risk_level(score_value)

        return ScoreResponse(
            score=round(score_value, 4),
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Credit Scoring API", "docs": "/docs"}