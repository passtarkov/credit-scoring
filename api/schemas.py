from pydantic import BaseModel, Field
from typing import Optional


class ClientData(BaseModel):
    """
    Входные данные клиента для скоринга.
    Optional — поле может быть None (пропуск).
    Field — добавляет пример для документации.
    """

    EXT_SOURCE_1: Optional[float] = Field(None, example=0.50)
    EXT_SOURCE_2: Optional[float] = Field(None, example=0.60)
    EXT_SOURCE_3: Optional[float] = Field(None, example=0.70)

    AMT_CREDIT: Optional[float] = Field(None, example=500000.0)
    AMT_ANNUITY: Optional[float] = Field(None, example=25000.0)
    AMT_INCOME_TOTAL: Optional[float] = Field(None, example=150000.0)
    AMT_GOODS_PRICE: Optional[float] = Field(None, example=450000.0)

    DAYS_BIRTH: Optional[float] = Field(None, example=-15000.0)
    DAYS_EMPLOYED: Optional[float] = Field(None, example=-2000.0)

    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, example=0.02)

    CODE_GENDER: Optional[str] = Field(None, example="M")
    NAME_CONTRACT_TYPE: Optional[str] = Field(None, example="Cash loans")
    NAME_EDUCATION_TYPE: Optional[str] = Field(None, example="Higher education")
    NAME_INCOME_TYPE: Optional[str] = Field(None, example="Working")
    OCCUPATION_TYPE: Optional[str] = Field(None, example="Laborers")
    ORGANIZATION_TYPE: Optional[str] = Field(None, example="Business Entity Type 3")

    BUREAU_CREDIT_COUNT: Optional[float] = Field(None, example=5.0)
    BUREAU_ACTIVE_CREDIT_COUNT: Optional[float] = Field(None, example=2.0)
    BUREAU_AMT_DEBT_MEAN: Optional[float] = Field(None, example=50000.0)
    BUREAU_AMT_OVERDUE_MAX: Optional[float] = Field(None, example=0.0)
    BUREAU_DAYS_OVERDUE_MEAN: Optional[float] = Field(None, example=0.0)
    BUREAU_AMT_CREDIT_MAX: Optional[float] = Field(None, example=200000.0)


class ScoreResponse(BaseModel):
    """
    Ответ API — риск-скор клиента.
    """
    score: float = Field(..., example=0.12,
                         description="Вероятность дефолта от 0 до 1")
    risk_level: str = Field(..., example="LOW",
                            description="LOW / MEDIUM / HIGH")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool