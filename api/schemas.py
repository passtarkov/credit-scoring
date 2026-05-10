from pydantic import BaseModel, Field
from typing import Optional


class ClientData(BaseModel):
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    CODE_GENDER: Optional[str] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    BUREAU_CREDIT_COUNT: Optional[float] = None
    BUREAU_ACTIVE_CREDIT_COUNT: Optional[float] = None
    BUREAU_AMT_DEBT_MEAN: Optional[float] = None
    BUREAU_AMT_OVERDUE_MAX: Optional[float] = None
    BUREAU_DAYS_OVERDUE_MEAN: Optional[float] = None
    BUREAU_AMT_CREDIT_MAX: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.7,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 25000,
                "AMT_INCOME_TOTAL": 150000,
                "DAYS_BIRTH": -15000,
                "DAYS_EMPLOYED": -2000,
                "CODE_GENDER": "M"
            }
        }
    }


class ScoreResponse(BaseModel):
    score: float
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool