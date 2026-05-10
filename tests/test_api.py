import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app, raise_server_exceptions=True)

def test_health():
    """Health endpoint возвращает 200 и модель загружена"""
    with TestClient(app) as c:
        response = c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_score_good_client():
    """Хороший клиент получает низкий скор"""
    with TestClient(app) as c:
        response = c.post("/score", json={
            "EXT_SOURCE_1": 0.7,
            "EXT_SOURCE_2": 0.8,
            "EXT_SOURCE_3": 0.75,
            "AMT_CREDIT": 300000,
            "AMT_ANNUITY": 15000,
            "AMT_INCOME_TOTAL": 200000,
            "DAYS_BIRTH": -18000,
            "DAYS_EMPLOYED": -5000,
            "CODE_GENDER": "F"
        })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "risk_level" in data
    assert 0 <= data["score"] <= 1
    assert data["risk_level"] == "LOW"


def test_score_bad_client():
    """Плохой клиент получает высокий скор"""
    with TestClient(app) as c:
        response = c.post("/score", json={
            "EXT_SOURCE_1": 0.05,
            "EXT_SOURCE_2": 0.05,
            "EXT_SOURCE_3": 0.05,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 45000,
            "AMT_INCOME_TOTAL": 60000,
            "DAYS_BIRTH": -8000,
            "DAYS_EMPLOYED": -200,
            "CODE_GENDER": "M"
        })
    assert response.status_code == 200
    data = response.json()
    assert data["score"] >= 0.4
    assert data["risk_level"] in ["MEDIUM", "HIGH"]


def test_score_minimal_data():
    """API работает даже если переданы только некоторые поля"""
    with TestClient(app) as c:
        response = c.post("/score", json={
            "EXT_SOURCE_2": 0.5,
            "AMT_INCOME_TOTAL": 100000,
        })
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["score"] <= 1


def test_score_empty_request():
    """Пустой запрос возвращает скор (все поля опциональны)"""
    with TestClient(app) as c:
        response = c.post("/score", json={})
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["score"] <= 1


def test_score_response_format():
    """Формат ответа соответствует схеме"""
    with TestClient(app) as c:
        response = c.post("/score", json={"EXT_SOURCE_2": 0.6})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["score"], float)
    assert isinstance(data["risk_level"], str)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]