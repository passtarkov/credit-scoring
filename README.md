---
title: Credit Scoring
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Credit Scoring Pipeline

ML pipeline для оценки кредитного риска на основе датасета 
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

## Результаты

| Модель | ROC-AUC |
|--------|---------|
| Logistic Regression (baseline) | 0.7258 |
| LogReg + Feature Engineering | 0.7407 |
| LightGBM (default) | 0.7568 |
| LightGBM + Optuna | **0.7611** |

## Стек

- **ML:** LightGBM, Scikit-learn, Optuna
- **Объяснимость:** SHAP (summary, waterfall, dependence plots)
- **API:** FastAPI, Pydantic, Uvicorn
- **Инфраструктура:** Docker
- **Тесты:** pytest (11 тестов)

## Структура проекта

    credit-scoring/
    ├── data/
    │   ├── raw/                  # application_train.csv, bureau.csv
    │   └── processed/            # обработанные данные, модель
    ├── notebooks/
    │   ├── 01_eda.ipynb          # EDA + baseline LogReg
    │   ├── 02_features.ipynb     # Feature engineering
    │   ├── 03_models.ipynb       # LightGBM + Optuna
    │   └── 04_shap.ipynb         # SHAP анализ
    ├── src/
    │   ├── features.py           # Feature engineering pipeline
    │   ├── train.py              # Обучение модели
    │   └── evaluate.py           # Метрики и SHAP
    ├── api/
    │   ├── main.py               # FastAPI приложение
    │   └── schemas.py            # Pydantic схемы
    ├── tests/
    │   ├── test_features.py      # 5 тестов
    │   └── test_api.py           # 6 тестов
    ├── reports/figures/          # SHAP графики
    ├── Dockerfile
    └── requirements.txt
## Ключевые решения

**Метрика — ROC-AUC вместо accuracy**  
Датасет несбалансирован (8% дефолтов). Модель предсказывающая всегда 0 
даёт accuracy 92% — это бесполезно. ROC-AUC измеряет качество ранжирования 
клиентов по риску независимо от порога.

**Feature Engineering**  
Агрегация bureau.csv (1.7М строк → одна строка на клиента), 
производные фичи (ANNUITY_INCOME_RATIO, EXT_SOURCE_MEAN и др.) 
дали прирост +0.015 ROC-AUC.

**LightGBM вместо LogReg**  
Нелинейные зависимости в кредитном скоринге — LightGBM улавливает 
взаимодействия между фичами которые линейная модель не видит. 
Прирост +0.020 ROC-AUC.

**SHAP для объяснимости**  
Банковский регулятор требует объяснения решений модели. SHAP даёт 
локальное объяснение для каждого клиента (waterfall plot) и 
глобальную картину важности фич (summary plot).

## SHAP Summary Plot

![SHAP Summary](reports/figures/shap_summary.png)

## Быстрый старт

### Локально

```bash
git clone https://github.com/passtarkov/credit-scoring
cd credit-scoring
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
docker build -t credit-scoring .
docker run -p 8000:8000 credit-scoring
```

### Запрос к API

```bash
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, 
          "EXT_SOURCE_3": 0.7, "AMT_INCOME_TOTAL": 150000,
          "AMT_CREDIT": 500000, "DAYS_BIRTH": -15000}'
```

Ответ:
```json
{"score": 0.0595, "risk_level": "LOW"}
```

### Тесты

```bash
pytest tests/ -v
```

## Что узнал

- Полный ML pipeline от EDA до задеплоенного API
- Работа с несбалансированными классами в кредитном скоринге
- Feature engineering на реальных банковских данных
- Байесовская оптимизация гиперпараметров через Optuna
- Объяснимость ML моделей через SHAP
- Упаковка ML сервиса в Docker контейнер