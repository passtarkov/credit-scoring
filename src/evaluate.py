import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.features import FEATURE_COLS


def compute_shap_values(model, X_sample: pd.DataFrame):
    """
    Считает SHAP values для выборки клиентов.
    
    Принимает: обученную модель, датафрейм с фичами
    Возвращает: explainer, shap_values (для класса 1)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
        
    return explainer, sv


def plot_summary(sv, X_sample: pd.DataFrame, save_path: str = None):
    """
    Summary plot — глобальная важность фич.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, plot_type='dot', 
                      max_display=15, show=False)
    plt.title('SHAP Summary Plot — влияние фич на риск дефолта')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_waterfall(model, explainer, sv, X_sample: pd.DataFrame,
                   client_idx: int, save_path: str = None):
    """
    Waterfall plot — объяснение для одного клиента.
    """
    shap_explanation = shap.Explanation(
        values=sv[client_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[client_idx],
        feature_names=FEATURE_COLS
    )
    
    pred = model.predict_proba(
        X_sample.iloc[[client_idx]]
    )[:, 1][0]
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, show=False)
    plt.title(f'Клиент — скор риска: {pred:.3f}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dependence(sv, X_sample: pd.DataFrame,
                    feature: str, interaction: str,
                    save_path: str = None):
    """
    Dependence plot — как фича влияет на риск по всей выборке.
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, sv, X_sample,
        interaction_index=interaction,
        show=False
    )
    plt.title(f'Dependence Plot — {feature} vs риск дефолта')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()