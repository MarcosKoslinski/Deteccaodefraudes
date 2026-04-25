# -*- coding: utf-8 -*-
"""
Módulo para análise de diferentes thresholds de decisão.
"""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
from config import REPORTS_DIR

def evaluate_thresholds(model, X_test, y_test, thresholds):
    """
    Avalia o modelo com diferentes thresholds de decisão.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        thresholds (list): Lista de thresholds para testar

    Returns:
        pd.DataFrame: Resultados da análise
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in thresholds:
        # Aplicar threshold
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # Calcular métricas
        precision = precision_score(y_test, y_pred_threshold, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred_threshold, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred_threshold, pos_label=1, zero_division=0)

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred_threshold)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })

    # Criar DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)

    # Salvar em CSV
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df_results.to_csv(os.path.join(REPORTS_DIR, 'threshold_analysis.csv'), index=False)

    print("Análise de threshold:")
    print(df_results.to_string(index=False))

    return df_results