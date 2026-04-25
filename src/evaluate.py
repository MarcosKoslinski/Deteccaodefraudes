# -*- coding: utf-8 -*-
"""
Módulo de avaliação dos modelos.
"""

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score
)
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia um modelo e retorna métricas detalhadas.

    Args:
        model: Modelo treinado
        X_test (pd.DataFrame): Features de teste
        y_test (pd.Series): Target de teste
        model_name (str): Nome do modelo

    Returns:
        dict: Dicionário com as métricas
    """
    # Fazer predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Métricas específicas da classe 1 (fraude)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)

    results = {
        'precision_class_1': precision_1,
        'recall_class_1': recall_1,
        'f1_class_1': f1_1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': report['accuracy'],
        'confusion_matrix': conf_matrix.tolist()
    }

    return results

def compare_models(results):
    """
    Compara os resultados dos modelos e retorna uma tabela.

    Args:
        results (dict): Resultados de evaluate_model para cada modelo

    Returns:
        pd.DataFrame: Tabela de comparação
    """
    comparison_data = {}
    for model_name, metrics in results.items():
        comparison_data[model_name] = {
            'Precision (Classe 1)': metrics['precision_class_1'],
            'Recall (Classe 1)': metrics['recall_class_1'],
            'F1-Score (Classe 1)': metrics['f1_class_1'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Accuracy': metrics['accuracy']
        }

    df_comparison = pd.DataFrame(comparison_data).T
    df_comparison = df_comparison.round(4)

    return df_comparison