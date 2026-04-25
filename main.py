# -*- coding: utf-8 -*-
"""
Arquivo principal do projeto de detecção de fraudes.
Executa o pipeline completo de machine learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import DATA_URL, RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from data_loader import load_data
from preprocessing import create_features
from sklearn.model_selection import train_test_split
from train import train_models
from evaluate import evaluate_model, compare_models
from plots import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)
from threshold import evaluate_thresholds
from explainability import generate_shap_explanation
import joblib
import pandas as pd

def main():
    """Executa o pipeline completo do projeto."""
    print("=== Iniciando Projeto de Detecção de Fraudes ===\n")

    # 1. Carregar dados
    print("1. Carregando dados...")
    df = load_data(DATA_URL)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas\n")

    # 2. Criar features
    print("2. Criando features...")
    X, y = create_features(df)
    print(f"Features criadas: X shape {X.shape}, y distribution: {y.value_counts().to_dict()}\n")

    # 3. Dividir treino/teste
    print("3. Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras\n")

    # 4. Treinar modelos
    print("4. Treinando modelos...")
    models = train_models(X_train, y_train)
    print(f"Modelos treinados: {list(models.keys())}\n")

    # 5. Avaliar modelos
    print("5. Avaliando modelos...")
    results = {}
    for name, model in models.items():
        print(f"Avaliando {name}...")
        results[name] = evaluate_model(model, X_test, y_test, name)

    # 6. Comparar modelos
    print("6. Comparando modelos...")
    comparison = compare_models(results)
    print(comparison)

    # 7. Selecionar melhor modelo (baseado em F1 da classe 1)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_class_1'])
    best_model = models[best_model_name]
    print(f"Melhor modelo selecionado: {best_model_name} (F1 classe 1: {results[best_model_name]['f1_class_1']:.4f})\n")

    # 8. Análise de threshold
    print("8. Analisando thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_results = evaluate_thresholds(best_model, X_test, y_test, thresholds)
    print("Análise de threshold salva em reports/threshold_analysis.csv\n")

    # 9. Salvar melhor modelo
    print("9. Salvando melhor modelo...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
    print("Modelo salvo em models/best_model.joblib\n")

    # 10. Gerar relatório
    print("10. Gerando relatório...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'model_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE DETECÇÃO DE FRAUDES ===\n\n")
        f.write(f"Melhor modelo: {best_model_name}\n\n")
        f.write("COMPARAÇÃO DE MODELOS:\n")
        f.write(comparison.to_string())
        f.write("\n\nDETALHES DO MELHOR MODELO:\n")
        for metric, value in results[best_model_name].items():
            f.write(f"{metric}: {value}\n")
    print("Relatório salvo em reports/model_report.txt\n")

    # 11. Gerar gráficos
    print("11. Gerando gráficos...")
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)
    plot_roc_curve(models, X_test, y_test)
    plot_precision_recall_curve(models, X_test, y_test)
    plot_feature_importance(best_model, X_test, best_model_name)
    print("Gráficos salvos em reports/\n")

    # 12. Explicabilidade com SHAP
    print("12. Gerando explicabilidade com SHAP...")
    try:
        generate_shap_explanation(best_model, X_test)
        print("Análise SHAP salva em reports/shap_summary.png\n")
    except Exception as e:
        print(f"Erro na análise SHAP: {e}\n")

    print("=== Projeto concluído com sucesso! ===")

if __name__ == "__main__":
    main()