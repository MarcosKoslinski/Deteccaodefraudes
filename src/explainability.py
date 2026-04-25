# -*- coding: utf-8 -*-
"""
Módulo para explicabilidade do modelo usando SHAP.
"""

import shap
import matplotlib.pyplot as plt
import os
from config import REPORTS_DIR

def generate_shap_explanation(model, X_test):
    """
    Gera explicação SHAP para o modelo (focado em XGBoost).

    Args:
        model: Modelo treinado
        X_test: Features de teste
    """
    try:
        # Verificar se é XGBoost
        if not str(type(model)).endswith("XGBClassifier'>"):
            print("SHAP implementado apenas para XGBoost. Pulando análise.")
            return

        print("Gerando explicação SHAP...")

        # Criar explainer
        explainer = shap.TreeExplainer(model)

        # Calcular SHAP values para uma amostra
        shap_values = explainer.shap_values(X_test)

        # Gráfico de resumo
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP Summary Plot - XGBoost')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Explicação SHAP gerada com sucesso.")

    except ImportError:
        print("Biblioteca SHAP não instalada. Instale com: pip install shap")
    except Exception as e:
        print(f"Erro na geração de SHAP: {e}")
        print("Continuando sem análise SHAP...")