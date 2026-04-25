# -*- coding: utf-8 -*-
"""
Módulo para geração de gráficos e visualizações.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import os
from config import REPORTS_DIR

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Gera e salva a matriz de confusão.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(models, X_test, y_test):
    """
    Gera e salva a curva ROC para todos os modelos.

    Args:
        models (dict): Dicionário com os modelos
        X_test: Features de teste
        y_test: Target de teste
    """
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Comparação de Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(models, X_test, y_test):
    """
    Gera e salva a curva Precision-Recall para todos os modelos.

    Args:
        models (dict): Dicionário com os modelos
        X_test: Features de teste
        y_test: Target de teste
    """
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {average_precision_score(y_test, y_pred_proba):.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall - Comparação de Modelos')
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, X_test, model_name):
    """
    Gera e salva o gráfico de importância das features.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        model_name: Nome do modelo
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # Para modelos com feature_importances_ (RandomForest, XGBoost)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Para modelos lineares
            importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
        else:
            # Para pipelines, tentar extrair do último step
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                elif hasattr(classifier, 'coef_'):
                    importances = abs(classifier.coef_[0]) if len(classifier.coef_.shape) > 1 else abs(classifier.coef_)
                else:
                    print(f"Modelo {model_name} não suporta importância de features")
                    return
            else:
                print(f"Modelo {model_name} não suporta importância de features")
                return

        # Criar DataFrame para ordenar
        feature_names = X_test.columns
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title(f'Importância das Features - Top 20 - {model_name}')
        plt.xlabel('Importância')
        plt.ylabel('Feature')

        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Erro ao gerar importância de features: {e}")