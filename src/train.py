# -*- coding: utf-8 -*-
"""
Módulo de treinamento dos modelos.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import numpy as np

def train_models(X_train, y_train):
    """
    Treina três modelos diferentes para detecção de fraudes.

    Args:
        X_train (pd.DataFrame): Features de treino
        y_train (pd.Series): Target de treino

    Returns:
        dict: Dicionário com os modelos treinados
    """
    # Calcular scale_pos_weight para XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight para XGBoost: {scale_pos_weight:.2f}")

    # Modelo 1: LogisticRegression com class_weight balanced e StandardScaler
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ))
    ])

    # Modelo 2: RandomForestClassifier com class_weight balanced
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_estimators=100,
        max_depth=10
    )

    # Modelo 3: XGBClassifier com scale_pos_weight
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    # Treinar modelos
    print("Treinando LogisticRegression...")
    lr_pipeline.fit(X_train, y_train)

    print("Treinando RandomForest...")
    rf_model.fit(X_train, y_train)

    print("Treinando XGBoost...")
    xgb_model.fit(X_train, y_train)

    models = {
        'LogisticRegression': lr_pipeline,
        'RandomForest': rf_model,
        'XGBoost': xgb_model
    }

    return models