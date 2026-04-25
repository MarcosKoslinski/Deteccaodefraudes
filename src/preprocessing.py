# -*- coding: utf-8 -*-
"""
Módulo de pré-processamento e criação de features.
"""

import numpy as np
import pandas as pd

def create_features(df):
    """
    Cria features adicionais e separa X e y.

    Args:
        df (pd.DataFrame): DataFrame com os dados brutos

    Returns:
        tuple: (X, y) onde X são as features e y é o target
    """
    # Criar cópia para não modificar o original
    df_processed = df.copy()

    # Criar feature Amount_log
    df_processed['Amount_log'] = np.log1p(df_processed['Amount'])

    # Separar features e target
    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']

    print(f"Feature 'Amount_log' criada")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y