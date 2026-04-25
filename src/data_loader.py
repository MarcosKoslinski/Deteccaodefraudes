# -*- coding: utf-8 -*-
"""
Módulo para carregamento e validação dos dados.
"""

import pandas as pd
import sys
import os

def load_data(url):
    """
    Carrega o dataset de fraudes em cartões de crédito da URL fornecida.

    Args:
        url (str): URL do arquivo CSV

    Returns:
        pd.DataFrame: DataFrame com os dados carregados

    Raises:
        ValueError: Se a coluna 'Class' não existir ou houver valores nulos
    """
    try:
        print(f"Carregando dados de: {url}")
        df = pd.read_csv(url)

        # Validações
        if 'Class' not in df.columns:
            raise ValueError("Coluna 'Class' não encontrada no dataset")

        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contém valores nulos")

        print(f"Dataset carregado com sucesso!")
        print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"Distribuição da classe alvo:")
        print(df['Class'].value_counts(normalize=True) * 100)

        return df

    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        sys.exit(1)