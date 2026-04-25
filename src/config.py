# -*- coding: utf-8 -*-
"""
Módulo de configuração do projeto.
Define constantes e caminhos utilizados no projeto.
"""

import os

# URL do dataset
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Semente para reprodutibilidade
RANDOM_STATE = 42

# Caminhos das pastas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DATA_DIR = os.path.join(BASE_DIR, 'data')