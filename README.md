# Detecção de Fraudes em Cartões de Crédito

Este projeto implementa uma solução completa de machine learning para detecção de fraudes em transações de cartões de crédito, utilizando o dataset público disponível em: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Contexto do Problema

O dataset contém transações de cartões de crédito, onde a variável alvo "Class" indica:
- 0: Transação normal
- 1: Fraude

O problema é altamente desbalanceado, com aproximadamente 99,83% das transações sendo normais e apenas 0,17% sendo fraudes. Isso torna a acurácia uma métrica inadequada, pois um modelo que classifica tudo como normal teria alta acurácia, mas falharia completamente em detectar fraudes.

## Métricas Principais

Devido ao desbalanceamento, priorizamos métricas focadas na classe minoritária (fraude):
- **Recall da classe 1**: Capacidade de identificar fraudes (sensibilidade)
- **Precision da classe 1**: Precisão nas classificações de fraude
- **F1-Score da classe 1**: Equilíbrio entre precision e recall
- **ROC-AUC**: Área sob a curva ROC
- **PR-AUC**: Área sob a curva Precision-Recall (mais adequada para dados desbalanceados)
- **Matriz de Confusão**: Para visualizar verdadeiros positivos, falsos positivos, etc.
- **Curvas ROC e Precision-Recall**: Para análise visual do desempenho

## Estrutura do Projeto

```
fraud-detection-project/
│
├── README.md                    # Esta documentação
├── requirements.txt             # Dependências do projeto
├── main.py                      # Script principal para executar o pipeline
├── .gitignore                   # Arquivos a ignorar no Git
│
├── data/                        # Dados (vazio inicialmente)
│   └── .gitkeep
│
├── models/                      # Modelos treinados salvos
│   └── .gitkeep
│
├── reports/                     # Relatórios e gráficos gerados
│   └── .gitkeep
│
├── src/                         # Código fonte modular
│   ├── __init__.py
│   ├── config.py                # Configurações e constantes
│   ├── data_loader.py           # Carregamento e validação dos dados
│   ├── preprocessing.py         # Criação de features e separação X/y
│   ├── train.py                 # Treinamento dos modelos
│   ├── evaluate.py              # Avaliação e comparação de modelos
│   ├── plots.py                 # Geração de gráficos
│   ├── threshold.py             # Análise de thresholds
│   └── explainability.py        # Explicabilidade com SHAP
│
└── notebooks/                   # Notebooks para análise exploratória
    └── 01_exploratory_analysis.ipynb
```

## Instalação e Execução

### 1. Criar Ambiente Virtual

```bash
python -m venv .venv
```

### 2. Ativar Ambiente Virtual

**No Windows:**
```bash
.venv\Scripts\activate
```

**No Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o Projeto

```bash
python main.py
```

O script executará todo o pipeline automaticamente, gerando:
- Modelos treinados em `models/`
- Relatórios em `reports/`
- Gráficos e análises

## Interpretação dos Resultados

Após a execução, verifique:
- **Relatório em `reports/model_report.txt`**: Comparação detalhada dos modelos
- **Gráficos em `reports/`**: Matrizes de confusão, curvas ROC/PR, importância de features, SHAP
- **Análise de threshold em `reports/threshold_analysis.csv`**: Melhor threshold para o modelo selecionado

O modelo com melhor F1-score ou PR-AUC será salvo como `models/best_model.joblib`.

## Cuidados contra Vazamento de Dados

- O `StandardScaler` é aplicado apenas dentro de Pipelines, evitando vazamento entre treino e teste
- O SMOTE (se usado) é aplicado apenas nos dados de treino
- A separação treino/teste é feita antes de qualquer pré-processamento
- Features são criadas apenas com dados disponíveis no momento da predição

## Modelos Implementados

1. **LogisticRegression** com `class_weight="balanced"`
2. **RandomForestClassifier** com `class_weight="balanced"`
3. **XGBClassifier** com `scale_pos_weight` calculado automaticamente

## Análise de Threshold

Testamos thresholds de 0.1 a 0.5 para encontrar o melhor equilíbrio entre recall e precision da classe fraude.

## Explicabilidade

Utilizamos SHAP para explicar as predições do modelo XGBoost, gerando gráficos de importância global das features.

## Próximos Passos

- Implementar validação cruzada estratificada
- Testar outros algoritmos (LightGBM, CatBoost)
- Adicionar feature engineering mais avançado
- Implementar deploy em produção (API Flask/FastAPI)
- Monitoramento contínuo e retraining automático
- Análise de custo-benefício das decisões de threshold

## Referências

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Técnicas para dados desbalanceados: SMOTE, class weights, etc.
- Métricas apropriadas: ROC-AUC, PR-AUC, F1-score

---

**Nota**: Este projeto serve como base sólida para detecção de fraudes. Em produção, considere auditorias de segurança, compliance e testes extensivos.

**Nota Educacional**: Este projeto foi desenvolvido para fins educacionais com base no curso da DIO.me, demonstrando técnicas de machine learning para detecção de fraudes em cartões de crédito.