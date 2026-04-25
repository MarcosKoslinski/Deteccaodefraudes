# Detecção de Fraudes em Cartões de Crédito

Este projeto implementa uma solução completa de machine learning para detectar fraudes em transações de cartão de crédito usando o dataset público disponível em:
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

## Objetivo do Projeto

O objetivo é construir um pipeline profissional que:
- carrega e valida dados diretamente da URL
- desenvolve features relevantes para fraude
- treina e compara diferentes modelos
- usa métricas apropriadas para dados desbalanceados
- gera gráficos, relatórios e explicações do modelo
- salva o melhor modelo treinado em disco

## Contexto do Problema

O dataset contém transações de cartão de crédito com a variável alvo `Class`:
- `0`: transação normal
- `1`: fraude

A classe de fraude é extremamente rara, o que exige cuidado especial: um modelo que sempre prevê "normal" pode atingir alta acurácia, mas não detecta nenhum caso de fraude.

### O que isso significa na prática?

- Acurácia não é uma métrica confiável
- É mais importante encontrar fraudes do que apenas acertar transações normais
- Falsos negativos (fraudes não detectadas) são críticos
- Falsos positivos (transações normais marcadas como fraude) também têm custo, mas são toleráveis até certo ponto

## Etapas do Projeto

### 1. Carregamento e validação de dados

O dataset é lido diretamente da URL para garantir reprodutibilidade e transparência.
A validação verifica:
- presença da coluna `Class`
- ausência de valores nulos
- dimensões do dataset

### 2. Análise exploratória de dados (EDA)

A análise inicial mostra:
- distribuição de classes muito desequilibrada
- características numéricas com diferentes escalas
- necessidade de transformar variáveis de valor extremo

### 3. Engenharia de features

Foi adicionada a feature:
- `Amount_log = np.log1p(Amount)`

Essa transformação reduz a assimetria da variável `Amount`, mantendo a informação original e ajudando o modelo a lidar com valores muito altos.

### 4. Separação treino/teste

A divisão é feita com:
- `test_size=0.3`
- `random_state=42`
- `stratify=y`

Isso garante que a proporção de fraudes seja mantida em treino e teste, evitando enviesamento.

### 5. Treinamento de modelos

Foram treinados três modelos:
1. `LogisticRegression` com `class_weight='balanced'` dentro de um `Pipeline` que usa `StandardScaler`
2. `RandomForestClassifier` com `class_weight='balanced'`
3. `XGBClassifier` com `scale_pos_weight` calculado como a proporção entre classes 0 e 1

### 6. Avaliação de desempenho

Os modelos são comparados usando métricas focadas na classe de fraude:
- recall da classe 1
- precision da classe 1
- f1-score da classe 1
- ROC-AUC
- PR-AUC

A comparação é feita em um relatório consolidado para selecionar o melhor modelo com base no equilíbrio entre recall e precision.

### 7. Análise de threshold

Testamos diferentes limites de classificação (`0.1`, `0.2`, `0.3`, `0.4`, `0.5`) para o melhor modelo.
Essa etapa mostra como a sensibilidade muda conforme alteramos o ponto de corte:
- thresholds menores aumentam recall, mas reduzem precision
- thresholds maiores reduzem recall, mas aumentam precision

### 8. Geração de gráficos

O projeto gera gráficos importantes para análise visual:
- matriz de confusão
- curva ROC
- curva Precision-Recall
- importância de features

### 9. Explicabilidade com SHAP

Usamos SHAP no modelo XGBoost para entender quais features têm maior impacto nas predições de fraude.
Isso ajuda a interpretar o modelo e a validar se as decisões fazem sentido para o negócio.

### 10. Salvamento e relatório final

O melhor modelo é salvo em:
- `models/best_model.joblib`

O relatório gerado inclui:
- comparação de métricas entre modelos
- métricas do melhor modelo
- análise de threshold

## Principais Análises Encontradas

### Sobre o desbalanceamento

O projeto confirma que as fraudes representam apenas uma fração muito pequena dos dados. Por isso, o modelo precisa ser avaliado principalmente pela capacidade de encontrar fraudes, não pela quantidade total de acertos.

### Sobre a transformação do `Amount`

A criação da feature `Amount_log` ajuda a reduzir a influência de valores extremos e a melhorar a estabilidade do treinamento, sem remover a informação original de `Amount`.

### Sobre a comparação de modelos

- `LogisticRegression` oferece uma linha de base interpretável
- `RandomForest` e `XGBoost` tendem a capturar relações não lineares mais complexas
- `XGBoost` é indicado quando há alta assimetria entre classes e muitas features

### Sobre a análise de threshold

A comparação de thresholds revela o tradeoff clássico entre:
- **recall** (capturar mais fraudes)
- **precision** (reduzir falsos positivos)

O melhor threshold depende do nível de tolerância ao risco do negócio.

### Sobre explicabilidade

O uso de SHAP permite identificar quais features mais influenciam a detecção de fraude, o que é útil para:
- validar o modelo
- dar transparência para áreas de negócio
- priorizar ações preventivas

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

O script executa o pipeline completo e salva resultados em `models/` e `reports/`.

## O que esperar dos resultados

- Um modelo treinado e salvo em `models/best_model.joblib`
- Relatório de desempenho em `reports/model_report.txt`
- Análise de threshold em `reports/threshold_analysis.csv`
- Gráficos em `reports/` mostrando desempenho e explicabilidade

## Cuidados contra Vazamento de Dados

- `StandardScaler` é aplicado apenas dentro de `Pipeline`
- O SMOTE (se usado) é aplicado apenas nos dados de treino
- A divisão treino/teste é feita antes de criar features adicionais ou treinar modelos
- Não há uso de dados de teste durante o treinamento

## Notas Finais

Este projeto foi desenvolvido como uma solução educacional baseada no curso da DIO.me. Ele mostra boas práticas para dados desbalanceados, validação de resultados e explicabilidade de modelos em problemas de detecção de fraude.

**Nota**: Em produção, é recomendado adicionar monitoramento contínuo, testes A/B e validações específicas de negócio.

**Nota Educacional**: Este projeto foi desenvolvido para fins educacionais com base no curso da DIO.me, demonstrando técnicas de machine learning para detecção de fraudes em cartões de crédito.