# Predição de Phase-Out de SKUs

## Visão Geral

Este projeto foi desenvolvido com de identificar padrões que antecedem o declínio de produtos (SKUs) e desenvolver um modelo preditivo capaz de antecipar quais SKUs entrarão em fase de descontinuação (phase-out) nos próximos 25 ciclos, permitindo ações preventivas para redução de estoque, preservação de margem e melhoria da gestão do portfólio. 

---

## 🎯 Problema de Negócio

Atualmente, muitos SKUs são identificados como candidatos à descontinuação apenas quando já apresentam queda significativa de desempenho comercial.

Esse atraso gera impactos relevantes:

- Acúmulo de estoque
- Redução de giro
- Necessidade de descontos agressivos
- Perda de margem
- Ineficiência na gestão do portfólio

O objetivo deste projeto foi construir um sistema de alerta antecipado para identificar sinais de declínio antes que a deterioração comercial se torne evidente. 

---

## Objetivos do Projeto

### 1. Análise Exploratória

Identificar padrões históricos associados ao declínio dos SKUs através da análise de:

- Receita
- Expansão de pontos de venda
- Conversão
- Ruptura
- Marketing
- Ciclo de vida dos produtos

### 2. Modelagem Preditiva

Desenvolver um modelo capaz de prever:

> Se um SKU entrará em phase-out nos próximos 25 ciclos.

---

## Principais Insights Encontrados

As análises exploratórias mostraram que produtos próximos ao phase-out apresentam padrões consistentes de deterioração.

Principais sinais encontrados:

✅ Queda acelerada da receita

✅ Redução da expansão de novos PDVs

✅ Redução da base total de PDVs

✅ Diminuição do momentum de vendas

✅ Aumento da incidência de ruptura

✅ Redução gradual da conversão

Esses comportamentos começam a aparecer vários ciclos antes da descontinuação efetiva do SKU. 

---

## Engenharia de Variáveis

Foram desenvolvidas variáveis para capturar o comportamento temporal dos produtos.

Exemplos:

### Ciclo de Vida

- tempo_atual
- idade do SKU
- distância até o phase-out

### Receita

- Receita corrigida
- Receita diária
- Receita por PDV

### Momentum

- Momentum da receita
- Momentum da base de PDVs
- Momentum de novos PDVs
- Momentum da conversão
- Momentum da ruptura
- Momentum de investimento em marketing

### Eficiência Comercial

- Receita por PDV
- Novos PDVs sobre total de PDVs
- Marketing sobre receita

As variáveis foram projetadas para capturar sinais de enfraquecimento comercial ao longo do ciclo de vida do SKU. 

---

## Modelo Preditivo

Foi desenvolvido um modelo supervisionado, Regressão Logística, para prever a ocorrência de phase-out nos próximos 25 ciclos.

### Variável Alvo

```text
target_phaseout_25
```

Valor:

- 1 → SKU entrará em phase-out nos próximos 25 ciclos
- 0 → SKU não entrará em phase-out nos próximos 25 ciclos

---

## 📈 Resultados Obtidos

Modelo treinado após remoção de variáveis com potencial vazamento de informação (data leakage).

### Métricas

| Métrica | Resultado |
|----------|------------|
| ROC-AUC | 0.7225 |
| Recall | 60.96% |
| Precision | 63% |
| F1-Score | 62% |
| PR-AUC | 0.6380 |
| Baseline PR-AUC | 0.4295 |
| Ganho sobre baseline | 1.49x |

---

## Interpretação de Negócio

Os resultados indicam que:

- O modelo consegue identificar aproximadamente **61% dos futuros phase-outs** antes que ocorram.
- Quando o modelo gera um alerta, ele está correto em aproximadamente **63% dos casos**.
- A capacidade de priorização é **49% superior à seleção aleatória**.

Em termos práticos:

> Para cada 100 SKUs que entrarão em declínio, o modelo é capaz de antecipar cerca de 61 deles, permitindo que o negócio tome ações preventivas para reduzir perdas financeiras.

---

## 📁 Estrutura do Projeto

```text
├── README.md
│
├── notebooks/
│   ├── case_boticario.ipynb
│
├── data/
│   ├── base-de-dados.xlsx
│   ├── df_train.parquet
│   ├── df_train.parquet
│   ├── bloco_1_compreensao_ciclo_vida.xlsx
│   ├── bloco_2_antes_phaseout.xlsx
│   ├── bloco_3_fatores_declinio.xlsx
│   ├── hipotese_4_1_janela_25_ciclos.xlsx
│   ├── hipotese_4_2_ranking_features.xlsx
│
├── src/
│   ├── func_util.py
|
└── presentation/
    ├── apresentacao_final.pdf
```

---

## Descrição dos Arquivos

### Notebook Principal

| Arquivo | Descrição |
|----------|------------|
| `case_boticario.ipynb` | Pipeline completo do projeto: coleta, exploração, tratamento, engenharia de variáveis, modelagem e avaliação. |

---

### Bases de Dados

| Arquivo | Descrição |
|----------|------------|
| `base-de-dados.xlsx` | Base original disponibilizada para o case. |
| `df_processado.parquet` | Dataset final após o processamento dos dados. |
| `df_train.parquet` | Dataset final preparado para treinamento do modelo. |
| `df_test.parquet` | Dataset de test sem nenhuma modificação para validação do algoritmo. |

---

### Scripts Auxiliares

| Arquivo | Descrição |
|----------|------------|
| `func_util.py` | Arquivo em python com importação de libs, variáveis/constantes e funções auxiliares para Construção de variáveis derivadas e temporais. |

---

## 🚀 Como Executar

### Executar o Notebook

```bash
jupyter notebook
```

Abra:

```text
case_boticario.ipynb
```

---

## 📚 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## ✅ Conclusão

As análises demonstraram que o declínio dos SKUs não ocorre de forma repentina.

Sinais como desaceleração das vendas, redução da expansão comercial e deterioração da base de PDVs aparecem diversos ciclos antes do phase-out.

O modelo desenvolvido mostrou capacidade de identificar aproximadamente 61% dos futuros phase-outs com um desempenho significativamente superior ao baseline, demonstrando potencial para apoiar decisões de gestão de portfólio de forma preventiva e baseada em dados.
