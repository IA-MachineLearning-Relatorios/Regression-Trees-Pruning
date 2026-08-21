# Previsão de Preços de Veículos: Análise Comparativa de Técnicas de Poda em Árvores de Regressão

## 1. Contexto e Problema de Negócio

Em um cenário de compra e venda de veículos, como o de uma concessionária, precificar corretamente os carros é vital para garantir margem de lucro e competitividade.

O desafio analítico é construir um modelo capaz de prever preços com precisão, utilizando características técnicas e comerciais dos veículos, sem criar uma árvore excessivamente complexa e suscetível a overfitting.

---

## 2. Objetivo de Machine Learning

Desenvolver um modelo de regressão capaz de prever o preço sugerido (`MSRP`) de veículos.

O foco principal deste laboratório foi estudar:

* Decision Tree Regression;
* Baseline de previsão;
* RMSE e MAE;
* R²;
* Overfitting e generalização;
* Pré-pruning;
* Pós-pruning;
* `max_depth`;
* `ccp_alpha`;
* Relação entre complexidade estrutural e desempenho.

---

## 3. Dataset

Os dados representam características técnicas e comerciais de veículos.

* **Registros:** 11.914
* **Target:** `MSRP`
* **Target:** preço sugerido pelo fabricante em dólares
* **Features principais:** `Make`, `Model`, `Year`, `Engine HP`, `Engine Cylinders`, `Transmission Type`, entre outras.
* **Fonte:** Kaggle — Car Features and MSRP

O dataset contém variáveis numéricas e categóricas. As variáveis categóricas foram transformadas utilizando One-Hot Encoding.

Os valores ausentes numéricos foram tratados utilizando a mediana calculada exclusivamente sobre o conjunto de treinamento, reduzindo o risco de data leakage.

---

## 4. Separação dos Dados

O dataset foi dividido em:

* **70% treinamento**
* **30% teste**

A mediana utilizada para imputação foi calculada no conjunto de treinamento e posteriormente aplicada ao conjunto de teste.

Da mesma forma, o One-Hot Encoder foi ajustado no treinamento e apenas transformado no teste.

---

## 5. Baseline

Antes do treinamento da árvore, foi estabelecido um baseline utilizando a média do `MSRP` presente no conjunto de treinamento.

**RMSE do Baseline: ~US$ 59.188,58**

Esse valor representa o desempenho de uma estratégia ingênua que simplesmente prevê aproximadamente o mesmo preço para todos os veículos.

O resultado demonstra que a árvore de regressão consegue extrair uma quantidade muito maior de informação das características dos veículos.

---

## 6. Modelo

Foi utilizado:

```python
DecisionTreeRegressor(random_state=42)
```

A árvore foi escolhida por permitir estudar diretamente a relação entre:

**complexidade → overfitting → poda → generalização**

---

# 7. Experimentos de Pruning

## 7.1 Árvore Livre

A árvore foi inicialmente treinada sem restrição explícita de profundidade.

Resultados:

| Métrica |       Treino |        Teste |
| ------- | -----------: | -----------: |
| RMSE    | US$ 5.062,42 | US$ 7.163,33 |
| MAE     | US$ 1.204,53 | US$ 3.097,59 |
| R²      |       0,9930 |       0,9854 |

**Gap de R²:** 0,0076

A árvore apresentou desempenho muito alto tanto no treinamento quanto no teste. A pequena diferença entre os dois conjuntos indica boa generalização neste experimento, embora a estrutura da árvore seja bastante complexa.

---

## 7.2 Pré-pruning

Foi realizado um experimento variando:

```python
max_depth = [5, 10, 15, 20, 25, 30]
```

O objetivo foi observar como a limitação da profundidade influencia o erro e a complexidade da árvore.

A configuração escolhida foi:

```python
max_depth = 15
```

Resultados:

| Métrica |       Treino |        Teste |
| ------- | -----------: | -----------: |
| RMSE    | US$ 5.869,64 | US$ 7.288,13 |
| MAE     |            — |            — |
| R²      |            — |            — |

A árvore apresentou **1.331 folhas**.

Apesar de reduzir consideravelmente a complexidade em relação à árvore livre, o pré-pruning não apresentou o melhor desempenho de teste.

> Observação: os valores de MAE e R² apresentados posteriormente foram calculados para o modelo com `max_depth=5` que estava armazenado na variável `modelo_arvore_podada`, enquanto a análise experimental identificou `max_depth=15` como a melhor configuração entre as profundidades testadas. Por isso, esses resultados não devem ser misturados.

---

## 7.3 Pós-pruning — Cost Complexity Pruning

Foi utilizado:

```python
cost_complexity_pruning_path()
```

para obter diferentes valores de `ccp_alpha`.

O melhor valor observado no experimento foi:

```python
ccp_alpha = 4324.41
```

Resultados finais:

| Métrica |           Treino |            Teste |
| ------- | ---------------: | ---------------: |
| RMSE    | **US$ 5.257,81** | **US$ 7.048,46** |
| MAE     | **US$ 1.847,56** | **US$ 3.009,48** |
| R²      |       **0,9924** |       **0,9858** |

**Gap de R²:** 0,0066

Estrutura final:

* **Profundidade:** 48
* **Folhas:** 940

* ### Evolução da Poda por Custo-Complexidade
<img width="868" height="552" alt="RMSExalpha" src="https://github.com/user-attachments/assets/4402bb5a-75b6-432f-a5c7-1384723fe657" />

---

# 8. Avaliação por R²

O R² foi utilizado para complementar a análise realizada anteriormente com RMSE.

O melhor modelo apresentou:

> **R² Treino = 0,9924**

> **R² Teste = 0,9858**

Isso significa que o modelo consegue explicar aproximadamente **98,58% da variabilidade observada nos preços do conjunto de teste**.

É importante destacar que R² **não representa uma porcentagem de acerto nem uma taxa de erro das previsões**.

Portanto:

> R² = 0,9858 não significa que o modelo "erra 1,42%".

O 1,42% representa aproximadamente a parcela da variabilidade do alvo que não é explicada pelo modelo segundo essa métrica.

---

# 9. MAE vs. RMSE

A inclusão do MAE permitiu interpretar o erro de maneira mais prática.

Para o modelo pós-podado:

* **MAE Teste:** US$ 3.009,48
* **RMSE Teste:** US$ 7.048,46

O MAE indica que, em média, as previsões ficam aproximadamente **US$ 3 mil afastadas do preço real**.

Já o RMSE é consideravelmente maior, indicando a existência de previsões com erros absolutos significativamente maiores.

Essa diferença será investigada na próxima etapa do laboratório.

### Interpretação das métricas

| Métrica              | Pergunta respondida                                            |
| -------------------- | -------------------------------------------------------------- |
| **R²**               | Quanto da variabilidade dos preços o modelo consegue explicar? |
| **MAE**              | Qual é o erro absoluto médio em dólares?                       |
| **RMSE**             | Existem erros grandes que merecem investigação?                |
| **Gap Treino/Teste** | O modelo generaliza ou apresenta sinais de overfitting?        |

---

# 10. Comparação Final

| Modelo       |       RMSE Teste |        MAE Teste |   R² Teste |     Gap R² |  Folhas |
| ------------ | ---------------: | ---------------: | ---------: | ---------: | ------: |
| Árvore Livre |     US$ 7.163,33 |     US$ 3.097,59 |     0,9854 |     0,0076 |   4.392 |
| Pré-pruning* |    US$ 21.174,73 |     US$ 8.684,02 |     0,8720 |     0,0169 |      21 |
| Pós-pruning  | **US$ 7.048,46** | **US$ 3.009,48** | **0,9858** | **0,0066** | **940** |

* Os valores de Pré-pruning desta tabela correspondem ao modelo `max_depth=5` utilizado na avaliação final, enquanto o experimento de profundidades identificou `max_depth=15` como a melhor configuração de pré-pruning.

* ### Eficiência Estrutural: Complexidade vs. Erro
<img width="922" height="548" alt="graficoRMSExCOMPLEXIDADE" src="https://github.com/user-attachments/assets/e21eae38-610b-471e-b9f3-b8a8fb2839b9" />

---

# 11. Conclusão Atual

Os resultados indicam que o **pós-pruning apresentou o melhor desempenho entre os modelos avaliados na configuração final**.

O modelo conseguiu:

* reduzir o RMSE de aproximadamente **US$ 7.163 para US$ 7.048** em relação à árvore livre;
* reduzir o número de folhas de **4.392 para 940**;
* manter um **R² de teste de 0,9858**;
* apresentar um **MAE de aproximadamente US$ 3.009**;
* manter uma diferença pequena entre desempenho de treinamento e teste.

O resultado é particularmente interessante porque a poda reduziu significativamente a complexidade estrutural sem sacrificar o desempenho preditivo.

---

# 12. Próxima Etapa — Investigação dos Erros

Apesar dos resultados positivos, a análise ainda não está encerrada.

O próximo passo será investigar **por que o RMSE (US$ 7.048) é muito maior que o MAE (US$ 3.009)**.

Serão realizadas:

### 12.1 Real vs. Predito

Construção de um gráfico comparando:

```text
Preço Real × Preço Predito
```

O objetivo será verificar visualmente onde o modelo consegue acompanhar os preços e onde começa a apresentar desvios maiores.

### 12.2 Análise dos Resíduos

Será calculado:

```text
resíduo = preço_real - preço_predito
```

e analisada sua distribuição.

### 12.3 Identificação dos maiores erros

Serão identificados os veículos nos quais o modelo apresentou os maiores erros absolutos.

A investigação buscará responder:

* Existem outliers?
* Existem veículos muito caros que o modelo não consegue prever?
* Existem padrões nos erros?
* O modelo subestima ou superestima determinados veículos?
* Quais características estão associadas aos maiores erros?

### 12.4 Diagnóstico Final

A partir dessa análise será possível determinar se o alto RMSE é causado por:

* poucos erros extremos;
* outliers no preço;
* segmentos específicos de veículos;
* limitações da árvore;
* ou características que o dataset não representa adequadamente.

Somente depois dessa investigação será feita a conclusão definitiva sobre a qualidade do modelo.

---

## 13. Próximo Grande Passo

Após finalizar a análise dos resíduos, o laboratório poderá avançar para **Ensemble Learning**, comparando a árvore individual com métodos capazes de combinar múltiplas árvores.

O próximo laboratório deverá explorar conceitos como:

* Bagging;
* Bootstrap;
* Out-of-Bag Error;
* Random Forest;
* redução de variância;
* comparação com a árvore individual.

O objetivo será verificar se combinar múltiplas árvores consegue superar o desempenho atual de aproximadamente **US$ 7.048 de RMSE** mantendo boa capacidade de generalização.
