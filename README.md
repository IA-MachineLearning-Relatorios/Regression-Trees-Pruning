# Previsão de Preços de Veículos: Uma Análise Comparativa de Técnicas de Poda em Árvores de Regressão

## 1. Contexto e Problema de Negócio
Em um cenário de compra e venda de veículos (como o de uma concessionária), precificar corretamente os carros é vital para garantir a margem de lucro e a competitividade. O grande desafio analítico é criar regras de precificação precisas baseadas nos atributos técnicos e visuais dos veículos, sem que essas regras se tornem "viciadas" apenas no histórico conhecido (o que causaria perdas financeiras ao avaliar carros inéditos).

## 2. Objetivo de Machine Learning
Desenvolver um modelo de regressão capaz de prever o preço sugerido (MSRP) de veículos. O foco primário deste projeto é **comparar experimentalmente duas técnicas de regularização (Pré-pruning e Pós-pruning)** para controlar o *overfitting* em algoritmos baseados em árvores, avaliando não apenas o erro financeiro, mas também a eficiência estrutural das regras geradas.

## 3. Dataset
Os dados utilizados representam características de mercado e especificações técnicas de veículos.
* **Tamanho:** 11.914 registros.
* **Target:** `MSRP` (Manufacturer's Suggested Retail Price - Preço Sugerido em Dólares).
* **Features Principais:** `Make` (Marca), `Model` (Modelo), `Year` (Ano), `Engine HP` (Potência do Motor) e `Transmission Type` (Tipo de Transmissão).
* **Fonte:** [Kaggle: Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset)

## 4. Baseline
Para estabelecer uma base de comparação do valor real gerado pelo algoritmo, o modelo de referência inicial adotado foi uma previsão ingênua pela média histórica.
* **RMSE Baseline:** ~$ 59.188

## 5. Modelo e Estratégia
**Algoritmo utilizado:** `DecisionTreeRegressor` (Scikit-Learn). 
A escolha das Árvores de Decisão se deu pela sua interpretabilidade estrutural (nós e folhas transparentes) e por ser o ambiente ideal para a aplicação direta e o estudo isolado dos métodos de poda (Pruning).

## 6. Experimentos e Resultados
O projeto testou a evolução do erro de generalização em relação à complexidade da árvore construída.

1. **Árvore Livre (Baseline de Overfitting):** 
   O modelo decorou os dados de treino, gerando regras excessivas sem ganho no mundo real.
   * RMSE Treino: ~$ 5.091 | RMSE Teste: ~$ 7.315 | Total de Folhas: 4.392

2. **Pré-pruning (Teste de `max_depth`):** 
   A melhor configuração observada nos experimentos limitou o crescimento da árvore rigidamente.
   * Configuração Ótima: `max_depth = 15`
   * RMSE Teste: ~$ 7.288 | Total de Folhas: 1.331

3. **Pós-pruning (Teste de `ccp_alpha`):** 
   Utilizando o método `cost_complexity_pruning_path`, a melhor configuração observada agiu dinamicamente podando ramos ineficientes.
   * Configuração Ótima: `ccp_alpha = 4324.41`
   * RMSE Teste: ~$ 7.048 | Total de Folhas: 940 | Profundidade Máxima: 48

### Evolução da Poda por Custo-Complexidade
![Evolução do Erro vs Alpha](images/image_e1b04d.png)

## 7. Análise e Conclusão
A configuração utilizando **Pós-pruning** apresentou o melhor desempenho geral nos experimentos realizados. Além de reduzir o erro financeiro médio nas previsões, a técnica destacou-se pela sua **eficiência estrutural**. 

Enquanto o Pré-pruning limitou a árvore como um "teto" rígido no nível 15 (criando 1.331 folhas), o Pós-pruning permitiu que a árvore crescesse até o nível 48 para explicar casos pontuais e complexos, mas podou agressivamente regras rasas e ruidosas. O resultado foi um modelo mais enxuto (940 folhas), muito mais inteligente na alocação de regras e com melhor capacidade de generalização para novos dados.

### Eficiência Estrutural: Complexidade vs. Erro
![Comparação de Eficiência Estrutural](images/image_e1b04a.png)

## 8. Limitações e Próximos Passos
* **Limitações:** O modelo escolhido representa a melhor configuração observada neste espaço de busca específico. No entanto, Árvores de Decisão individuais, por natureza matemática, são propensas a instabilidades caso o conjunto de dados sofra grandes variações.
* **Próximos Passos:** Evoluir a arquitetura da solução implementando métodos de *Ensemble* (como Random Forest ou Gradient Boosting) para verificar se a combinação de múltiplas árvores reduz ainda mais o erro financeiro mantendo a robustez encontrada através do controle de complexidade.
