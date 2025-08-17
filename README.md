# Análise e Predição de Churn de Clientes - TelecomX

## Propósito da Análise

Este projeto tem como objetivo principal analisar e prever o churn (evasão) de clientes da empresa de telecomunicações TelecomX. Através da aplicação de técnicas de Ciência de Dados e Machine Learning, buscamos:

*   Identificar os principais fatores e características dos clientes que levam à evasão.
*   Desenvolver modelos preditivos capazes de classificar clientes em risco de churn.
*   Fornecer insights acionáveis para a criação de estratégias de retenção eficazes.

## Estrutura do Projeto e Organização dos Arquivos

O projeto está organizado da seguinte forma:

```
.
├── TelecomX_Data.json             # Dados brutos dos clientes da TelecomX
├── telecomx_processed_data.csv    # Dados pré-processados após limpeza e transformação
├── model_results.json             # Resultados da avaliação dos modelos preditivos
├── churn_proportion.png           # Gráfico da proporção de churn
├── correlation_matrix.png         # Matriz de correlação das variáveis
├── tenure_vs_churn.png            # Boxplot de tempo de contrato vs. churn
├── total_charges_vs_churn.png     # Boxplot de gastos totais vs. churn
├── rf_feature_importance.png      # Gráfico de importância de features do Random Forest
├── TelecomX_Churn_Analysis_Full.ipynb # Notebook Jupyter/Colab com toda a análise
├── relatorio_churn_telecomx.md    # Relatório detalhado da análise e estratégias
├── README.md                      # Este arquivo
├── load_and_preprocess.py         # Script Python para carregamento e pré-processamento de dados
├── eda_and_visualizations.py      # Script Python para EDA e visualizações
├── model_training_and_evaluation.py # Script Python para treinamento e avaliação de modelos
├── feature_importance_and_comparison.py # Script Python para análise de importância de features e comparação de modelos
└── create_telecomx_full_notebook.py # Script Python para gerar o notebook completo
```

## Exemplos de Gráficos e Insights Obtidos

Durante a Análise Exploratória de Dados (EDA) e a modelagem, foram gerados diversos gráficos e insights. Alguns exemplos incluem:

*   **Proporção de Churn**: Visualização da distribuição de clientes que evadiram vs. não evadiram, mostrando o desbalanceamento da classe.
*   **Matriz de Correlação**: Identificação de relações entre as variáveis, destacando aquelas com maior impacto no churn.
*   **Tempo de Contrato vs. Churn**: Boxplots que revelam que clientes com menor tempo de contrato (tenure) têm maior probabilidade de churn.
*   **Importância das Features (Random Forest)**: Gráfico de barras mostrando as variáveis mais importantes para a predição de churn, como `customer.tenure`, `account.Charges.Total`, `account.Contract_Month-to-month`, entre outras.

**Insights Chave:**

*   O churn é significativamente influenciado pelo tempo de contrato do cliente e pelo tipo de contrato (mensal vs. anual).
*   Métodos de pagamento e tipo de serviço de internet também desempenham um papel importante na decisão de evasão.
*   Modelos de Machine Learning, como Regressão Logística e SVM, apresentaram bom desempenho na identificação de clientes em risco.

## Instruções para Executar o Notebook

Para executar o notebook `TelecomX_Churn_Analysis_Full.ipynb` no Google Colab, siga os passos abaixo:

1.  **Faça o upload dos arquivos**: Faça o upload dos seguintes arquivos para o ambiente do Google Colab:
    *   `TelecomX_Data.json` (o arquivo de dados original)
    *   `TelecomX_Churn_Analysis_Full.ipynb` (o notebook)

2.  **Abra o Notebook**: No Google Colab, abra o arquivo `TelecomX_Churn_Analysis_Full.ipynb`.

3.  **Execute as Células**: Execute as células do notebook sequencialmente. O notebook está estruturado para guiar você por todas as etapas da análise, desde o carregamento e pré-processamento dos dados até a modelagem e a análise de resultados.

    *   **Observação**: As visualizações (gráficos) serão exibidas diretamente no notebook após a execução das células correspondentes. Os arquivos `.png` gerados pelos scripts Python são apenas para referência e não são estritamente necessários para a execução do notebook no Colab, mas podem ser úteis para visualização externa ou inclusão em relatórios.

4.  **Verifique as Saídas**: Preste atenção às saídas de cada célula, que incluem informações sobre o pré-processamento, métricas de desempenho dos modelos e análises de importância de variáveis.

Este notebook é auto-contido e deve ser executado sem problemas no ambiente do Google Colab, desde que o arquivo de dados `TelecomX_Data.json` esteja presente no mesmo diretório do notebook.

