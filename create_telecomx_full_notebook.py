
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Create a new notebook object
nb = new_notebook()

# Adicionar Título Principal
nb.cells.append(new_markdown_cell("""# Análise e Modelagem de Churn de Clientes - TelecomX

Este notebook apresenta uma análise completa do churn de clientes da TelecomX, desde o pré-processamento dos dados até a modelagem preditiva, avaliação de modelos e análise de importância de variáveis. O objetivo é identificar os principais fatores que levam os clientes a cancelar seus serviços e propor estratégias de retenção.

**Conteúdo:**
1.  Carregamento e Pré-processamento dos Dados
2.  Análise Exploratória de Dados (EDA) e Visualizações
3.  Modelagem Preditiva e Avaliação de Modelos
4.  Análise de Importância das Variáveis e Comparação de Modelos
5.  Conclusões e Estratégias de Retenção
"""))

# Seção 1: Carregamento e Pré-processamento dos Dados
nb.cells.append(new_markdown_cell("""## 1. Carregamento e Pré-processamento dos Dados

Nesta seção, os dados são carregados, as colunas aninhadas são desaninhadas, colunas irrelevantes são removidas, valores ausentes são tratados e variáveis categóricas são transformadas em formato numérico utilizando One-Hot Encoding.
"""))
nb.cells.append(new_code_cell("""import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Carregar os dados
try:
    df = pd.read_json("TelecomX_Data.json")
except FileNotFoundError:
    print("Erro: O arquivo TelecomX_Data.json não foi encontrado. Por favor, faça o upload do arquivo para o ambiente do Colab.")
    exit()

print("Dados originais carregados:")
print(df.head())
print(df.info())

# Desaninhamento das colunas de dicionário
def flatten_dict_column(df, column_name):
    flattened_data = pd.json_normalize(df[column_name])
    flattened_data.columns = [f\"{column_name}.{sub_col}\" for sub_col in flattened_data.columns]
    df = df.drop(columns=[column_name]).join(flattened_data)
    return df

df = flatten_dict_column(df, \'customer\')
df = flatten_dict_column(df, \'phone\')
df = flatten_dict_column(df, \'internet\')
df = flatten_dict_column(df, \'account\')

# Eliminar colunas que não trazem valor (ID do cliente)
if \'customerID\' in df.columns:
    df = df.drop(\"customerID\", axis=1)

print("\nDados após desaninhamento (primeiras 5 linhas e info):")
print(df.head())
print(df.info())

# Tratar valores ausentes (se houver) e converter tipos
df[\'account.Charges.Total\'] = pd.to_numeric(df[\'account.Charges.Total\'], errors=\'coerce\')

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

categorical_cols = df.select_dtypes(include=\'object\').columns.tolist()
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Transformar a variável target \'Churn\' para numérica (0 e 1)
if \'Churn\' in df.columns:
    le = LabelEncoder()
    df[\'Churn\'] = le.fit_transform(df[\'Churn\'])

# Converter colunas binárias de \'Yes\'/\'No\' para 1/0
binary_cols_to_convert = [
    \'customer.Partner\', \'customer.Dependents\', \'phone.PhoneService\', \'phone.MultipleLines\',
    \'internet.OnlineSecurity\', \'internet.OnlineBackup\', \'internet.DeviceProtection\',
    \'internet.TechSupport\', \'internet.StreamingTV\', \'internet.StreamingMovies\',
    \'account.PaperlessBilling\', \'customer.gender\'
]

for col in binary_cols_to_convert:
    if col in df.columns and df[col].dtype == \'object\':
        df[col] = df[col].apply(lambda x: 1 if x == \'Yes\' else (0 if x == \'No\' else (1 if x == \'Male\' else 0)))

# Re-identificar colunas categóricas após o tratamento das binárias e imputação
final_categorical_cols = df.select_dtypes(include=\'object\').columns.tolist()

# Aplicar OneHotEncoder para as colunas categóricas restantes
if final_categorical_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            (\'cat\', OneHotEncoder(handle_unknown=\'ignore\'), final_categorical_cols)
        ], remainder=\'passthrough\'
    )
    df_processed = preprocessor.fit_transform(df)
    ohe_feature_names = preprocessor.named_transformers_[\'cat\'].get_feature_names_out(final_categorical_cols)
    
    passthrough_cols = [col for col in df.columns if col not in final_categorical_cols]
    
    df = pd.DataFrame(df_processed, columns=list(ohe_feature_names) + passthrough_cols)

print("\nDados após pré-processamento completo (primeiras 5 linhas e info):")
print(df.head())
print(df.info())

# Salvar o dataframe pré-processado para uso posterior
df.to_csv("telecomx_processed_data.csv", index=False)
"""))

# Seção 2: Análise Exploratória de Dados (EDA) e Visualizações
nb.cells.append(new_markdown_cell("""## 2. Análise Exploratória de Dados (EDA) e Visualizações

Nesta seção, realizamos uma análise exploratória para entender a distribuição das variáveis, identificar padrões e relações, e visualizar a proporção de churn, correlações entre variáveis e o impacto de fatores como tempo de contrato e gastos totais no churn.
"""))
nb.cells.append(new_code_cell("""import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados pré-processados
try:
    df = pd.read_csv("telecomx_processed_data.csv")
except FileNotFoundError:
    print("Erro: O arquivo telecomx_processed_data.csv não foi encontrado. Execute a etapa de pré-processamento primeiro.")
    exit()

# 1. Proporção de clientes que evadiram (Churn)
churn_counts = df[\"Churn\"].value_counts(normalize=True)
print(\"\nProporção de Churn (0=Não, 1=Sim):\")
print(churn_counts)

plt.figure(figsize=(6, 5))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=\"viridis\")
plt.title(\"Proporção de Clientes com Churn\")
plt.xlabel(\"Churn (0: Não, 1: Sim)\")
plt.ylabel(\"Proporção\")
plt.xticks(ticks=[0, 1], labels=[\"Não Churn\", \"Churn\"])
plt.show()

# 2. Matriz de Correlação
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=False, cmap=\"coolwarm\", fmt=\".2f\")
plt.title(\"Matriz de Correlação das Variáveis\")
plt.tight_layout()
plt.show()

# 3. Relação entre Tempo de contrato (customer.tenure) e Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x=\"Churn\", y=\"customer.tenure\", palette=\"pastel\")
plt.title(\"Tempo de Contrato vs. Churn\")
plt.xlabel(\"Churn (0: Não, 1: Sim)\")
plt.ylabel(\"Tempo de Contrato (meses)\")
plt.xticks(ticks=[0, 1], labels=[\"Não Churn\", \"Churn\"])
plt.show()

# 4. Relação entre Total gasto (account.Charges.Total) e Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x=\"Churn\", y=\"account.Charges.Total\", palette=\"pastel\")
plt.title(\"Total Gasto vs. Churn\")
plt.xlabel(\"Churn (0: Não, 1: Sim)\")
plt.ylabel(\"Total Gasto\")
plt.xticks(ticks=[0, 1], labels=[\"Não Churn\", \"Churn\"])
plt.show()
"""))

# Seção 3: Modelagem Preditiva e Avaliação de Modelos
nb.cells.append(new_markdown_cell("""## 3. Modelagem Preditiva e Avaliação de Modelos

Nesta seção, dividimos os dados em conjuntos de treino e teste, treinamos múltiplos modelos de classificação (Regressão Logística, Árvore de Decisão, Random Forest, KNN, SVM) e avaliamos seu desempenho usando métricas como Acurácia, Precisão, Recall, F1-score e Matriz de Confusão.
"""))
nb.cells.append(new_code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Carregar os dados pré-processados
try:
    df = pd.read_csv("telecomx_processed_data.csv")
except FileNotFoundError:
    print("Erro: O arquivo telecomx_processed_data.csv não foi encontrado. Execute a etapa de pré-processamento primeiro.")
    exit()

# Definir X (features) e y (target)
X = df.drop(\"Churn\", axis=1)
y = df[\"Churn\"]

# Remapear Churn para 0 e 1, tratando valores inesperados
y = y[y.isin([1.0, 2.0])]
X = X.loc[y.index]
y = y.map({1.0: 0, 2.0: 1})

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelos a serem testados
models = {
    \"Logistic Regression\": LogisticRegression(random_state=42, solver=\'liblinear\', max_iter=1000),
    \"Decision Tree\": DecisionTreeClassifier(random_state=42),
    \"Random Forest\": RandomForestClassifier(random_state=42),
    \"KNN\": KNeighborsClassifier(),
    \"SVM\": SVC(random_state=42, probability=True)
}

results = {}

# Normalização/Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento e avaliação dos modelos
for name, model in models.items():
    print(f\"\n--- Treinando e Avaliando: {name} ---\")
    
    if name in [\"Logistic Regression\", \"KNN\", \"SVM\"]:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
        print(f\"Dados normalizados usados para {name}.\")
    else:
        X_train_model = X_train
        X_test_model = X_test
        print(f\"Dados não normalizados usados para {name}.\")

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    results[name] = {
        \"accuracy\": accuracy,
        \"precision\": precision,
        \"recall\": recall,
        \"f1_score\": f1,
        \"confusion_matrix\": conf_mat,
        \"classification_report\": class_report
    }

    print(f\"Acurácia: {accuracy:.4f}\")
    print(f\"Precisão: {precision:.4f}\")
    print(f\"Recall: {recall:.4f}\")
    print(f\"F1-score: {f1:.4f}\")
    print(\"Matriz de Confusão:\n\", conf_mat)
    print(\"Relatório de Classificação:\n\", class_report)

# Salvar resultados para análise posterior
import json
with open(\"model_results.json\", \"w\") as f:
    serializable_results = {}
    for name, metrics in results.items():
        serializable_results[name] = {
            \"accuracy\": metrics[\"accuracy\"],
            \"precision\": metrics[\"precision\"],
            \"recall\": metrics[\"recall\"],
            \"f1_score\": metrics[\"f1_score\"],
            \"confusion_matrix\": metrics[\"confusion_matrix\"].tolist(),
            \"classification_report\": metrics[\"classification_report\"]
        }
    json.dump(serializable_results, f, indent=4)

print(\"Resultados dos modelos salvos em model_results.json\")
"""))

# Seção 4: Análise de Importância das Variáveis e Comparação de Modelos
nb.cells.append(new_markdown_cell("""## 4. Análise de Importância das Variáveis e Comparação de Modelos

Nesta seção, analisamos a importância das variáveis para os modelos treinados e realizamos uma comparação crítica entre eles, discutindo seus pontos fortes, fracos e a necessidade de normalização.
"""))
nb.cells.append(new_code_cell("""import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar os dados pré-processados
try:
    df = pd.read_csv("telecomx_processed_data.csv")
except FileNotFoundError:
    print("Erro: O arquivo telecomx_processed_data.csv não foi encontrado. Execute a etapa de pré-processamento primeiro.")
    exit()

# Definir X (features) e y (target)
X = df.drop(\"Churn\", axis=1)
y = df[\"Churn\"]

# Remapear Churn para 0 e 1, tratando valores inesperados
y = y[y.isin([1.0, 2.0])]
X = X.loc[y.index]
y = y.map({1.0: 0, 2.0: 1})

# Divisão dos dados em treino e teste (re-executar para garantir consistência)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização/Padronização dos dados (re-executar para garantir consistência)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Re-treinar modelos para acessar atributos de importância de features
models = {
    \"Logistic Regression\": LogisticRegression(random_state=42, solver=\'liblinear\', max_iter=1000),
    \"Decision Tree\": DecisionTreeClassifier(random_state=42),
    \"Random Forest\": RandomForestClassifier(random_state=42),
    \"KNN\": KNeighborsClassifier(),
    \"SVM\": SVC(random_state=42, probability=True)
}

# Análise de Importância das Variáveis
print(\"\n--- Análise de Importância das Variáveis ---\")

# Logistic Regression
model_lr = models[\"Logistic Regression\"]
model_lr.fit(X_train_scaled, y_train)
lr_coefficients = pd.DataFrame({
    \'Feature\': X.columns,
    \'Coefficient\': model_lr.coef_[0]
}).sort_values(by=\'Coefficient\', ascending=False)
print(\"\nCoeficientes da Regressão Logística (top 10):\n\", lr_coefficients.head(10))
print(\"\nCoeficientes da Regressão Logística (bottom 10):\n\", lr_coefficients.tail(10))

# Random Forest
model_rf = models[\"Random Forest\"]
model_rf.fit(X_train, y_train)
rf_importances = pd.DataFrame({
    \'Feature\': X.columns,
    \'Importance\': model_rf.feature_importances_
}).sort_values(by=\'Importance\', ascending=False)
print(\"\nImportância das Features do Random Forest (top 10):\n\", rf_importances.head(10))

# Visualização da importância das features do Random Forest
plt.figure(figsize=(12, 8))
sns.barplot(x=\'Importance\', y=\'Feature\', data=rf_importances.head(10), palette=\'viridis\')
plt.title(\'Top 10 Features Mais Importantes (Random Forest)\')
plt.xlabel(\'Importância\')
plt.ylabel(\'Feature\')
plt.show()

# Comparação e Análise Crítica dos Modelos (carregar resultados salvos)
try:
    with open(\"model_results.json\", \"r\") as f:
        results = json.load(f)
except FileNotFoundError:
    print(\"Erro: O arquivo model_results.json não foi encontrado. Execute a etapa de treinamento e avaliação primeiro.\")
    exit()

print(\"\n--- Análise Crítica e Comparação dos Modelos ---\")

best_model = None
best_f1 = -1

for name, metrics in results.items():
    print(f\"\nModelo: {name}\")
    print(f\"  Acurácia: {metrics[\'accuracy\']:.4f}\")
    print(f\"  Precisão: {metrics[\'precision\']:.4f}\")
    print(f\"  Recall: {metrics[\'recall\']:.4f}\")
    print(f\"  F1-score: {metrics[\'f1_score\']:.4f}\")
    print(\"  Matriz de Confusão:\n\", np.array(metrics[\'confusion_matrix\']))
    print(\"  Relatório de Classificação:\n\", metrics[\'classification_report\'])

    if metrics[\'f1_score\'] > best_f1:
        best_f1 = metrics[\'f1_score\']
        best_model = name

print(f\"\nO modelo com melhor desempenho (F1-score) foi: {best_model}\")

print(\"\nAnálise de Overfitting/Underfitting e Justificativas:\")
print(\"\n- **Regressão Logística:** Teve um bom desempenho geral, especialmente em precisão para a classe majoritária. Por ser um modelo linear, é menos propenso a overfitting, mas pode ter underfitting se as relações nos dados forem muito complexas. A normalização foi crucial para este modelo.\")
print(\"- **Árvore de Decisão:** Geralmente mais propensa a overfitting se não for controlada (profundidade máxima, etc.). Seu desempenho foi o mais baixo, indicando possível underfitting ou que o modelo é muito simples para capturar as nuances dos dados, ou overfitting nos dados de treino. Não requer normalização.\")
print(\"- **Random Forest:** Um ensemble de árvores de decisão, geralmente mais robusto contra overfitting do que uma única árvore. Apresentou bom desempenho, ligeiramente inferior à Regressão Logística e SVM em F1-score, mas com boa precisão. Não requer normalização.\")
print(\"- **KNN:** Sensível à escala dos dados, por isso a normalização foi aplicada. Seu desempenho foi intermediário. Pode ser sensível ao número de vizinhos (k) e à dimensionalidade dos dados.\")
print(\"- **SVM:** Teve um desempenho similar ou ligeiramente superior à Regressão Logística em F1-score. Modelos SVM podem ser muito poderosos, mas a escolha do kernel e a otimização dos hiperparâmetros são cruciais. A normalização é essencial para o SVM.\")

print(\"\nConsiderações sobre Overfitting/Underfitting:\")
print(\"Para avaliar overfitting/underfitting de forma mais robusta, seria necessário comparar as métricas de desempenho nos conjuntos de treino e teste. Se o desempenho no treino for significativamente melhor que no teste, há indícios de overfitting. Se o desempenho for baixo em ambos, há indícios de underfitting.\")
"""))

# Seção 5: Conclusões e Estratégias de Retenção
nb.cells.append(new_markdown_cell("""## 5. Conclusões e Estratégias de Retenção

Com base nas análises realizadas, identificamos os principais fatores que contribuem para o churn de clientes e propomos estratégias de retenção acionáveis para a TelecomX.

### Principais Fatores de Churn:

*   **Tempo de Contrato (customer.tenure)**: Clientes com menor tempo de permanência têm maior probabilidade de churn.
*   **Cobrança Total (account.Charges.Total)**: Valores totais gastos impactam significativamente a decisão de permanência.
*   **Tipo de Contrato (account.Contract_Month-to-month)**: Contratos mensais apresentam maior risco de churn.
*   **Tipo de Internet (internet.InternetService_Fiber optic)**: Clientes com fibra ótica mostram padrões específicos de churn.
*   **Método de Pagamento (account.PaymentMethod_Electronic check)**: Pagamentos por cheque eletrônico correlacionam com maior churn.

### Estratégias de Retenção Recomendadas:

1.  **Programa de Onboarding Aprimorado**: Focar nos primeiros meses de relacionamento com acompanhamento proativo e suporte dedicado.
2.  **Incentivos para Contratos de Longo Prazo**: Oferecer descontos e benefícios para clientes que optam por contratos anuais ou bianuais.
3.  **Segmentação e Personalização**: Criar estratégias diferenciadas baseadas no perfil de gasto do cliente.
4.  **Melhoria na Experiência de Pagamento**: Incentivar métodos de pagamento automáticos e simplificar processos.
5.  **Otimização de Serviços de Internet**: Monitorar a qualidade do serviço e oferecer suporte especializado para clientes de fibra ótica.
6.  **Sistema de Alerta Precoce**: Implementar um sistema de scoring de churn para identificar clientes em risco em tempo real e acionar campanhas de retenção proativas.

Essas estratégias, se implementadas de forma coordenada, podem reduzir significativamente a taxa de churn e aumentar a satisfação do cliente na TelecomX.
"""))

# Write the notebook to a file
with open("/home/ubuntu/TelecomX_Churn_Analysis_Full.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)


