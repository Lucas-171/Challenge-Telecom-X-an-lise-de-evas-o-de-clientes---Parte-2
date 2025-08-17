import pandas as pd
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
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Remapear Churn para 0 e 1, tratando valores inesperados
y = y[y.isin([1.0, 2.0])]
X = X.loc[y.index] # Manter X alinhado com y
y = y.map({1.0: 0, 2.0: 1}) # Remapear 1.0 para 0 e 2.0 para 1

# Divisão dos dados em treino e teste (re-executar para garantir consistência)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização/Padronização dos dados (re-executar para garantir consistência)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Re-treinar modelos para acessar atributos de importância de features
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42, probability=True)
}

# Análise de Importância das Variáveis
print("\n--- Análise de Importância das Variáveis ---")

# Logistic Regression
model_lr = models["Logistic Regression"]
model_lr.fit(X_train_scaled, y_train)
lr_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_lr.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\nCoeficientes da Regressão Logística (top 10):\n", lr_coefficients.head(10))
print("\nCoeficientes da Regressão Logística (bottom 10):\n", lr_coefficients.tail(10))

# Random Forest
model_rf = models["Random Forest"]
model_rf.fit(X_train, y_train)
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nImportância das Features do Random Forest (top 10):\n", rf_importances.head(10))

# Visualização da importância das features do Random Forest
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importances.head(10), palette='viridis')
plt.title('Top 10 Features Mais Importantes (Random Forest)')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# SVM (coeficientes para kernel linear, para outros kernels é mais complexo)
# Para SVM com kernel não linear, a importância direta dos coeficientes não é aplicável.
# Poderíamos usar Permutation Importance, mas para manter a simplicidade do desafio, focaremos nos coeficientes para LR e feature_importances para RF.
# Se o SVM fosse linear, seria:
# model_svm = models["SVM"]
# model_svm.fit(X_train_scaled, y_train)
# if hasattr(model_svm, 'coef_'):
#     svm_coefficients = pd.DataFrame({
#         'Feature': X.columns,
#         'Coefficient': model_svm.coef_[0]
#     }).sort_values(by='Coefficient', ascending=False)
#     print("\nCoeficientes do SVM (top 10):\n", svm_coefficients.head(10))

# Comparação e Análise Crítica dos Modelos (carregar resultados salvos)
try:
    with open("model_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Erro: O arquivo model_results.json não foi encontrado. Execute a etapa de treinamento e avaliação primeiro.")
    exit()

print("\n--- Análise Crítica e Comparação dos Modelos ---")

best_model = None
best_f1 = -1

for name, metrics in results.items():
    print(f"\nModelo: {name}")
    print(f"  Acurácia: {metrics['accuracy']:.4f}")
    print(f"  Precisão: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")
    print("  Matriz de Confusão:\n", np.array(metrics['confusion_matrix']))
    print("  Relatório de Classificação:\n", metrics['classification_report'])

    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        best_model = name

print(f"\nO modelo com melhor desempenho (F1-score) foi: {best_model}")

print("\nAnálise de Overfitting/Underfitting e Justificativas:")
print("\n- **Regressão Logística:** Teve um bom desempenho geral, especialmente em precisão para a classe majoritária. Por ser um modelo linear, é menos propenso a overfitting, mas pode ter underfitting se as relações nos dados forem muito complexas. A normalização foi crucial para este modelo.")
print("- **Árvore de Decisão:** Geralmente mais propensa a overfitting se não for controlada (profundidade máxima, etc.). Seu desempenho foi o mais baixo, indicando possível underfitting ou que o modelo é muito simples para capturar as nuances dos dados, ou overfitting nos dados de treino. Não requer normalização.")
print("- **Random Forest:** Um ensemble de árvores de decisão, geralmente mais robusto contra overfitting do que uma única árvore. Apresentou bom desempenho, ligeiramente inferior à Regressão Logística e SVM em F1-score, mas com boa precisão. Não requer normalização.")
print("- **KNN:** Sensível à escala dos dados, por isso a normalização foi aplicada. Seu desempenho foi intermediário. Pode ser sensível ao número de vizinhos (k) e à dimensionalidade dos dados.")
print("- **SVM:** Teve um desempenho similar ou ligeiramente superior à Regressão Logística em F1-score. Modelos SVM podem ser muito poderosos, mas a escolha do kernel e a otimização dos hiperparâmetros são cruciais. A normalização é essencial para o SVM.")

print("\nConsiderações sobre Overfitting/Underfitting:")
print("Para avaliar overfitting/underfitting de forma mais robusta, seria necessário comparar as métricas de desempenho nos conjuntos de treino e teste. Se o desempenho no treino for significativamente melhor que no teste, há indícios de overfitting. Se o desempenho for baixo em ambos, há indícios de underfitting.")

