
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados pré-processados
try:
    df = pd.read_csv("telecomx_processed_data.csv")
except FileNotFoundError:
    print("Erro: O arquivo telecomx_processed_data.csv não foi encontrado. Execute a etapa de pré-processamento primeiro.")
    exit()

print("Dados pré-processados carregados para modelagem:")
print(df.head())
print(df.info())

# Definir X (features) e y (target)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Ajustar o Churn para ser binário (0 e 1) se houver mais de 2 classes
# Isso é necessário porque o LabelEncoder pode ter criado 0, 1, 2 se houvesse 3 categorias
# E o desafio de churn é binário (evadiu/não evadiu)
# Se a proporção de churn_counts mostrou 0.0, 1.0, 2.0, precisamos mapear para 0 e 1
# Assumindo que 1.0 é 'Não Churn' e 2.0 é 'Churn' (baseado na saída anterior do EDA)
# E 0.0 é uma categoria que não deveria existir ou é um erro de label encoding
# Vamos remapear para garantir que Churn seja 0 ou 1
# 0 -> Não Churn, 1 -> Churn
# Se a saída do LabelEncoder for 0, 1, 2, e 1.0 é a maioria (Não Churn) e 2.0 é a minoria (Churn)
# Vamos remapear 1.0 para 0 e 2.0 para 1. O 0.0 é um outlier que precisa ser tratado.

# Vamos verificar os valores únicos de y antes de remapear
print(f"Valores únicos de y (Churn) antes do remapeamento: {y.unique()}")

# Remapear Churn para 0 e 1, tratando valores inesperados
# Se 0.0, 1.0, 2.0 são os valores, e 1.0 é a maioria (Não Churn) e 2.0 é a minoria (Churn)
# Vamos assumir que 0.0 é um erro e pode ser tratado como a maioria ou removido.
# Para simplificar, vamos considerar apenas 1.0 e 2.0 como as classes principais
# e remapear 1.0 para 0 (Não Churn) e 2.0 para 1 (Churn).
# Se 0.0 ainda existir, ele será tratado como um valor inesperado.

# Filtrar apenas as classes 1.0 e 2.0 para o target
y = y[y.isin([1.0, 2.0])]
X = X.loc[y.index] # Manter X alinhado com y

y = y.map({1.0: 0, 2.0: 1}) # Remapear 1.0 para 0 e 2.0 para 1

print(f"Valores únicos de y (Churn) após o remapeamento: {y.unique()}")

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Avaliar a proporção de Churn nos conjuntos de treino e teste
print("\nProporção de Churn no conjunto de treino:")
print(y_train.value_counts(normalize=True))
print("\nProporção de Churn no conjunto de teste:")
print(y_test.value_counts(normalize=True))

# Modelos a serem testados
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42, probability=True) # probability=True para feature importance
}

results = {}

# Normalização/Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento e avaliação dos modelos
for name, model in models.items():
    print(f"\n--- Treinando e Avaliando: {name} ---")
    
    # Decidir se normalizar ou não
    if name in ["Logistic Regression", "KNN", "SVM"]:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
        print(f"Dados normalizados usados para {name}.")
    else:
        X_train_model = X_train
        X_test_model = X_test
        print(f"Dados não normalizados usados para {name}.")

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_mat,
        "classification_report": class_report
    }

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusão:\n", conf_mat)
    print("Relatório de Classificação:\n", class_report)

# Comparação dos modelos
print("\n--- Comparação dos Modelos ---")
for name, metrics in results.items():
    print(f"\nModelo: {name}")
    print(f"  Acurácia: {metrics['accuracy']:.4f}")
    print(f"  Precisão: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")

# Salvar resultados para análise posterior
import json
with open("model_results.json", "w") as f:
    # Convert numpy arrays to list for JSON serialization
    serializable_results = {}
    for name, metrics in results.items():
        serializable_results[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "classification_report": metrics["classification_report"]
        }
    json.dump(serializable_results, f, indent=4)

print("Resultados dos modelos salvos em model_results.json")


