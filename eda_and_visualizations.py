
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados pré-processados
try:
    df = pd.read_csv("telecomx_processed_data.csv")
except FileNotFoundError:
    print("Erro: O arquivo telecomx_processed_data.csv não foi encontrado. Execute a etapa de pré-processamento primeiro.")
    exit()

print("Dados pré-processados carregados para EDA:")
print(df.head())
print(df.info())

# 1. Proporção de clientes que evadiram (Churn)
churn_counts = df["Churn"].value_counts(normalize=True)
print("\nProporção de Churn (0=Não, 1=Sim):")
print(churn_counts)

plt.figure(figsize=(6, 5))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="viridis")
plt.title("Proporção de Clientes com Churn")
plt.xlabel("Churn (0: Não, 1: Sim)")
plt.ylabel("Proporção")
plt.xticks(ticks=[0, 1], labels=["Não Churn", "Churn"])
plt.savefig("churn_proportion.png")
plt.close()

# 2. Matriz de Correlação
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação das Variáveis")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# 3. Relação entre Tempo de contrato (customer.tenure) e Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Churn", y="customer.tenure", palette="pastel")
plt.title("Tempo de Contrato vs. Churn")
plt.xlabel("Churn (0: Não, 1: Sim)")
plt.ylabel("Tempo de Contrato (meses)")
plt.xticks(ticks=[0, 1], labels=["Não Churn", "Churn"])
plt.savefig("tenure_vs_churn.png")
plt.close()

# 4. Relação entre Total gasto (account.Charges.Total) e Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Churn", y="account.Charges.Total", palette="pastel")
plt.title("Total Gasto vs. Churn")
plt.xlabel("Churn (0: Não, 1: Sim)")
plt.ylabel("Total Gasto")
plt.xticks(ticks=[0, 1], labels=["Não Churn", "Churn"])
plt.savefig("total_charges_vs_churn.png")
plt.close()

print("EDA e visualizações concluídas. Gráficos salvos como arquivos PNG.")


