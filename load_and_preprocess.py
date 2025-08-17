
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Carregar os dados
try:
    df = pd.read_json("/home/ubuntu/challenge2-data-science-main/challenge2-data-science-main/TelecomX_Data.json")
except FileNotFoundError:
    print("Erro: O arquivo TelecomX_Data.json não foi encontrado. Por favor, faça o upload do arquivo.")
    exit()

print("Dados originais carregados:")
print(df.head())
print(df.info())

# Desaninhamento das colunas de dicionário
def flatten_dict_column(df, column_name):
    flattened_data = pd.json_normalize(df[column_name])
    flattened_data.columns = [f"{column_name}.{sub_col}" for sub_col in flattened_data.columns]
    df = df.drop(columns=[column_name]).join(flattened_data)
    return df

df = flatten_dict_column(df, 'customer')
df = flatten_dict_column(df, 'phone')
df = flatten_dict_column(df, 'internet')
df = flatten_dict_column(df, 'account')

# Eliminar colunas que não trazem valor (ID do cliente)
# Assumindo que 'customerID' é o identificador único
if 'customerID' in df.columns:
    df = df.drop("customerID", axis=1)

print("\nDados após desaninhamento (primeiras 5 linhas e info):")
print(df.head())
print(df.info())

# Tratar valores ausentes (se houver) e converter tipos
# Converter 'account.Charges.Total' para numérico, tratando erros
df['account.Charges.Total'] = pd.to_numeric(df['account.Charges.Total'], errors='coerce')

# Imputação para colunas numéricas (ex: com a média)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# Imputação para colunas categóricas (ex: com a moda)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Transformar a variável target 'Churn' para numérica (0 e 1)
if 'Churn' in df.columns:
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

# Converter colunas binárias de 'Yes'/'No' para 1/0
binary_cols_to_convert = [
    'customer.Partner', 'customer.Dependents', 'phone.PhoneService', 'phone.MultipleLines',
    'internet.OnlineSecurity', 'internet.OnlineBackup', 'internet.DeviceProtection',
    'internet.TechSupport', 'internet.StreamingTV', 'internet.StreamingMovies',
    'account.PaperlessBilling', 'customer.gender' # Adicionado gender aqui
]

for col in binary_cols_to_convert:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else (1 if x == 'Male' else 0)))

# Re-identificar colunas categóricas após o tratamento das binárias e imputação
# Agora, apenas colunas com mais de 2 categorias ou que não foram tratadas acima
final_categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Aplicar OneHotEncoder para as colunas categóricas restantes
if final_categorical_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), final_categorical_cols)
        ], remainder='passthrough'
    )
    df_processed = preprocessor.fit_transform(df)
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(final_categorical_cols)
    
    # Certificar-se de que o número de colunas corresponde antes de criar o DataFrame
    # O ColumnTransformer pode reordenar as colunas ou adicionar novas
    # A melhor prática é obter os nomes das colunas resultantes do ColumnTransformer
    
    # Obter todos os nomes de colunas após a transformação
    all_feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            all_feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(columns))
        else:
            # Para 'remainder=passthrough', as colunas originais são mantidas
            # Precisamos garantir que a ordem seja a mesma que o ColumnTransformer produz
            # Isso é um pouco complexo, uma abordagem mais simples é converter para DataFrame e depois renomear
            pass # Trataremos isso abaixo com o DataFrame final

    # Criar um DataFrame com as colunas transformadas
    # A ordem das colunas no df_processed será: OHE cols, then passthrough cols
    # Precisamos obter os nomes das colunas passthrough na ordem correta
    passthrough_cols = [col for col in df.columns if col not in final_categorical_cols]
    
    df = pd.DataFrame(df_processed, columns=list(ohe_feature_names) + passthrough_cols)

print("\nDados após pré-processamento completo (primeiras 5 linhas e info):")
print(df.head())
print(df.info())

# Salvar o dataframe pré-processado para uso posterior
df.to_csv("telecomx_processed_data.csv", index=False)


