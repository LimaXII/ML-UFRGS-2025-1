import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def detect_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return ((col < lower_bound) | (col > upper_bound)).sum()


def preprocessing(data: str):
    matplotlib.use('Agg')
    df = pd.read_csv(data)

    # 4. Identificação e Tratamento de Problemas

    # a) Dados Faltantes
    print("\n=== DADOS FALTANTES ===\n")
    print("Quantidade de valores nulos por coluna:")
    print(df.isnull().sum())

    # Visualização gráfica dos dados faltantes
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Mapa de Valores Nulos no Dataset")
    plt.savefig("graphs/preprocessing/null_map.png")
    plt.close()

    # b) Identificação de Outliers
    print("\n=== OUTLIERS ===\n")

    # Verificar outliers nas colunas numéricas
    numeric_cols = ['duration', 'days_left', 'price']
    for col in numeric_cols:
        outliers = detect_outliers(col=df[col])
        print(f"{col}: {outliers} outliers")

    # Visualização de outliers para a variável Price
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['price'])
    plt.title("Boxplot de Preços (Identificação de Outliers)")
    plt.savefig("graphs/preprocessing/outliers.png")
    plt.close()

    # Opções para tratar outliers:
    # 1. Remover
    # df = df[~outliers_mask]
    # 2. Transformar (log)
    # df['Price_log'] = np.log(df['Price'])
    # 3. Winsorizar
    # from scipy.stats.mstats import winsorize
    # df['Price_winsorized'] = winsorize(df['Price'], limits=[0.05, 0.05])

    # c) Engenharia de Features
    print("\n=== ENGENHARIA DE FEATURES ===\n")

    print("Valores únicos em 'departure_time':", df['departure_time'].unique())
    print("Valores únicos em 'arrival_time':", df['arrival_time'].unique())

    # Converter departure_time e arrival_time para categorias temporais
    time_mapping: dict = {
        'Early_Morning': 0,
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Night': 4,
        'Late_Night': 5
    }

    df['departure_time_num'] = df['departure_time'].map(time_mapping)
    df['arrival_time_num'] = df['arrival_time'].map(time_mapping)

    # Criar variável de rota combinada
    df['route_combined'] = df['source_city'] + '-' + df['destination_city']

    # Converter stops para numérico (se necessário)
    stop_mapping = {
        'zero': 0,
        'one': 1,
        'two_or_more': 2,
        'unknown': np.nan
    }

    df['stops_numeric'] = df['stops'].map(stop_mapping)
    df['stops_numeric'].fillna(0, inplace=True)

    print("\nNovas features criadas:")
    print(df[['departure_time_num', 'arrival_time_num',
              'route_combined', 'stops_numeric']].head())

    # d) Normalização
    print("\n=== NORMALIZAÇÃO ===\n")

    # Verificar a necessidade de normalização
    print("Estatísticas antes da normalização:")
    print(df[numeric_cols].describe())

    # Normalização da coluna price
    scaler = MinMaxScaler()
    df['price_normalized'] = scaler.fit_transform(df[['price']])

    print("\nEstatísticas após normalização do price:")
    print(df['price_normalized'].describe())

    # e) Balanceamento
    print("\n=== BALANCEAMENTO ===\n")

    # Verificar balanceamento nas colunas categóricas
    categorical_cols = ['airline', 'source_city',
                        'destination_city', 'stops', 'class']
    for col in categorical_cols:
        print(f"\nDistribuição de {col}:")
        print(df[col].value_counts(normalize=True) * 100)

        # Visualização
        plt.figure(figsize=(10, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribuição de {col}')
        plt.savefig(f"graphs/preprocessing/distribution-{col}.png")
        plt.close()

    # Se houver classes muito desbalanceadas, considerar:
    # 1. Undersampling
    # 2. Oversampling (SMOTE)
    # 3. Ponderar classes no modelo

    # f) Tratamento de Inconsistências
    print("\n=== INCONSISTÊNCIAS ===\n")

    # Verificar durações inconsistentes
    print("Voos com duração suspeita:")
    print(df[df['duration'] <= 0]
          [['source_city', 'destination_city', 'duration']])

    # Verificar preços zero ou negativos
    print("\nVoos com preço zero ou negativo:")
    print(df[df['price'] <= 0]
          [['airline', 'source_city', 'destination_city', 'price']])

    # ### g) Remoção de Colunas Problemáticas
    # print("\n=== REMOÇÃO DE COLUNAS ===\n")

    # # Colunas que podem ser removidas após processamento
    # cols_to_drop = ['flight']  # Exemplo - coluna de ID do voo pode não ser útil
    # print(f"Colunas a serem removidas: {cols_to_drop}")
    # df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # print("\nEstrutura final do DataFrame:")
    # print(df.info())

    # ### h) Salvando o DataFrame processado
    # df.to_csv('flights_data_processed.csv', index=False)
    # print("\nDataFrame processado salvo como 'flights_data_processed.csv'")

    # Remove as colunas originais que não são mais necessárias
    df.drop(columns=["departure_time", "arrival_time", "stops",
            "source_city", "destination_city"], inplace=True)

    return df
