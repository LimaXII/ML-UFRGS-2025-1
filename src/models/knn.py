import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def train_knn_model(
    df: pd.DataFrame
):
    """
    Train a  regression model on a preprocessed DataFrame.
    Treina um modelo K-Nearest Neighbors (KNN) utilizando um DataFrame já pré-processado

    Args:
        df (pd.DataFrame): Dataset já pré-processado, que incluí a coluna que deve ser predita, 'price'.
    """
    # Colunas com valores categóricos.
    categorical_cols: list = ["airline", "route_combined", "class"]
    # One-hot encode para dados categóricos.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["price", "Unnamed: 0"], errors="ignore")
    # Converte os valores de 'price' com log, para normalizar a resposta.
    y = np.log1p(df["price"])

    # Separa os dados (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=30
    )

    # Inicializa o Knn, com k=2.
    knn = KNeighborsRegressor(
        n_neighbors=2
    )
    knn.fit(X_train, y_train)

    # Realiza as predições.
    y_pred_log = knn.predict(X_test)
    # Converte os valores de log de volta.
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    # Avalia o modelo.
    mae: float = mean_absolute_error(y_test_real, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2: float = r2_score(y_test_real, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    # Plot 1: Valor Real x Valor predito.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.6)
    plt.plot(
        [y_test_real.min(), y_test_real.max()],
        [y_test_real.min(), y_test_real.max()],
        'r--'
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/knn/actual_vs_predicted.png")
    plt.close()

    # Plot 2: Valor residual x Valor Predito.
    residuals = y_test_real - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/knn/residuals_vs_predicted.png")
    plt.close()
