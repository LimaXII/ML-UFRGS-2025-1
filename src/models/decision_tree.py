import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_decision_tree_model(df: pd.DataFrame):
    """
    Train a Decision Tree Regressor model on a preprocessed DataFrame.

    Args:
        df (pd.DataFrame): Dataset já pré-processado, que inclui a coluna 'price' como alvo.
    """
    # Colunas com valores categóricos.
    categorical_cols: list = ["airline", "route_combined", "class"]
    # One-hot encode para dados categóricos.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["price", "Unnamed: 0"], errors="ignore")
    y = np.log1p(df["price"])

    # Separa os dados (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30
    )

    # Inicializa o Decision Tree Regressor (sem poda)
    tree = DecisionTreeRegressor(random_state=30)
    tree.fit(X_train, y_train)

    # Realiza as predições
    y_pred_log = tree.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    # Avaliação do modelo
    mae: float = mean_absolute_error(y_test_real, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2: float = r2_score(y_test_real, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    # Plot 1: Valor Real x Valor Predito
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.6)
    plt.plot(
        [y_test_real.min(), y_test_real.max()],
        [y_test_real.min(), y_test_real.max()],
        'r--'
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices (Decision Tree)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/decision_tree/actual_vs_predicted.png")
    plt.close()

    # Plot 2: Resíduos x Valor Predito
    residuals = y_test_real - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Prices (Decision Tree)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/decision_tree/residuals_vs_predicted.png")
    plt.close()
