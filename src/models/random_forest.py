import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_random_forest(
    df: pd.DataFrame
):
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

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=30
    )

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    # Converte os valores de log de volta.
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    # Avalia o modelo.
    mae: float = mean_absolute_error(y_test_real, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2: float = r2_score(y_test_real, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

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
    plt.savefig("graphs/random_forest/actual_vs_predicted.png")
    plt.close()

    return model
