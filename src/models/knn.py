import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from processing import plot_graph

def train_knn_model(X_train, X_test, y_train, y_test):

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

    plot_graph(y_test_real, y_pred, "KNN", "knn")
