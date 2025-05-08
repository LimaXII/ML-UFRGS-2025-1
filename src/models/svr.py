import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_svr_model(X_train, X_test, y_train, y_test):

    # Inicializa o SVR com kernel RBF (padrão)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    svr.fit(X_train, y_train)

    # Realiza as predições
    y_pred_log = svr.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    # Avaliação do modelo
    mae: float = mean_absolute_error(y_test_real, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2: float = r2_score(y_test_real, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    plot_graph(y_test_real, y_pred, "SVR", "svr")
