import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from processing import plot_graph

def train_histogram_gb(X_train, X_test, y_train, y_test):
    

    model = HistGradientBoostingRegressor(random_state=30)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)

    print(f"[HistGB] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    plot_graph(y_test_real, y_pred, "Histogram Gradient Boosting", "histogram_gb")

