import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

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

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.6)
    plt.plot([y_test_real.min(), y_test_real.max()],
             [y_test_real.min(), y_test_real.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted (HistGradientBoosting)")
    plt.tight_layout()
    plt.savefig("graphs/histogram_gb/actual_vs_predicted.png")
    plt.close()

    residuals = y_test_real - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted (HistGradientBoosting)")
    plt.tight_layout()
    plt.savefig("graphs/histogram_gb/residuals_vs_predicted.png")
    plt.close()
