import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def train_knn_model(df: pd.DataFrame):
    """
    Train a K-Nearest Neighbors (KNN) regression model on a preprocessed DataFrame.

    Args:
        df (pd.DataFrame): Preprocessed dataset including the target column 'price'.
    """
    # Categorical columns to encode
    categorical_cols = ["airline", "flight", "route_combined", "class"]
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Initialize the KNN regressor. k=5.
    knn = KNeighborsRegressor(
        n_neighbors=5
    )
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/knn/actual_vs_predicted.png")
    plt.close()

    # Plot 2: Residuals vs Predicted
    residuals = y_test - y_pred
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
