import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_random_forest(
    df: pd.DataFrame
):
    # Categorical columns to encode
    categorical_cols: list = ["airline", "route_combined", "class"]

    # Features and target
    X = df.drop(columns=["price", "Unnamed: 0"], errors="ignore")
    y = df["price"]

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=30
    )

    # Pré-processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Pipeline
    model = Pipeline(
        steps=[
            (
                'preprocessor',
                preprocessor
            ),
            (
                'regressor', RandomForestRegressor(
                    n_estimators=100,
                    random_state=30
                )
            )
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Create graph.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/random_forest/actual_vs_predicted.png")
    plt.close()

    return model
