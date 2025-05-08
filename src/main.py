import matplotlib
import numpy as np
import pandas as pd

from models.models import (
    train_bayesian_ridge,
    train_decision_tree_model,
    train_elastic_net,
    train_gradient_boosting_model,
    train_histogram_gb,
    train_knn_model,
    train_linear_regression_model,
    train_mlp_model,
    train_random_forest,
    train_svr_model,
)
from preprocessing import preprocessing

matplotlib.use('Agg')
# Comentado pois não precisamos processar novamente os dados, por agora.
# df = preprocessing(data_path="data/Clean_Dataset.csv")


def splitData(
    df: pd.DataFrame
):
    """
    Divide os dados em variáveis features (X) e target (y).
    A variável target é o preço do voo, enquanto as variáveis features
    são todas as outras colunas do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados de voos.
    """

    categorical_cols: list = [
        "airline",
        "route_combined",
        "class"
    ]
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    X = df.drop(
        columns=[
            "price",
            "Unnamed: 0"],
        errors="ignore"
    )
    y = np.log1p(df["price"])
    return X, y


df: pd.DataFrame = pd.read_csv("data/flights_data_processed.csv")
X, y = splitData(df=df)

# Model 1: Bayesian Ridge Regression
train_bayesian_ridge(X=X, y=y)

# Model 2: Decision Tree Regressor
train_decision_tree_model(X=X, y=y)

# Model 3: Elastic Net
train_elastic_net(X=X, y=y)

# Model 4: Gradient Boosting Regressor
train_gradient_boosting_model(X=X, y=y)

# Model 5: Histogram based Gradient Boosting
train_histogram_gb(X=X, y=y)

# Model 6: KNN
train_knn_model(X=X, y=y)

# Model 7: Linear Regression
train_linear_regression_model(X=X, y=y)

# Model 8: MLP Regressor
train_mlp_model(X=X, y=y)

# Model 9: Random Forest
train_random_forest(X=X, y=y)

# Model 10: SVR
train_svr_model(X=X, y=y)
