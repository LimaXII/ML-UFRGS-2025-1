import matplotlib
import numpy as np
import pandas as pd

from models.models import (
    train_decision_tree_model,
    train_random_forest,
)

from models.hyperopt import (
    optimize_decision_tree,
    optimize_random_forest
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


best_dt = optimize_decision_tree(X, y)
best_rf = optimize_random_forest(X, y)

# Model 2: Decision Tree Regressor
# train_decision_tree_model(X=X, y=y)

# Model 9: Random Forest
# train_random_forest(X=X, y=y)
