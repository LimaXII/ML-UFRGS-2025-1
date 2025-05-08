import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.knn import train_knn_model
from models.random_forest import train_random_forest
from models.gradient_boosting import train_gradient_boosting_model
from models.svr import train_svr_model
from models.decision_tree import train_decision_tree_model
from models.linear_regression import train_linear_regression_model
from models.mlp import train_mlp_model
from models.bayesian_ridge import train_bayesian_ridge
from models.elastic_net import train_elastic_net
from models.histogram_gb import train_histogram_gb
from preprocessing import preprocessing

# DEBUG: Se remover isso o código quebra kk
matplotlib.use('Agg')

def splitData (df: pd.DataFrame):
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

    return X_train, X_test, y_train, y_test

# Comentado pois não precisamos processar novamente os dados, por agora.
# df = preprocessing(data="data/Clean_Dataset.csv")

df = pd.read_csv("data/flights_data_processed.csv")
X_train, X_test, y_train, y_test = splitData(df=df)

# Model 1: KNN
#train_knn_model(X_train, X_test, y_train, y_test)

# Model 2: Random Forest
#train_random_forest(X_train, X_test, y_train, y_test)

# Model 3: Gradient Boosting Regressor
#train_gradient_boosting_model(X_train, X_test, y_train, y_test)

# Model 4: SVR
#train_svr_model(X_train, X_test, y_train, y_test)

# Model 5: Decision Tree Regressor
#train_decision_tree_model(X_train, X_test, y_train, y_test)

# Model 6: Linear Regression
#train_linear_regression_model(X_train, X_test, y_train, y_test)

#M odel 7: MLP Regressor
#train_mlp_model(X_train, X_test, y_train, y_test)

# Model 8: Bayesian Ridge Regression
train_bayesian_ridge(X_train, X_test, y_train, y_test)

# Model 9: Elastic Net
#train_elastic_net(X_train, X_test, y_train, y_test)

# Model 10: Histogram based Gradient Boosting
#train_histogram_gb(X_train, X_test, y_train, y_test)

