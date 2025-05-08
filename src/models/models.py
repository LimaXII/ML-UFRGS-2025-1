from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from processing import evaluate, plot_graph


def train_bayesian_ridge(X, y):
    print("\nModelo número 1: Bayesian Ridge\n")
    y_test_real, y_pred = evaluate(
        model=BayesianRidge(),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Bayesian Ridge",
        dir_name="bayesian_ridge"
    )


def train_decision_tree_model(X, y):
    print("\nModelo número 2: Decision Tree\n")

    y_test_real, y_pred = evaluate(
        model=DecisionTreeRegressor(
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Decision Tree",
        dir_name="decision_tree"
    )


def train_elastic_net(X, y):
    print("\nModelo número 3: Elastic Net\n")
    y_test_real, y_pred = evaluate(
        model=ElasticNet(
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="ElasticNet",
        dir_name="elastic_net"
    )


def train_gradient_boosting_model(X, y):
    print("\nModelo número 4: Gradient Boosting\n")
    y_test_real, y_pred = evaluate(
        model=GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Gradiente Boosting",
        dir_name="gradient_boosting"
    )


def train_histogram_gb(X, y):
    print("\nModelo número 5: Histogram based Gradient Boosting\n")
    y_test_real, y_pred = evaluate(
        model=HistGradientBoostingRegressor(
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Histogram Gradient Boosting",
        dir_name="histogram_gb"
    )


def train_knn_model(X, y):
    print("\nModelo número 6: KNN\n")
    y_test_real, y_pred = evaluate(
        model=KNeighborsRegressor(
            n_neighbors=2
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="KNN",
        dir_name="knn"
    )


def train_linear_regression_model(X, y):
    print("\nModelo número 7: Linear Regression\n")
    y_test_real, y_pred = evaluate(
        model=LinearRegression(),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Linear Regression",
        dir_name="linear_regression"
    )


def train_mlp_model(X, y):
    print("\nModelo número 8: MLP Regressor\n")
    y_test_real, y_pred = evaluate(
        model=MLPRegressor(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="MLP Regressor",
        dir_name="mlp"
    )


def train_random_forest(X, y):
    print("\nModelo número 9: Random Forest\n")
    y_test_real, y_pred = evaluate(
        model=RandomForestRegressor(
            n_estimators=200,
            random_state=30
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Random Forest",
        dir_name="random_forest"
    )


def train_svr_model(X, y):
    print("\nModelo número 10: SVR\n")
    y_test_real, y_pred = evaluate(
        model=SVR(
            kernel="linear",
            C=1.0,
            epsilon=0.2
        ),
        X=X,
        y=y
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="SVR",
        dir_name="svr"
    )
