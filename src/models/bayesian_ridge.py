
from sklearn.linear_model import BayesianRidge

from processing import evaluate, plot_graph


def train_bayesian_ridge(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 1: Bayesian Ridge \n")

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Bayesian Ridge",
        dir_name="bayesian_ridge"
    )
