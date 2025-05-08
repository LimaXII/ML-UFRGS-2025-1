from sklearn.ensemble import HistGradientBoostingRegressor

from processing import evaluate, plot_graph


def train_histogram_gb(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 4: Histogram Gradient Boosting \n")

    model = HistGradientBoostingRegressor(
        random_state=30
    )
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Histogram Gradient Boosting",
        dir_name="histogram_gb"
    )
