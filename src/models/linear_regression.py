from sklearn.linear_model import LinearRegression

from processing import evaluate, plot_graph


def train_linear_regression_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 7: Linear Regression \n")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="Linear Regression",
        dir_name="linear_regression"
    )
