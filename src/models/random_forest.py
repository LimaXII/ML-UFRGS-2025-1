from sklearn.ensemble import RandomForestRegressor

from processing import evaluate, plot_graph


def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 9: Random Forest \n")

    model = RandomForestRegressor(
        n_estimators=200,
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
        graph_name="Random Forest",
        dir_name="random_forest"
    )
