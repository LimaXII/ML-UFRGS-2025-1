from sklearn.neural_network import MLPRegressor

from processing import evaluate, plot_graph


def train_mlp_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 8: MLP Regressor \n")

    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=500,
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
        graph_name="MLP Regressor",
        dir_name="mlp"
    )
