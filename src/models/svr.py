from sklearn.svm import SVR

from processing import evaluate, plot_graph


def train_svr_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 10: SVR \n")

    # Inicializa o SVR com kernel RBF (padrão)
    model = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.2
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
        graph_name="SVR",
        dir_name="svr"
    )
