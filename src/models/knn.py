from sklearn.neighbors import KNeighborsRegressor

from processing import evaluate, plot_graph


def train_knn_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 6: KNN \n")

    print("Treinamendo com k=2. \n")
    # Inicializa o Knn, com k=2.
    knn = KNeighborsRegressor(
        n_neighbors=2
    )
    knn.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(
        model=knn,
        X_test=X_test,
        y_test=y_test
    )

    plot_graph(
        y_test_real=y_test_real,
        y_pred=y_pred,
        graph_name="KNN",
        dir_name="knn"
    )
