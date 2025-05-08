from sklearn.neighbors import KNeighborsRegressor
from processing import plot_graph, evaluate

def train_knn_model(X_train, X_test, y_train, y_test):

    # Inicializa o Knn, com k=2.
    knn = KNeighborsRegressor(
        n_neighbors=2
    )
    knn.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    plot_graph(y_test_real, y_pred, "KNN", "knn")
