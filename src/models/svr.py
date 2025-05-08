from sklearn.svm import SVR
from processing import plot_graph, evaluate

def train_svr_model(X_train, X_test, y_train, y_test):

    # Inicializa o SVR com kernel RBF (padrão)
    model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "SVR", "svr")
