from sklearn.linear_model import LinearRegression
from processing import plot_graph, evaluate

def train_linear_regression_model(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "Linear Regression", "linear_regression")
