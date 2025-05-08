from sklearn.ensemble import HistGradientBoostingRegressor
from processing import plot_graph, evaluate


def train_histogram_gb(X_train, X_test, y_train, y_test):
    

    model = HistGradientBoostingRegressor(random_state=30)
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "Histogram Gradient Boosting", "histogram_gb")

