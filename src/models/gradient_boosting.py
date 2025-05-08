from sklearn.ensemble import GradientBoostingRegressor
from processing import plot_graph

def train_gradient_boosting_model(X_train, X_test, y_train, y_test):

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=30
    )
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "Gradiente Boosting", "gradient_boosting")
