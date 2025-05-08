
from sklearn.linear_model import BayesianRidge
from processing import plot_graph, evaluate

def train_bayesian_ridge(X_train, X_test, y_train, y_test):

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    print(f"[Bayesian Ridge] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    plot_graph(y_test_real, y_pred, "Bayesian Ridge", "bayesian_ridge")

    
