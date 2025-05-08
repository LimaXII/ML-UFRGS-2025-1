from sklearn.tree import DecisionTreeRegressor

from processing import plot_graph, evaluate

def train_decision_tree_model(X_train, X_test, y_train, y_test):

    # Inicializa o Decision Tree Regressor (sem poda)
    model = DecisionTreeRegressor(random_state=30)
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "Decision Tree", "decision_tree")
