from sklearn.tree import DecisionTreeRegressor

from processing import evaluate, plot_graph


def train_decision_tree_model(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 2: Decision Tree \n")

    # Inicializa o Decision Tree Regressor (sem poda)
    model = DecisionTreeRegressor(
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
        graph_name="Decision Tree",
        dir_name="decision_tree"
    )
