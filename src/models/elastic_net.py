from sklearn.linear_model import ElasticNet

from processing import evaluate, plot_graph


def train_elastic_net(
    X_train,
    X_test,
    y_train,
    y_test
):
    print("\nModelo número 3: Elastic Net \n")

    model = ElasticNet(
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
        graph_name="ElasticNet",
        dir_name="elastic_net"
    )
