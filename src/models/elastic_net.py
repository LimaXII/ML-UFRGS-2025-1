from sklearn.linear_model import ElasticNet
from processing import plot_graph, evaluate

def train_elastic_net(X_train, X_test, y_train, y_test):

    model = ElasticNet(random_state=30)
    model.fit(X_train, y_train)

    y_test_real, y_pred = evaluate(model, X_test, y_test)

    plot_graph(y_test_real, y_pred, "ElasticNet", "elastic_net")
