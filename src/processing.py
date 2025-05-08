import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


def evaluate(
    model,
    X,
    y
):
    """
    Avalia o modelo de regressão utilizando as métricas MAE, RMSE e R²,
    com validação cruzada.

    Args:
        model: Modelo de regressão a ser avaliado.
        X: Conjunto de dados de entrada (features).
        y: Conjunto de dados de saída (target).
    """

    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=30)
    y_pred_log = cross_val_predict(model, X, y, cv=cv)

    # Converte os valores de volta da escala log para original
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y)

    # Avaliação do modelo
    mae: float = mean_absolute_error(y_test_real, y_pred)
    rmse: float = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2: float = r2_score(y_test_real, y_pred)

    print("Resultados: \n")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    return y_test_real, y_pred


def plot_graph(
    y_test_real,
    y_pred,
    graph_name,
    dir_name
):
    """
    Plota os gráficos de comparação entre os valores reais e preditos,
    e os resíduos em relação aos valores preditos.

    Args:
        y_test_real: Valores reais do conjunto de teste.
        y_pred: Valores preditos pelo modelo.
        graph_name: Nome do gráfico a ser salvo.
        dir_name: Nome do diretório onde os gráficos serão salvos.
    """

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.6)
    plt.plot([y_test_real.min(), y_test_real.max()],
             [y_test_real.min(), y_test_real.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted " + graph_name)
    plt.tight_layout()
    plt.savefig("graphs/" + dir_name + "/actual_vs_predicted.png")
    plt.close()

    residuals = y_test_real - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted " + graph_name)
    plt.tight_layout()
    plt.savefig("graphs/" + dir_name + "/residuals_vs_predicted.png")
    plt.close()
