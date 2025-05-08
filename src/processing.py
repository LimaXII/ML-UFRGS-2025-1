import matplotlib.pyplot as plt
import seaborn as sns

def plot_graph(y_test_real, y_pred, graph_name, dir_name) :
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.6)
    plt.plot([y_test_real.min(), y_test_real.max()],
             [y_test_real.min(), y_test_real.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted "+ graph_name)
    plt.tight_layout()
    plt.savefig("graphs/"+ dir_name + "/actual_vs_predicted.png")
    plt.close()

    residuals = y_test_real - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted "+ graph_name)
    plt.tight_layout()
    plt.savefig("graphs/"+ dir_name + "/residuals_vs_predicted.png")
    plt.close()