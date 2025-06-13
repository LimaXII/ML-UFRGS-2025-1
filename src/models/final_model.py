import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Funções de plotagem e avaliação
def plot_feature_importance(pipeline, feature_names, filename="rf_feature_importance.png"):
    model = pipeline.named_steps['regressor']
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances}
    ).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Importância dos Atributos no Modelo Final', fontsize=16)
    plt.xlabel('Importância', fontsize=12)
    plt.ylabel('Atributo', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de importância dos atributos salvo como '{filename}'.")

def plot_residuals(pipeline, X_test, y_test, filename="rf_residuals_plot.png"):
    y_pred = pipeline.predict(X_test)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
    plt.title('Valores Previstos vs. Valores Reais', fontsize=16)
    plt.xlabel('Valores Reais (Preço Log)', fontsize=12)
    plt.ylabel('Valores Previstos (Preço Log)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de resíduos salvo como '{filename}'.")

def evaluate_final_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n--- Avaliação do Modelo Final no Conjunto de Teste ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    print("-------------------------------------------------------")

def analyze_final_model(X, y, best_rf_params):
    # Dividir os dados em treino e teste (usando o mesmo random_state de antes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Criar o pipeline do modelo final
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(random_state=30, **best_rf_params))
    ])

    # Treinar o modelo final
    print("Treinando o modelo final com os melhores hiperparâmetros...")
    final_pipeline.fit(X_train, y_train)
    print("Treinamento concluído.")

    # Avaliar e gerar os gráficos
    evaluate_final_model(final_pipeline, X_test, y_test)

    plot_feature_importance(
        final_pipeline,
        X.columns, # Passa os nomes das colunas para o gráfico
        filename="rf_feature_importance.png"
    )

    plot_residuals(
        final_pipeline,
        X_test,
        y_test,
        filename="rf_residuals_plot.png"
    )

    print("\nAnálise concluída! Gráficos salvos na raiz do projeto.")