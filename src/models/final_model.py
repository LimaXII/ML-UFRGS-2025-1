import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. FUNÇÕES DE PLOTAGEM E AVALIAÇÃO
# ==============================================================================

def plot_feature_importance(pipeline, feature_names, model_name):
    """Gera um gráfico de barras da importância dos atributos."""
    filename = f"{model_name.replace(' ', '_').lower()}_feature_importance.png"
    model = pipeline.named_steps['regressor']
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances}
    ).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis') # Mostra os 20 mais importantes
    plt.title(f'Importância dos Atributos - {model_name}', fontsize=16)
    plt.xlabel('Importância', fontsize=12)
    plt.ylabel('Atributo', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico de importância dos atributos salvo como '{filename}'.")

def plot_model_analysis(pipeline, X_test, y_test_log, model_name):
    """Gera os gráficos de "Actual vs Predicted" e "Residuals vs Predicted"."""
    y_pred_log = pipeline.predict(X_test)
    y_test_original = np.expm1(y_test_log)
    y_pred_original = np.expm1(y_pred_log)

    # Gráfico 1: Actual vs Predicted
    filename_avp = f"{model_name.replace(' ', '_').lower()}_actual_vs_predicted.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_original, y=y_pred_original, alpha=0.6)
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(filename_avp)
    plt.close()
    print(f"Gráfico 'Actual vs Predicted' salvo como '{filename_avp}'.")

    # Gráfico 2: Residuals vs Predicted
    filename_rvp = f"{model_name.replace(' ', '_').lower()}_residuals_vs_predicted.png"
    residuals = y_test_original - y_pred_original
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred_original, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(filename_rvp)
    plt.close()
    print(f"Gráfico 'Residuals vs Predicted' salvo como '{filename_rvp}'.")

def evaluate_on_original_scale(model_name: str, pipeline, X_test, y_test_log):
    """Avalia um pipeline e retorna o RMSE na escala original."""
    y_pred_log = pipeline.predict(X_test)
    y_test_original = np.expm1(y_test_log)
    y_pred_original = np.expm1(y_pred_log)
    
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"\n--- Modelo: {model_name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.2f}")
    print("-----------------------------------")
    return rmse

# ==============================================================================
# 2. FUNÇÃO PRINCIPAL DE ANÁLISE
# ==============================================================================

def compare_final_models(X, y, dt_params, rf_params):
    """
    Treina, compara os 4 modelos finalistas e gera gráficos para o melhor.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Decision Tree (Padrão)": DecisionTreeRegressor(random_state=30),
        "Decision Tree (Otimizado)": DecisionTreeRegressor(random_state=30, **dt_params),
        "Random Forest (Padrão)": RandomForestRegressor(random_state=30),
        "Random Forest (Otimizado)": RandomForestRegressor(random_state=30, **rf_params)
    }

    results = {}
    trained_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        
        print(f"Treinando {model_name}...")
        pipeline.fit(X_train, y_train)
        
        rmse = evaluate_on_original_scale(model_name, pipeline, X_test, y_test)
        
        # Guarda os resultados e o pipeline treinado
        results[model_name] = rmse
        trained_pipelines[model_name] = pipeline

    # Encontra o nome do modelo com o menor RMSE
    best_model_name = min(results, key=results.get)
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f"\nGerando gráficos para o melhor modelo: {best_model_name} (RMSE: {results[best_model_name]:.2f})")

    # Gera os gráficos para o melhor modelo
    plot_feature_importance(
        best_pipeline,
        X.columns,
        model_name=best_model_name
    )
    
    plot_model_analysis(
        best_pipeline,
        X_test,
        y_test,
        model_name=best_model_name
    )