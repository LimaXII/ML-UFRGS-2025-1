import time

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RESULTS_FILE = "resultados_modelos.txt"


def evaluate_on_test(model, X_test, y_test, file_mode="a"):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open(RESULTS_FILE, file_mode) as f:
        f.write("\nAvaliação no conjunto de teste:\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R²: {r2:.2f}\n")
    return rmse, mae, r2


def optimize_decision_tree(X, y, n_trials=30):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trial_times = []
    trial_scores = []

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    with open(RESULTS_FILE, "w") as f_out:
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", DecisionTreeRegressor(random_state=30, **params))
            ])

            start = time.time()
            score = cross_val_score(
                pipeline, X_trainval, y_trainval,
                cv=inner_cv,
                scoring="neg_mean_squared_error"
            ).mean()
            end = time.time()

            trial_times.append(end - start)
            trial_scores.append(score)

            f_out.write(
                f"[Decision Tree] Trial with params {params} took {end - start:.2f} seconds.\n")
            return score

        study = optuna.create_study(direction="maximize")
        total_start = time.time()
        study.optimize(objective, n_trials=n_trials)
        total_end = time.time()

        f_out.write(f"Best params for Decision Tree: {study.best_params}\n")
        f_out.write(
            f"Total optimization time: {total_end - total_start:.2f} seconds.\n")

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", DecisionTreeRegressor(random_state=30, **study.best_params))
    ])
    final_pipeline.fit(X_trainval, y_trainval)

    evaluate_on_test(final_pipeline, X_test, y_test, file_mode="a")
    return final_pipeline


def optimize_random_forest(X, y, n_trials=30):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trial_times = []
    trial_scores = []

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    with open(RESULTS_FILE, "a") as f_out:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            }

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(random_state=30, **params))
            ])

            start = time.time()
            score = cross_val_score(
                pipeline, X_trainval, y_trainval,
                cv=inner_cv,
                scoring="neg_mean_squared_error"
            ).mean()
            end = time.time()

            trial_times.append(end - start)
            trial_scores.append(score)

            f_out.write(
                f"[Random Forest] Trial with params {params} took {end - start:.2f} seconds.\n")
            return score

        study = optuna.create_study(direction="maximize")
        total_start = time.time()
        study.optimize(objective, n_trials=n_trials)
        total_end = time.time()

        f_out.write(f"Best params for Random Forest: {study.best_params}\n")
        f_out.write(
            f"Total optimization time: {total_end - total_start:.2f} seconds.\n")

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(random_state=30, **study.best_params))
    ])
    final_pipeline.fit(X_trainval, y_trainval)

    evaluate_on_test(final_pipeline, X_test, y_test, file_mode="a")
    return final_pipeline
