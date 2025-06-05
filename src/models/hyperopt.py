import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def optimize_decision_tree(X, y, n_trials=30):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = DecisionTreeRegressor(random_state=30, **params)
        score = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_squared_error"
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params for Decision Tree:", study.best_params)
    best_model = DecisionTreeRegressor(random_state=30, **study.best_params)
    best_model.fit(X, y)
    return best_model

def optimize_random_forest(X, y, n_trials=5):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }
        model = RandomForestRegressor(random_state=30, **params)
        score = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_squared_error"
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params for Random Forest:", study.best_params)
    best_model = RandomForestRegressor(random_state=30, **study.best_params)
    best_model.fit(X, y)
    return best_model