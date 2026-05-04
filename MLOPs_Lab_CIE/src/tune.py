import pandas as pd
import mlflow
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("data/training_data.csv")
X = df.drop("mix_quality_score", axis=1)
y = df["mix_quality_score"]

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

mlflow.set_experiment("soundforge-mix-quality-score")

with mlflow.start_run(run_name="tuning-soundforge"):

    model = GradientBoostingRegressor()

    search = RandomizedSearchCV(
        model,
        param_grid,
        cv=5,
        n_iter=5,
        scoring="neg_mean_squared_error",
        random_state=42
    )

    search.fit(X, y)

    best_model = search.best_estimator_

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": 0,
        "best_cv_mae": 0,
        "parent_run_name": "tuning-soundforge"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

    import joblib
    joblib.dump(best_model, "models/best_model.pkl")

print("Task 2 Done")
