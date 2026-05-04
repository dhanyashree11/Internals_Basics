import mlflow
import json
import pandas as pd
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

model = GradientBoostingRegressor()

search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=5, scoring="neg_mean_absolute_error")

mlflow.set_experiment("soundforge-mix-quality-score")

with mlflow.start_run(run_name="tuning-soundforge"):
    search.fit(X, y)

    output = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": -search.best_score_,
        "best_cv_mae": -search.best_score_,
        "parent_run_name": "tuning-soundforge"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)
