import pandas as pd
import mlflow
import mlflow.sklearn
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("data/training_data.csv")

X = df.drop("mix_quality_score", axis=1)
y = df["mix_quality_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("soundforge-mix-quality-score")

results = []

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})
        mlflow.set_tag("priority", "high")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

models = {
    "Ridge": Ridge(),
    "GradientBoosting": GradientBoostingRegressor()
}

for name, model in models.items():
    evaluate(model, name)

best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "soundforge-mix-quality-score",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 Done")
