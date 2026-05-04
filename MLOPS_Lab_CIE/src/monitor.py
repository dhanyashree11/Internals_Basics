import json
import pandas as pd

train = pd.read_csv("data/training_data.csv")

train_mean_track = train["track_count"].mean()
train_mean_sr = train["sample_rate_khz"].mean()

logs = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

live_df = pd.DataFrame([l["input"] for l in logs])

live_track = live_df["track_count"].mean()
live_sr = live_df["sample_rate_khz"].mean()

alerts = []

def check(feature, train_mean, live_mean, threshold):
    shift = abs(live_mean - train_mean)
    status = "ALERT" if shift > threshold else "OK"
    return {
        "feature": feature,
        "train_mean": train_mean,
        "live_mean": live_mean,
        "shift": shift,
        "threshold": threshold,
        "status": status
    }

alerts.append(check("track_count", train_mean_track, live_track, 6.17))
alerts.append(check("sample_rate_khz", train_mean_sr, live_sr, 31.82))

mean_prediction = sum(l["prediction"] for l in logs) / len(logs)

output = {
    "total_predictions": len(logs),
    "mean_prediction": mean_prediction,
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

with open("results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=4)
