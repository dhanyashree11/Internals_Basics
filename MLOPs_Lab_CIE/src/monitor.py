import json
import pandas as pd

train = pd.read_csv("data/training_data.csv")

train_track = train["track_count"].mean()
train_sample = train["sample_rate_khz"].mean()

logs = []

with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line)["input"])

df = pd.DataFrame(logs)

live_track = df["track_count"].mean()
live_sample = df["sample_rate_khz"].mean()

alerts = []

def check(feature, train_val, live_val, threshold):
    shift = abs(live_val - train_val)
    status = "ALERT" if shift > threshold else "OK"
    return {
        "feature": feature,
        "train_mean": train_val,
        "live_mean": live_val,
        "shift": shift,
        "threshold": threshold,
        "status": status
    }

alerts.append(check("track_count", train_track, live_track, 6.17))
alerts.append(check("sample_rate_khz", train_sample, live_sample, 31.82))

output = {
    "total_predictions": len(df),
    "mean_prediction": 0,
    "drift_detected": any(a["status"]=="ALERT" for a in alerts),
    "alerts": alerts
}

with open("results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Monitoring Done")
