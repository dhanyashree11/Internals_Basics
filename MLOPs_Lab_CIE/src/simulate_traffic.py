import requests
import random

url = "http://127.0.0.1:8080/infer"

# normal
for _ in range(35):
    data = {
        "track_count": random.randint(2,32),
        "sample_rate_khz": random.randint(22,192),
        "dynamic_range_db": random.uniform(6,20),
        "is_mastered": random.randint(0,1)
    }
    requests.post(url, json=data)

# drift
# drift (VALID BUT SHIFTED)
for _ in range(15):
    data = {
        "track_count": random.randint(50,80),   # upper bound
        "sample_rate_khz": random.randint(250,400),  # upper bound
        "dynamic_range_db": random.uniform(6,20),
        "is_mastered": random.randint(0,1)
    }
    requests.post(url, json=data)
print("Traffic sent")
