import requests
import random

url = "http://127.0.0.1:8080/infer"

for _ in range(35):
    data = {
        "track_count": random.randint(2, 32),
        "sample_rate_khz": random.randint(22, 192),
        "dynamic_range_db": random.uniform(6, 20),
        "is_mastered": random.randint(0, 1)
    }
    requests.post(url, json=data)

for _ in range(15):
    data = {
        "track_count": random.randint(30, 70),
        "sample_rate_khz": random.randint(200, 350),
        "dynamic_range_db": random.uniform(6, 20),
        "is_mastered": random.randint(0, 1)
    }
    requests.post(url, json=data)
