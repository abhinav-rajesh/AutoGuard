import pandas as pd
import numpy as np

num_samples=100

data = {
    "car_id": [f"C{id:03d}" for id in range(1, num_samples + 1)],
    "total_km": np.random.randint(5000, 200000, num_samples),
    "last_service_km": np.random.randint(0, 190000, num_samples),
    "tyre_wear_level": np.random.randint(10, 100, num_samples),
    "engine_oil_quality": np.random.randint(10, 100, num_samples),
    "engine_heat": np.random.normal(90, 10, num_samples).astype(int),
    "battery_level": np.random.randint(20, 100, num_samples),
}

data["predicted_service_km"] = (
    data["total_km"] - data["last_service_km"] + np.random.randint(5000, 15000, num_samples)
)
data["predicted_tyre_km"] = (
    (100 - data["tyre_wear_level"]) * np.random.randint(50, 100, num_samples)
)
df = pd.DataFrame(data)
df.to_csv("data/car_maintenance_data.csv", index=False)
print("✅ Dataset saved to data/car_maintenance_data.csv")