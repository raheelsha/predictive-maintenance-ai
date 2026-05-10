import numpy as np
import pandas as pd
from datetime import datetime , timedelta
import os

np.random.seed(42)
TOTAL_HOURS = 720              # 30 days of data
FALIURE_START=648           # Failure starts after 27 days (648 hours)

timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(TOTAL_HOURS)]

normal_hours = FALIURE_START
failure_hours = TOTAL_HOURS - FALIURE_START

temperature =np.concatenate([
    np.random.normal(70 ,2 , normal_hours) ,  # Normal operation
    np.linspace(70, 95 , failure_hours) + np.random.normal(0,1, failure_hours)
])
vibration = np.concatenate([
    np.random.normal(0.5, 0.05, normal_hours),
    np.linspace(0.5, 1.8, failure_hours) + np.random.normal(0, 0.05, failure_hours)
])

pressure = np.concatenate([
    np.random.normal(30, 1, normal_hours),
    np.linspace(30, 18, failure_hours) + np.random.normal(0, 0.5, failure_hours)
])

rotation_speed = np.concatenate([
    np.random.normal(3000, 50, normal_hours),
    np.linspace(3000, 2200, failure_hours) + np.random.normal(0, 30, failure_hours)
])

# The AI will use this column to learn the difference between "Normal" and "Breaking."
label =np.concatenate([
    np.zeros(normal_hours),  # Normal operation
    np.ones(failure_hours)    # Failure
])
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': np.round(temperature, 2),
    'vibration': np.round(vibration, 3),
    'pressure': np.round(pressure, 2),
    'rotation_speed': np.round(rotation_speed, 1),
    'failure': label.astype(int)
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/sensor_data.csv', index=False)
print("Sensor data generated and saved to 'data/sensor_data.csv'")
os.makedirs('data', exist_ok=True)
df.to_csv('data/sensor_data.csv', index=False)

print(f"Dataset generated: {df.shape[0]} rows")
print(f"Normal readings: {int(df['failure'].value_counts()[0])}")
print(f"Failure readings: {int(df['failure'].value_counts()[1])}")
print(df.tail())

