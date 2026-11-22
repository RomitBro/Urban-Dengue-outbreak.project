from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

matches = sorted(Path(".").glob("hackathon_preli_dataset*.csv"))
if not matches:
  raise FileNotFoundError("No hackathon_preli_dataset*.csv file found.")

df = pd.read_csv(matches[0])
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "ward_id", "dengue_cases"]).sort_values(
    ["ward_id", "date"]
)

# Weekly lag features (1 to 3 weeks)
for lag in [1, 2, 3]:
  df[f"rain_lag_{lag}w"] = df.groupby("ward_id")["rainfall_mm"].shift(lag)
  df[f"temp_lag_{lag}w"] = df.groupby("ward_id")["avg_temp_c"].shift(lag)
  df[f"humidity_lag_{lag}w"] = df.groupby("ward_id")[
      "relative_humidity_pct"
  ].shift(lag)
  df[f"cases_lag_{lag}w"] = df.groupby("ward_id")["dengue_cases"].shift(lag)

# Drop early rows where lags are NaN
df_model = df.dropna().copy()

feature_cols = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "rainfall_mm",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "relative_humidity_pct",
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "temp_lag_1w",
    "temp_lag_2w",
    "temp_lag_3w",
    "humidity_lag_1w",
    "humidity_lag_2w",
    "humidity_lag_3w",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
]

# Chronological split: train through 2023, validate on 2024-2025
train_df = df_model[df_model["date"] < "2024-01-01"]
val_df = df_model[df_model["date"] >= "2024-01-01"]

X_train, y_train = train_df[feature_cols], train_df["dengue_cases"]
X_val, y_val = val_df[feature_cols], val_df["dengue_cases"]

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = root_mean_squared_error(y_val, preds)

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
print(f"Validation MAE: {mae:.2f} | RMSE: {rmse:.2f}")