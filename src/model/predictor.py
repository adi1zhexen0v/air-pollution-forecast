import os
import json
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from src.preprocess.scaler import normalize_features
from src.preprocess.feature_selector import select_features

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(base_dir, "outputs", "models", "cnn_lstm_model.keras")
scaler_path = os.path.join(base_dir, "outputs", "models", "scaler_params.json")
raw_data_dir = os.path.join(base_dir, "data", "processed")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_csv = os.path.join(base_dir, "outputs", "predictions", "predicted_pm25_" + timestamp + ".csv")
n_past_days = 30
n_future_days = 7

def find_latest_raw_csv():
  files = []
  for file in glob(os.path.join(raw_data_dir, "before_normalization_*.csv")):
    files.append(file)

  if len(files) == 0:
    raise FileNotFoundError("No before_normalization_*.csv files found.")

  latest_file = max(files, key=os.path.getctime)
  return latest_file


def generate_forecast():
  print("Loading the latest raw input file...")
  latest_file = find_latest_raw_csv()
  df_raw = pd.read_csv(latest_file, parse_dates=["date"])
  print("Using file:", latest_file)

  df_filtered = select_features(df_raw)
  non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']
  feature_cols = []
  for col in df_filtered.columns:
    if col not in non_feature_columns:
      feature_cols.append(col)

  df_scaled = normalize_features(df_filtered, columns_to_scale=feature_cols)

  print("Loading the model...")
  model = load_model(model_path)

  with open(scaler_path, "r") as f:
    scaler_params = json.load(f)

  pm_index = scaler_params["feature_names"].index("PM2.5")
  pm_mean = scaler_params["mean"][pm_index]
  pm_std = scaler_params["scale"][pm_index]

  predictions = []

  for name, group in df_scaled.groupby("station_name"):
    group = group.sort_values("date")

    if len(group) < n_past_days:
      print("Not enough data for", name)
      continue

    last_block = group.iloc[-n_past_days:]
    input_seq = last_block[feature_cols].values.astype(np.float32)
    input_seq = np.expand_dims(input_seq, axis=0)

    pred = model.predict(input_seq)[0]

    last_date = pd.to_datetime(group["date"].iloc[-1])

    for i in range(n_future_days):
      pred_date = last_date + timedelta(days=i + 1)
      original_pm = pred[i] * pm_std + pm_mean
      prediction = {
        "date": pred_date.strftime("%Y-%m-%d"),
        "station_name": name,
        "latitude": group["latitude"].iloc[0],
        "longitude": group["longitude"].iloc[0],
        "PM2.5": round(float(original_pm), 2)
      }
      predictions.append(prediction)

  os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  df_forecast = pd.DataFrame(predictions)
  df_forecast.to_csv(output_csv, index=False)
  print("Forecast saved to:", output_csv)
