import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import json

from src.preprocess.scaler import normalize_features
from src.preprocess.feature_selector import select_features

# === Абсолютные пути ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "cnn_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "models", "scaler_params.json")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "predictions", f"predicted_pm25_{timestamp}.csv")


# === Настройки прогноза ===
N_PAST_DAYS = 30
N_FUTURE_DAYS = 7

def find_latest_raw_csv():
    files = glob(os.path.join(RAW_DATA_DIR, "before_normalization_*.csv"))
    if not files:
        raise FileNotFoundError("No before_normalization_*.csv files found.")
    return max(files, key=os.path.getctime)

def generate_forecast():
    print("[INFO] Loading latest raw input file...")
    latest_file = find_latest_raw_csv()
    df_raw = pd.read_csv(latest_file, parse_dates=["date"])
    print(f"[INFO] Using file: {latest_file}")

    # === Фильтрация и нормализация ===
    df_filtered = select_features(df_raw)
    non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']
    feature_cols = [col for col in df_filtered.columns if col not in non_feature_columns]

    df_scaled = normalize_features(df_filtered, columns_to_scale=feature_cols, mode="standard")

    # === Загрузка модели
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # === Загрузка параметров нормализации
    with open(SCALER_PATH) as f:
        scaler_params = json.load(f)
    pm_index = scaler_params["feature_names"].index("PM2.5")
    pm_mean = scaler_params["mean"][pm_index]
    pm_std = scaler_params["scale"][pm_index]

    predictions = []

    for name, group in df_scaled.groupby("station_name"):
        group = group.sort_values("date")
        if len(group) < N_PAST_DAYS:
            print(f"[SKIP] Not enough data for {name}")
            continue

        last_block = group.iloc[-N_PAST_DAYS:]
        input_seq = last_block[feature_cols].values.astype(np.float32)
        input_seq = np.expand_dims(input_seq, axis=0)  # shape (1, 30, N)

        pred = model.predict(input_seq)[0]  # shape (7,)

        last_date = pd.to_datetime(group["date"].iloc[-1])

        for i in range(N_FUTURE_DAYS):
            pred_date = last_date + timedelta(days=i+1)
            original_pm = pred[i] * pm_std + pm_mean
            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "station_name": name,
                "latitude": group["latitude"].iloc[0],
                "longitude": group["longitude"].iloc[0],
                "PM2.5": round(float(original_pm), 2)
            })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_forecast = pd.DataFrame(predictions)
    df_forecast.to_csv(OUTPUT_CSV, index=False)
    print(f"[SAVED] Forecast saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_forecast()
