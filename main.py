import os
import pandas as pd
from datetime import datetime
from src.extract.sensor_loader import collect_all_stations_data
from src.preprocess.filter_data import filter_dataframe
from src.weather.open_meteo_fetcher import add_weather_columns
from src.preprocess.time_features import extract_time_features
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import normalize_features
from src.model.cnn_lstm_model import train_cnn_lstm_model
from src.model.predictor import generate_forecast
from src.visualization.scaling_visualization import save_pm25_standard_scaling_histogram

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
    # print("\n=== [STAGE 1] Data Collection and Preprocessing ===")
    # df = collect_all_stations_data()
    # df.to_csv("data/processed/all_stations.csv")
    #
    # print("[INFO] Adding weather features...")
    # df = add_weather_columns(df)
    #
    # print("[INFO] Filtering outliers and NaNs...")
    # df = df.dropna(subset=["PM2.5", "temperature", "humidity", "wind_speed", "pressure"], how="all")
    #
    # print("[INFO] Extracting time features...")
    # df = extract_time_features(df)
    #
    # df = filter_dataframe(df, 0.9, 300)
    #
    # print("[INFO] Selecting best features...")
    # df_filtered = select_features(df)
    #
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # os.makedirs("data/processed", exist_ok=True)
    #
    # before_filename = f"data/processed/before_normalization_{timestamp}.csv"
    # df_filtered.to_csv(before_filename, index=False)
    # print(f"[SAVED] Filtered (raw) features saved to {before_filename}")
    #
    # print("[INFO] Normalizing selected features...")
    # non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']
    # feature_cols = [col for col in df_filtered.columns if col not in non_feature_columns]
    #
    # df_normalized = normalize_features(df_filtered, columns_to_scale=feature_cols, mode="standard")
    #
    # save_pm25_standard_scaling_histogram(
    #     df_raw=df_filtered,
    #     df_standard=df_normalized
    # )
    #
    # after_filename = f"data/processed/after_normalization_{timestamp}.csv"
    # df_normalized.to_csv(after_filename, index=False)
    # print(f"[SAVED] Normalized features saved to {after_filename}")

    print("\n=== [STAGE 2] Model Training ===")
    train_cnn_lstm_model()

    print("\n=== [STAGE 3] Generating Forecast ===")
    # generate_forecast()

    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
