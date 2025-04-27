import os
import pandas as pd
from datetime import datetime
from src.extract.sensor_loader import collect_all_stations_data
from src.preprocess.filter_data import filter_dataframe
from src.weather.open_meteo_fetcher import add_weather_columns
from src.preprocess.time_features import extract_time_features
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import normalize_features
from src.model.cnn_lstm_trainer import train_cnn_lstm_model
from src.model.predictor import generate_forecast
from src.visualization.scaling_visualization import save_pm25_histograms

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
    print("Stage 1: Data Collection and Preprocessing")
    df = collect_all_stations_data()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/all_stations.csv", index=False)
    print("Collected station data saved to: data/processed/all_stations.csv")

    print("Adding weather features...")
    df = add_weather_columns(df)

    print("Filtering outliers and missing values...")
    df = df.dropna(subset=["PM2.5", "temperature", "humidity", "wind_speed", "pressure"], how="all")

    print("Extracting time features...")
    df = extract_time_features(df)

    df = filter_dataframe(df, min_station_coverage=0.9, pm25_max=300)

    print("Selecting best features...")
    df_filtered = select_features(df)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    before_filename = f"data/processed/before_normalization_{timestamp}.csv"
    df_filtered.to_csv(before_filename, index=False)
    print("Filtered features saved to:", before_filename)

    print("Normalizing selected features...")
    non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']
    feature_cols = []
    for col in df_filtered.columns:
        if col not in non_feature_columns:
            feature_cols.append(col)

    df_normalized = normalize_features(df_filtered, columns_to_scale=feature_cols)

    save_pm25_histograms(df_raw=df_filtered, df_standard=df_normalized)

    after_filename = f"data/processed/after_normalization_{timestamp}.csv"
    df_normalized.to_csv(after_filename, index=False)
    print("Normalized features saved to:", after_filename)

    print("Stage 2: Model Training")
    train_cnn_lstm_model()

    print("Stage 3: Generating Forecast")
    generate_forecast()

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
