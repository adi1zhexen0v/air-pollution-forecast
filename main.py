import pandas as pd
from datetime import datetime
from src.extract.sensor_loader import collect_all_stations_data
from src.weather.open_meteo_fetcher import add_weather_columns
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import normalize_features
from src.visualization.scaling_visualization import save_scaling_histograms

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# df = collect_all_stations_data()
# df = add_weather_columns(df)
# df = select_features(df, threshold=0.9)

df = pd.read_csv('data/processed/2025-03-24_15-07-23.csv')

columns_to_scale = ['PM2.5', 'PM10', 'temperature', 'humidity', 'wind_speed', 'pressure']
df_raw = df.copy()
df_standard = normalize_features(df_raw, columns_to_scale, mode='standard')
df_minmax = normalize_features(df_raw, columns_to_scale, mode='minmax')
save_scaling_histograms(df_raw, df_standard, df_minmax, columns_to_scale)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"data/processed/{timestamp}.csv"
df.to_csv(filename, index=False)

print(df)
