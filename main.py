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


# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"data/processed/{timestamp}.csv"
# df.to_csv(filename, index=False)

df = pd.read_csv('./data/processed/2025-04-14_06-09-44.csv')
df_n = select_features(df, threshold=0.9)
columns_to_scale = ['PM2.5', 'temperature', 'humidity', 'wind_speed', 'pressure']
df_n = normalize_features(df_n, columns_to_scale, mode='minmax')
print("Before:")
print(df.head())
print(df.tail())
print("After:")
print(df_n.head())
print(df_n.tail())

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"data/processed/{timestamp}.csv"
df_n.to_csv(filename, index=False)