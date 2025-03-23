import pandas as pd
from datetime import datetime
from src.extract.sensor_loader import collect_all_stations_data
from src.weather.open_meteo_fetcher import add_weather_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = collect_all_stations_data()
df = add_weather_columns(df)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"data/processed/{timestamp}.csv"
df.to_csv(filename, index=False)

print(df)
