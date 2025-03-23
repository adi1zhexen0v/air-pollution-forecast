from src.extract.sensor_loader import collect_all_stations_data
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = collect_all_stations_data()
print(df.head(10))
