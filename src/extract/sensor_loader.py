import pandas as pd
from data.stations import stations
import os

def load_parameter_column_from_file(file_path: str, param_name: str):
    try:
        df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
    except Exception as e:
        print(f"Error during reading {file_path}: {e}")
        return None

    if 'date' in df.columns and 'median' in df.columns:
        df = df[['date', 'median']].copy()
        df.rename(columns={'median': param_name}, inplace=True)
        return df
    else:
        print(f"File {file_path} skipped: no necessary columns")
        return None

def load_station_data_from_csv(folder_path: str, station_name: str, latitude: float, longitude: float) -> pd.DataFrame:
    param_dfs = {}
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_files.append(file)

    for file in csv_files:
        param_name = file.split('_')[-1].replace('.csv', '')
        file_path = os.path.join(folder_path, file)
        param_df = load_parameter_column_from_file(file_path, param_name)
        if param_df is not None:
            param_dfs[param_name] = param_df

    merged_df = None
    for df in param_dfs.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')

    if merged_df is not None:
        merged_df['station_name'] = station_name
        merged_df['latitude'] = latitude
        merged_df['longitude'] = longitude
        merged_df['date'] = pd.to_datetime(merged_df['date'])

        return merged_df
    else:
        return pd.DataFrame()

def collect_all_stations_data():
    station_dfs = []

    for station in stations:
        df = load_station_data_from_csv(
            folder_path=station['folder'],
            station_name=station['name'],
            latitude=station['lat'],
            longitude=station['lon']
        )
        if not df.empty:
            station_dfs.append(df)

    if station_dfs:
        return pd.concat(station_dfs, ignore_index=True)
    else:
        return pd.DataFrame()
