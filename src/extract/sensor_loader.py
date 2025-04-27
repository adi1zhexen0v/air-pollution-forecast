import os
import pandas as pd
from data.stations import stations

def load_parameter_column_from_file(file_path: str, param_name: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
    except Exception as e:
        print(f"Error while reading file '{file_path}': {e}")
        return None

    if 'date' in df.columns and 'median' in df.columns:
        df = df[['date', 'median']].copy()
        df.rename(columns={'median': param_name}, inplace=True)
        return df
    else:
        print(f"File '{file_path}' skipped: missing 'date' or 'median' column.")
        return None

def load_station_data_from_csv(folder_path: str, station_name: str, latitude: float, longitude: float) -> pd.DataFrame:
    param_dfs = {}

    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_files.append(file)

    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        if os.path.getsize(file_path) == 0:
            print(f"Skipped empty file: {file_path}")
            continue

        param_name = file.split('_')[-1].replace('.csv', '')
        param_df = load_parameter_column_from_file(file_path, param_name)

        if param_df is not None:
            param_dfs[param_name] = param_df

    # If no valid files found, return empty DataFrame
    if len(param_dfs) == 0:
        print(f"No valid parameter files found for station '{station_name}'.")
        return pd.DataFrame()

    merged_df = None
    for param_name in param_dfs:
        if merged_df is None:
            merged_df = param_dfs[param_name]
        else:
            merged_df = pd.merge(merged_df, param_dfs[param_name], on='date', how='outer')

    merged_df['station_name'] = station_name
    merged_df['latitude'] = latitude
    merged_df['longitude'] = longitude

    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.drop_duplicates(subset=['date', 'station_name'])
    merged_df = merged_df.sort_values('date')

    return merged_df

def collect_all_stations_data() -> pd.DataFrame:
    all_stations_data = []

    for station in stations:
        df = load_station_data_from_csv(
            folder_path=station['folder'],
            station_name=station['name'],
            latitude=station['lat'],
            longitude=station['lon']
        )
        if not df.empty:
            all_stations_data.append(df)
            print(f"Collected data for station: {station['name']}")
        else:
            print(f"No data collected for station: {station['name']}")

    # Combine all station DataFrames
    if len(all_stations_data) > 0:
        combined_df = pd.concat(all_stations_data, ignore_index=True)
        print(f"Collected data from {len(all_stations_data)} stations.")
        return combined_df
    else:
        print("No data collected from any station.")
        return pd.DataFrame()
