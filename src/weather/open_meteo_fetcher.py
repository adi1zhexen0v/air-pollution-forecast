import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_hourly_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure"
        ],
        "timezone": "Asia/Almaty"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(3).ValuesAsNumpy()
    })

    df['date'] = df['date'].dt.tz_localize(None)
    return df


def get_daily_weather(lat: float, lon: float, station_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    hourly_df = get_hourly_weather(lat, lon, start_date, end_date)
    hourly_df['station_name'] = station_name
    hourly_df['day'] = pd.to_datetime(hourly_df['date'].dt.date)

    daily_df = hourly_df.groupby(['day', 'station_name']).agg({
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'wind_speed_10m': 'mean',
        'surface_pressure': 'mean'
    }).reset_index()

    return daily_df


def add_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
  start_date = df['date'].min().strftime('%Y-%m-%d')
  end_date = df['date'].max().strftime('%Y-%m-%d')

  stations = df[['station_name', 'latitude', 'longitude']].drop_duplicates()
  all_weather_daily = []

  for _, row in stations.iterrows():
    daily_df = get_daily_weather(
      lat=row['latitude'],
      lon=row['longitude'],
      station_name=row['station_name'],
      start_date=start_date,
      end_date=end_date
    )

    all_weather_daily.append(daily_df)

  weather_df = pd.concat(all_weather_daily, ignore_index=True)

  df['day'] = pd.to_datetime(df['date'].dt.date)
  weather_df['day'] = pd.to_datetime(weather_df['day'])

  merged_df = pd.merge(df, weather_df, on=['day', 'station_name'], how='left')
  merged_df.drop(columns=['day'], inplace=True)

  merged_df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "surface_pressure": "pressure"
  }, inplace=True)

  for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
    if col in merged_df.columns:
      merged_df[col] = merged_df[col].round(2)

  return merged_df
