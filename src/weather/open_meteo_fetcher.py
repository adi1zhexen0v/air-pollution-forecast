import time

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

cache = requests_cache.CachedSession(".cache", expire_after=-1)
session = retry(cache, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=session)

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

    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
        "pressure": hourly.Variables(3).ValuesAsNumpy()
    })

    df["date"] = df["date"].dt.tz_localize(None)
    df["temperature"] = df["temperature"].round(2)
    df["humidity"] = df["humidity"].round(2)
    df["wind_speed"] = df["wind_speed"].round(2)
    df["pressure"] = df["pressure"].round(2)
    df["latitude"] = lat
    df["longitude"] = lon

    return df

def add_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    stations = df[["station_name", "latitude", "longitude"]].drop_duplicates()
    weather_all = []

    for i, (_, station) in enumerate(stations.iterrows()):
        print(f"[{i + 1}/{len(stations)}] Fetching weather for {station['station_name']}")

        time.sleep(15)

        weather_df = get_hourly_weather(
            lat=station["latitude"],
            lon=station["longitude"],
            start_date=start_date,
            end_date=end_date
        )
        weather_df["station_name"] = station["station_name"]
        weather_all.append(weather_df)

    weather_df = pd.concat(weather_all, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.tz_localize(None)

    merged_df = pd.merge(
        df,
        weather_df,
        on=["date", "station_name", "latitude", "longitude"],
        how="left"
    )

    return merged_df
