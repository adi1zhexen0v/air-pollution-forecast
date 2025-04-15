import pandas as pd

def filter_dataframe(df: pd.DataFrame, min_station_coverage: float = 0.9, pm25_max: float = 300):
    print("\n[INFO] Filtering by dynamic common timeframe based on station coverage...")

    # 1. Удалим экстремальные или ошибочные значения
    df = df[(df["PM2.5"] >= 0) & (df["PM2.5"] <= pm25_max)]
    print(f"[INFO] Filtered PM2.5 values: kept within 0–{pm25_max}")

    # 2. Фильтрация по покрытию станций
    total_station_count = df["station_name"].nunique()
    min_required_stations = int(total_station_count * min_station_coverage)

    station_counts_by_date = df.groupby("date")["station_name"].nunique()
    valid_dates = station_counts_by_date[station_counts_by_date >= min_required_stations].index

    if valid_dates.empty:
        print("[WARNING] No dates meet the required station coverage. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    first_valid_date = valid_dates.min().strftime('%Y-%m-%d')
    last_valid_date = valid_dates.max().strftime('%Y-%m-%d')

    print(f"[INFO] Selected {len(valid_dates)} valid dates from {first_valid_date} to {last_valid_date}")
    print(f"[INFO] With at least {min_required_stations}/{total_station_count} stations per date")

    df_filtered = df[df["date"].isin(valid_dates)].copy()

    return df_filtered
