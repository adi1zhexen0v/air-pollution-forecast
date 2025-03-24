import pandas as pd

def select_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    non_feature_columns  = ['date', 'station_name', 'latitude', 'longitude']

    feature_columns = []
    for column in df.columns:
        if column not in non_feature_columns :
            feature_columns.append(column)

    completeness = {}
    for column in feature_columns:
        not_null_count = df[column].notnull().sum()
        completeness[column] = not_null_count / len(df[column])

    print("\nFeature Completeness:")
    for key in sorted(completeness, key=completeness.get, reverse=True):
        print(f"{key}: {round(completeness[key] * 100, 2)}%")

    selected_features = []
    for key, value in completeness.items():
        if value >= threshold:
            selected_features.append(key)

    print(f"\nSelected features (more than {int(threshold * 100)}% filled): {selected_features}")
    final_columns = non_feature_columns  + selected_features
    filtered_df = df[final_columns].copy()
    filtered_df = filtered_df.dropna(subset=selected_features, how='all')

    return filtered_df
