import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_features(df, columns_to_scale, mode='standard') -> pd.DataFrame:
    df_scaled = df.copy()

    if mode == 'minmax':
        scaler = MinMaxScaler()
    elif mode == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid mode. Use 'standard' or 'minmax'.")

    scaled_values = scaler.fit_transform(df_scaled[columns_to_scale])

    for i, col in enumerate(columns_to_scale):
        df_scaled[col] = scaled_values[:, i]

    return df_scaled
