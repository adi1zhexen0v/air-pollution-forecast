import os
import json
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

    # === Сохраняем параметры нормализации только для standard
    if mode == 'standard':
        scaler_params = {
            "feature_names": columns_to_scale,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }

        save_dir = os.path.join("outputs", "models")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "scaler_params.json")
        with open(save_path, "w") as f:
            json.dump(scaler_params, f, indent=4)

        print(f"[SAVED] Scaler parameters saved to {save_path}")

    return df_scaled
