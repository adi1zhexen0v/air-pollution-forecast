import pandas as pd
import numpy as np

def prepare_input_for_prediction(csv_path, input_len=30):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.select_dtypes(include=[np.number])  # убрать дату перед подачей в модель
    data = df.values

    X_input = np.expand_dims(data[-input_len:], axis=0).astype(np.float32)
    last_date = df.index[-1]
    return X_input, df['date'].iloc[-1]

def load_minmax_params(csv_path, column="PM2.5"):
    df = pd.read_csv(csv_path)
    orig = pd.read_csv(csv_path.replace("after", "before"))
    min_val = orig[column].min()
    max_val = orig[column].max()
    return min_val, max_val

def inverse_transform_column(arr, min_val, max_val):
    return arr * (max_val - min_val) + min_val
