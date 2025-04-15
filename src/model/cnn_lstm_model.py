import os
import json
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import time

from src.model.error_estimation import evaluate_model

def create_sequences(data, input_len=30, pred_len=7):
    X, Y = [], []
    for i in range(len(data) - input_len - pred_len):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+pred_len, 0])  # PM2.5
    return np.array(X), np.array(Y)

def find_latest_after_norm_csv():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    CSV_DIR = os.path.join(BASE_DIR, "data", "processed")
    files = glob(os.path.join(CSV_DIR, "after_normalization_*.csv"))
    if not files:
        raise FileNotFoundError("No after_normalization_*.csv files found.")
    return max(files, key=os.path.getctime)

def train_cnn_lstm_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = find_latest_after_norm_csv()
    model_output_path = os.path.join(BASE_DIR, "outputs", "models", "cnn_lstm_model.keras")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    print(f"[INFO] Using dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    drop_cols = ['station_name', 'latitude', 'longitude']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    required = ['PM2.5', 'temperature', 'humidity', 'wind_speed', 'pressure', 'dayofweek', 'month', 'day']
    df = df[required]

    print(f"[DEBUG] Columns used for training: {df.columns.tolist()}")

    data = df.values
    print(f"[INFO] Loaded dataset with shape: {data.shape}")

    X, Y = create_sequences(data, input_len=30, pred_len=7)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    print(f"[INFO] Prepared {X.shape[0]} samples of input shape {X.shape[1:]}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),  # shape = (30, 8)
        Conv1D(64, 2, activation='relu'),
        MaxPooling1D(2),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(7)
    ])

    model.compile(optimizer=Adam(0.001), loss=Huber(delta=15.0), metrics=['mae'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("[INFO] Starting training...")

    weights = np.where(Y_train[:, -1] > 100, 2.0, 1.0)

    start = time.time()
    history = model.fit(
        X_train, Y_train,
        sample_weight=weights,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    end = time.time()

    print(f"[INFO] Training completed in {end - start:.2f} seconds.")
    model.save(model_output_path)
    print(f"[INFO] Model saved to {model_output_path}")

    scaler_path = os.path.join(BASE_DIR, "outputs", "models", "scaler_params.json")
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)

    evaluate_model(
        model_path=model_output_path,
        scaler_path=scaler_path,
        X_test=X_test,
        Y_test=Y_test,
        feature_names=scaler_params["feature_names"]
    )

    return model, history, X_test, Y_test

if __name__ == "__main__":
    train_cnn_lstm_model()
