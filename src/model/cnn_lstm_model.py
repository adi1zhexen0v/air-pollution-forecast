import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time

def create_sequences(data, input_len=30, pred_len=7):
    X, Y = [], []
    for i in range(len(data) - input_len - pred_len):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+pred_len, 0])  # PM2.5
    return np.array(X), np.array(Y)

def train_cnn_lstm_model():
    csv_path = "data/processed/after_normalization_2025-04-14_22-46-21.csv"
    model_output_path = "outputs/models/cnn_lstm_model.keras"

    df = pd.read_csv(csv_path)

    # Удалим нечисловые колонки, включая 'date' и 'location'
    df = df.select_dtypes(include=[np.number])

    print(f"[DEBUG] Columns after dropping non-numeric: {df.columns.tolist()}")

    data = df.values

    print(f"[INFO] Loaded dataset with shape: {data.shape}")

    X, Y = create_sequences(data, input_len=30, pred_len=7)

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    print(f"[INFO] Prepared {X.shape[0]} samples of input shape {X.shape[1:]}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        Conv1D(64, 2, activation='relu'),
        MaxPooling1D(2),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(7)
    ])

    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    model.summary()  # ✅ модельная структура

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("[INFO] Starting training...")
    start = time.time()

    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1  # ✅ вывод каждой эпохи, как в лекции
    )

    end = time.time()
    print(f"[INFO] Training completed in {end - start:.2f} seconds.")

    model.save(model_output_path)
    print(f"[INFO] Model saved to {model_output_path}")

    return model, history, X_test, Y_test
