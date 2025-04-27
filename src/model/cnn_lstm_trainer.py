import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from src.model.error_estimation import evaluate_model

def create_sequences(data, input_len=30, pred_len=7):
  X = []
  Y = []
  for i in range(len(data) - input_len - pred_len):
    x_seq = data[i:i + input_len]
    y_seq = data[i + input_len:i + input_len + pred_len, 0]  # PM2.5 values
    X.append(x_seq)
    Y.append(y_seq)
  return np.array(X), np.array(Y)


def find_latest_after_norm_csv():
  base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  csv_dir = os.path.join(base_dir, "data", "processed")
  files = []
  for file in glob(os.path.join(csv_dir, "after_normalization_*.csv")):
    files.append(file)

  if len(files) == 0:
    raise FileNotFoundError("No after_normalization_*.csv files found.")

  latest_file = max(files, key=os.path.getctime)
  return latest_file

def plot_training_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.join("outputs", "diagrams")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "train_val_loss.png")
    plt.savefig(plot_path)
    plt.close()

    print("Training history plot saved to:", plot_path)


def train_cnn_lstm_model():
  base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  csv_path = find_latest_after_norm_csv()
  model_output_path = os.path.join(base_dir, "outputs", "models", "cnn_lstm_model.keras")
  os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

  print("Using dataset:", csv_path)
  df = pd.read_csv(csv_path)

  drop_cols = ['station_name', 'latitude', 'longitude']
  cols_to_drop = []
  for col in drop_cols:
    if col in df.columns:
      cols_to_drop.append(col)
  df = df.drop(columns=cols_to_drop, errors='ignore')

  required_columns = ['PM2.5', 'temperature', 'humidity', 'wind_speed', 'pressure', 'dayofweek', 'month', 'day']
  df = df[required_columns]

  print("Columns used for training:", df.columns.tolist())

  data = df.values
  print("Loaded dataset with shape:", data.shape)

  X, Y = create_sequences(data, input_len=30, pred_len=7)
  X = X.astype(np.float32)
  Y = Y.astype(np.float32)
  print("Prepared", X.shape[0], "samples with input shape", X.shape[1:])

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

  model = Sequential()
  model.add(Input(shape=(X.shape[1], X.shape[2])))
  model.add(Conv1D(64, 2, activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(LSTM(64))
  model.add(Dropout(0.3))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(7))

  model.compile(optimizer=Adam(0.001), loss=Huber(delta=15.0), metrics=['mae'])
  model.summary()

  early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

  print("Starting training...")

  weights = np.ones(len(Y_train))
  for i in range(len(Y_train)):
    if Y_train[i, -1] > 100:
      weights[i] = 2.0

  start_time = time.time()

  history = model.fit(
    X_train, Y_train,
    sample_weight=weights,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
  )

  plot_training_history(history)

  end_time = time.time()
  print("Training completed in", round(end_time - start_time, 2), "seconds.")

  model.save(model_output_path)
  print("Model saved to:", model_output_path)

  scaler_path = os.path.join(base_dir, "outputs", "models", "scaler_params.json")
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
