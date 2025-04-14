import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.extract.sensor_loader import collect_all_stations_data
from src.preprocess.time_features import extract_time_features
from src.weather.open_meteo_fetcher import add_weather_columns
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import normalize_features


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# df = collect_all_stations_data()
# df = add_weather_columns(df)
# df = extract_time_features(df)
#
# df = df[df["PM2.5"] <= 300]
# df = df.dropna(subset=["PM2.5", "temperature", "humidity", "wind_speed", "pressure"], how="all")
#
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# before_filename = f"data/processed/before_normalization_{timestamp}.csv"
# df.to_csv(before_filename, index=False)

# df = pd.read_csv('./data/processed/before_normalization_2025-04-14_22-33-59.csv')
# df = select_features(df, threshold=0.9)
# columns_to_scale = ["PM2.5", "temperature", "humidity", "wind_speed", "pressure"]
# df = normalize_features(df, columns_to_scale, mode='minmax')
#
# after_filename = f"data/processed/after_normalization_{timestamp}.csv"
# df.to_csv(after_filename, index=False)

# print(f"✅ Saved BEFORE normalization: {before_filename}")
# print(f"✅ Saved AFTER normalization: {after_filename}")

# df = pd.read_csv('./data/processed/after_normalization_2025-04-14_19-08-29.csv')
# df["date"] = pd.to_datetime(df["date"])
# unique_dates = df["date"].dt.date.unique()
#
# for date in unique_dates:
#     df_day = df[df["date"].dt.date == date]
#     heatmap = generate_heatmap(df_day, resolution=128, normalize=False)
#
#     if heatmap is not None:
#         filename = f"heatmap_{date}.npy"
#         filepath = f"outputs/heatmaps/{filename}"
#         np.save(filepath, heatmap)
#
#         append_heatmap_metadata(date, filename, df_day)
#         print(f"✅ Heatmap saved for {date}")


# def plot_training_history(history, save_path=None):
#     epoch_range = range(1, len(history.history['loss']) + 1)
#
#     plt.figure(figsize=(12, 5), dpi=120)
#
#     plt.subplot(1, 2, 1)
#     plt.plot(epoch_range, history.history['loss'], 'b', label='Training loss')
#     plt.plot(epoch_range, history.history['val_loss'], 'r', label='Validation loss')
#     plt.title('Loss over epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(epoch_range, history.history['mae'], 'b', label='Training MAE')
#     plt.plot(epoch_range, history.history['val_mae'], 'r', label='Validation MAE')
#     plt.title('MAE over epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Mean Absolute Error')
#     plt.legend()
#
#     plt.tight_layout()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.show()
#
# def plot_prediction(Y_true, Y_pred, example_id=0):
#     plt.figure(figsize=(10, 5))
#     plt.plot(Y_true[example_id], label='True')
#     plt.plot(Y_pred[example_id], label='Predicted')
#     plt.title(f"PM2.5 Forecast (7 days) — Example {example_id}")
#     plt.xlabel("Days Ahead")
#     plt.ylabel("Normalized PM2.5")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     model, history, X_test, Y_test = train_cnn_lstm_model()
#
#     pred = model.predict(X_test)
#
#     # 🔍 Визуализация прогноза
#     plot_prediction(Y_test, pred, example_id=0)
#
#     # 📈 Визуализация потерь и ошибок
#     plot_training_history(history, save_path="outputs/plots/training_history.pdf")