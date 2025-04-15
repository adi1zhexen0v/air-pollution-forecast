import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from datetime import datetime

def evaluate_model(model_path, scaler_path, X_test, Y_test, feature_names):
    model = load_model(model_path)
    Y_pred = model.predict(X_test)

    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)

    pm_index = feature_names.index("PM2.5")
    pm_mean = scaler_params["mean"][pm_index]
    pm_std = scaler_params["scale"][pm_index]

    # Обратное масштабирование
    Y_pred_real = Y_pred * pm_std + pm_mean
    Y_test_real = Y_test * pm_std + pm_mean

    # Метрики
    rmse = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real))
    mae = mean_absolute_error(Y_test_real, Y_pred_real)
    r2 = r2_score(Y_test_real, Y_pred_real)

    print("\n📊 Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")

    # === Сохранение графиков
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Линейный график
    line_path = os.path.join(output_dir, f"forecast_comparison_{timestamp}.png")
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test_real.flatten(), label='True PM2.5', color='blue')
    plt.plot(Y_pred_real.flatten(), label='Predicted PM2.5', color='red')
    plt.title("PM2.5 Forecast: True vs Predicted")
    plt.xlabel("Sample index")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(line_path)
    plt.close()
    print(f"[SAVED] Line plot saved to: {line_path}")

    # 2. Scatter plot: Y_true vs Y_pred
    scatter_path = os.path.join(output_dir, f"scatter_true_vs_pred_{timestamp}.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test_real.flatten(), Y_pred_real.flatten(), alpha=0.6, color="green")
    plt.plot([Y_test_real.min(), Y_test_real.max()],
             [Y_test_real.min(), Y_test_real.max()],
             color='red', linestyle='--', label="Ideal prediction")
    plt.xlabel("True PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Scatter Plot: True vs Predicted PM2.5")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    print(f"[SAVED] Scatter plot saved to: {scatter_path}")

    return rmse, mae, r2
