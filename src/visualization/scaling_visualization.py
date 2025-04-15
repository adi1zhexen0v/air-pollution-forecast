import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_pm25_standard_scaling_histogram(df_raw, df_standard, save_dir="outputs/diagrams"):
    os.makedirs(save_dir, exist_ok=True)

    column = "PM2.5"

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df_raw[column].dropna(), kde=True, bins=30)
    plt.title(f'Original: {column}')

    plt.subplot(1, 2, 2)
    sns.histplot(df_standard[column].dropna(), kde=True, bins=30, color='green')
    plt.title(f'StandardScaled: {column}')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{column}_standard_scaling.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[SAVED] Histogram saved to: {save_path}")
