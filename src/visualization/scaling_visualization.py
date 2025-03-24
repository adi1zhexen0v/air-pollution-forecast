import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_scaling_histograms(df_raw, df_standard, df_minmax, columns_to_plot, save_dir="outputs/scaling"):
    os.makedirs(save_dir, exist_ok=True)

    for column in columns_to_plot:
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(df_raw[column].dropna(), kde=True, bins=30)
        plt.title(f'Original: {column}')

        plt.subplot(1, 3, 2)
        sns.histplot(df_standard[column].dropna(), kde=True, bins=30, color='green')
        plt.title(f'StandardScaled: {column}')

        plt.subplot(1, 3, 3)
        sns.histplot(df_minmax[column].dropna(), kde=True, bins=30, color='orange')
        plt.title(f'MinMaxScaled: {column}')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{column}_compare_scaling.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")