"""
visualization.py
----------------
Module for visualizing historical weather data, model comparisons, and
confusion matrices using Matplotlib ONLY (strictly complying with lab restrictions).

Part of: Weather Prediction Using Data Mining Techniques
Lab: Data Mining and Warehousing Lab (CSAD2500)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rain_distribution(df: pd.DataFrame, show: bool = True, save_path: str = None):
    """
    Plots the distribution of Rain vs No Rain in the historical dataset.
    """
    target_col = "rain_today" if "rain_today" in df.columns else "rain_tomorrow"
    counts = df[target_col].value_counts().sort_index()

    labels = ["No Rain (0)", "Rain (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    percentages = [v / sum(values) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#3498db", "#e74c3c"], width=0.5, edgecolor="black")

    for bar, val, pct in zip(bars, values, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{val} days\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Historical Rainfall Class Distribution (Delhi 2023–2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Days", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[Plot Saved] {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_model_comparison(results: dict, show: bool = True, save_path: str = None):
    """
    Plots a grouped bar chart comparing Decision Tree and Gaussian Naive Bayes
    on Accuracy and Rainy-Class Recall.
    """
    dt_m = results["metrics"]["Decision Tree"]
    nb_m = results["metrics"]["Gaussian Naive Bayes"]

    models = ["Decision Tree", "Gaussian Naive Bayes"]
    accuracy_scores = [dt_m["accuracy"] * 100, nb_m["accuracy"] * 100]
    recall_scores = [dt_m["rain_recall"] * 100, nb_m["rain_recall"] * 100]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, accuracy_scores, width, label="Overall Accuracy (%)", color="#2ecc71", edgecolor="black")
    rects2 = ax.bar(x + width / 2, recall_scores, width, label="Rainy-Class Recall (%)", color="#e67e22", edgecolor="black")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax.set_title("ML Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[Plot Saved] {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrices(results: dict, show: bool = True, save_path: str = None):
    """
    Plots side-by-side Confusion Matrices for Decision Tree and Gaussian Naive Bayes
    using Matplotlib imshow and text annotations.
    """
    dt_cm = results["metrics"]["Decision Tree"]["confusion_matrix"]
    nb_cm = results["metrics"]["Gaussian Naive Bayes"]["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cms = [("Decision Tree", dt_cm), ("Gaussian Naive Bayes", nb_cm)]
    classes = ["No Rain (0)", "Rain (1)"]

    for ax, (title, cm) in zip(axes, cms):
        cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{title} Confusion Matrix", fontsize=12, fontweight="bold")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

        # Annotate cells with TN, FP, FN, TP labels and counts
        tag_map = [["True Neg (TN)", "False Pos (FP)"], ["False Neg (FN)", "True Pos (TP)"]]
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(
                    j, i, f"{cm[i, j]}\n({tag_map[i][j]})",
                    ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold"
                )

        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Actual Label", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[Plot Saved] {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_historical_trends(df: pd.DataFrame, show: bool = True, save_path: str = None):
    """
    Plots monthly historical weather trends (Average Temperature and Monthly Rainfall)
    over the 3-year record period.
    """
    df_copy = df.copy()
    if "date" in df_copy.columns:
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        df_copy.set_index("date", inplace=True)
    else:
        return

    # Monthly aggregation (compatible with pandas < 2.2 'M' and >= 2.2 'ME')
    freq = "ME"
    try:
        monthly = df_copy.resample("ME").agg({
            "temperature_c": "mean",
            "precipitation_mm": "sum" if "precipitation_mm" in df_copy.columns else "count"
        })
    except (ValueError, KeyError):
        monthly = df_copy.resample("M").agg({
            "temperature_c": "mean",
            "precipitation_mm": "sum" if "precipitation_mm" in df_copy.columns else "count"
        })

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_temp = "#e74c3c"
    ax1.set_xlabel("Timeline (Month)", fontsize=11)
    ax1.set_ylabel("Mean Temperature (°C)", color=color_temp, fontsize=11)
    line1 = ax1.plot(monthly.index, monthly["temperature_c"], color=color_temp, marker="o", linewidth=2, label="Mean Temp (°C)")
    ax1.tick_params(axis="y", labelcolor=color_temp)

    ax2 = ax1.twinx()
    color_rain = "#2980b9"
    ax2.set_ylabel("Total Precipitation (mm)", color=color_rain, fontsize=11)
    line2 = ax2.bar(monthly.index, monthly["precipitation_mm"], width=15, alpha=0.4, color=color_rain, label="Total Rain (mm)")
    ax2.tick_params(axis="y", labelcolor=color_rain)

    plt.title("Historical Weather Trends in Delhi (2023–2025)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[Plot Saved] {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_all_visualizations(df: pd.DataFrame, results: dict, show: bool = True, save_dir: str = None):
    """
    Generates and displays all key plots sequentially.
    """
    print("\n[Visualizations] Generating Matplotlib charts...")
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    p1 = os.path.join(save_dir, "rain_distribution.png") if save_dir else None
    p2 = os.path.join(save_dir, "model_comparison.png") if save_dir else None
    p3 = os.path.join(save_dir, "confusion_matrices.png") if save_dir else None
    p4 = os.path.join(save_dir, "historical_trends.png") if save_dir else None

    plot_rain_distribution(df, show=show, save_path=p1)
    plot_model_comparison(results, show=show, save_path=p2)
    plot_confusion_matrices(results, show=show, save_path=p3)
    plot_historical_trends(df, show=show, save_path=p4)
    print("[Visualizations] All charts displayed successfully.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "weather_historical.csv")
    if os.path.exists(csv_path):
        from model_training import train_and_evaluate_models
        df = pd.read_csv(csv_path)
        res = train_and_evaluate_models(df)
        plot_all_visualizations(df, res, show=False, save_dir=os.path.join(current_dir, "plots"))
    else:
        print("Historical CSV not found. Run data_collection.py first.")
