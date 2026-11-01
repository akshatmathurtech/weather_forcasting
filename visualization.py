"""
visualization.py
----------------
Simple, clean Matplotlib visualizations for the Weather Prediction project.
Adheres strictly to Data Mining Lab (CSAD2500) requirements.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish_plot(show: bool = True, save_path: str = None):
    """Helper to save and optionally display a plot."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Plot Saved] {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_model_comparison(results: dict, show: bool = True, save_path: str = None):
    """
    Plots a simple bar chart comparing Decision Tree and Naive Bayes
    on Accuracy and Rain Recall.
    """
    dt_m = results["metrics"]["Decision Tree"]
    nb_m = results["metrics"]["Gaussian Naive Bayes"]

    models = ["Decision Tree", "Gaussian Naive Bayes"]
    accuracies = [dt_m["accuracy"] * 100, nb_m["accuracy"] * 100]
    recalls = [dt_m["rain_recall"] * 100, nb_m["rain_recall"] * 100]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#3498db")
    plt.bar(x + width / 2, recalls, width, label="Rain Recall (%)", color="#e67e22")

    plt.title("Model Performance Comparison", fontsize=12, fontweight="bold")
    plt.xticks(x, models, fontsize=10)
    plt.ylabel("Score (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Add score text on top of bars
    for i in range(len(models)):
        plt.text(x[i] - width / 2, accuracies[i] + 1.5, f"{accuracies[i]:.1f}%", ha="center", fontsize=9)
        plt.text(x[i] + width / 2, recalls[i] + 1.5, f"{recalls[i]:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    _finish_plot(show, save_path)


def plot_confusion_matrices(results: dict, show: bool = True, save_path: str = None):
    """
    Plots side-by-side Confusion Matrices for Decision Tree and Naive Bayes.
    """
    dt_cm = results["metrics"]["Decision Tree"]["confusion_matrix"]
    nb_cm = results["metrics"]["Gaussian Naive Bayes"]["confusion_matrix"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    cms = [(ax1, "Decision Tree", dt_cm), (ax2, "Gaussian Naive Bayes", nb_cm)]

    for ax, name, cm in cms:
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} Matrix", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Rain", "Rain"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Rain", "Rain"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Put numbers in matrix cells
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    _finish_plot(show, save_path)


def plot_historical_trends(df: pd.DataFrame, show: bool = True, save_path: str = None):
    """
    Plots monthly average temperature and total rainfall over time.
    """
    df_copy = df.copy()
    if "date" not in df_copy.columns:
        return

    df_copy["date"] = pd.to_datetime(df_copy["date"])
    monthly = df_copy.groupby(df_copy["date"].dt.to_period("M")).mean(numeric_only=True)
    months = monthly.index.astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Temperature Trend
    ax1.plot(months, monthly["temperature_c"], color="#e74c3c", marker="o", linewidth=2)
    ax1.set_ylabel("Avg Temp (°C)")
    ax1.set_title("Historical Weather Trends in Delhi (Monthly)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Rainfall Trend
    if "precipitation_mm" in df_copy.columns:
        monthly_rain = df_copy.groupby(df_copy["date"].dt.to_period("M"))["precipitation_mm"].sum()
        ax2.bar(months, monthly_rain, color="#3498db", alpha=0.7)
        ax2.set_ylabel("Total Rain (mm)")
    ax2.set_xlabel("Month")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    _finish_plot(show, save_path)


def plot_all_visualizations(df: pd.DataFrame, results: dict, show: bool = True, save_dir: str = None):
    """
    Generates and saves all visualizations.
    """
    print("\n[Visualizations] Generating Matplotlib charts...")
    p1 = os.path.join(save_dir, "model_comparison.png") if save_dir else None
    p2 = os.path.join(save_dir, "confusion_matrices.png") if save_dir else None
    p3 = os.path.join(save_dir, "historical_trends.png") if save_dir else None

    plot_model_comparison(results, show=show, save_path=p1)
    plot_confusion_matrices(results, show=show, save_path=p2)
    plot_historical_trends(df, show=show, save_path=p3)
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
