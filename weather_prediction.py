"""
weather_prediction.py
=====================
AI/ML-Based Weather Forecasting with Historical Data and Live Predictions
Data Mining and Warehousing Lab (CSAD2500) Compatible Implementation

Strict Academic Constraints:
- Libraries: Pandas, NumPy, Matplotlib, Scikit-learn ONLY
- Algorithms: Decision Tree Classifier & Gaussian Naive Bayes ONLY
- Metrics: train_test_split, accuracy_score, confusion_matrix, classification_report
- Real Data: Open-Meteo Historical Archive (Delhi 2023-2025) & Live Forecast APIs
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Modular project imports
from data_collection import (
    load_or_fetch_historical_data,
    fetch_historical_weather,
    save_historical_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from model_training import (
    train_and_evaluate_models,
    display_model_evaluation
)
from live_forecast import (
    geocode_city,
    fetch_tomorrow_forecast,
    predict_tomorrow_weather,
    display_forecast_and_prediction
)
from visualization import (
    plot_model_comparison,
    plot_confusion_matrices,
    plot_historical_trends,
    plot_all_visualizations
)

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_CSV_PATH = str(DATA_DIR / "weather_historical.csv")
SAMPLE_CSV_PATH = str(DATA_DIR / "weather_sample.csv")
PLOTS_DIR = str(BASE_DIR / "plots")


def print_banner():
    """Prints the application banner."""
    print("\n" + "=" * 65)
    print("   WEATHER PREDICTION USING DATA MINING TECHNIQUES (AI/ML)")
    print("   Verified Historical Data & Real-Time Open-Meteo Forecasting")
    print("=" * 65)


def handle_train_models(df: pd.DataFrame):
    """Trains and returns models and evaluation metrics."""
    print("\n[Training] Training Decision Tree and Gaussian Naive Bayes models...")
    results = train_and_evaluate_models(df)
    dt_acc = results["metrics"]["Decision Tree"]["accuracy"]
    nb_acc = results["metrics"]["Gaussian Naive Bayes"]["accuracy"]
    print(f"[Done] Training complete on {len(df)} historical observations.")
    print(f"       Decision Tree Accuracy : {dt_acc * 100:.2f}%")
    print(f"       Naive Bayes Accuracy   : {nb_acc * 100:.2f}%")
    return results


def handle_live_prediction(results: dict):
    """Prompts user for a city name, fetches forecast, and predicts tomorrow's weather."""
    if results is None:
        print("[Error] Please train the models first (Option 1).")
        return

    city_name = input("\nEnter city name (e.g., Delhi, Mumbai, London, New York): ").strip()
    if not city_name:
        print("[Error] City name cannot be empty.")
        return

    print(f"\n[1/3] Geocoding '{city_name}' via Open-Meteo Geocoding API...")
    try:
        geo = geocode_city(city_name)
    except Exception as e:
        print(f"[Geocoding Error] {e}")
        return

    print(f"[2/3] Found: {geo['name']}, {geo['country']} (Lat: {geo['latitude']}, Lon: {geo['longitude']})")
    print(f"      Fetching tomorrow's live weather forecast from Open-Meteo Forecast API...")
    try:
        forecast = fetch_tomorrow_forecast(geo["latitude"], geo["longitude"], timezone=geo["timezone"])
    except Exception as e:
        print(f"[Forecast API Error] {e}")
        return

    print(f"[3/3] Running Decision Tree and Gaussian Naive Bayes ML models...")
    prediction = predict_tomorrow_weather(results["models"], forecast)

    display_forecast_and_prediction(geo, forecast, prediction)


def handle_manual_prediction(results: dict):
    """Allows manual entry of weather conditions (preserves original project behavior)."""
    if results is None:
        print("[Error] Please train the models first (Option 1).")
        return

    print("\n========== ENTER MANUAL WEATHER CONDITIONS ==========")
    try:
        temp = float(input("Temperature (°C)   : "))
        humidity = float(input("Humidity (%)       : "))
        pressure = float(input("Pressure (hPa)     : "))
        wind = float(input("Wind speed (km/h)  : "))
        cloud = float(input("Cloud cover (%)    : "))
    except ValueError:
        print("[Error] Invalid input. Please enter numerical values.")
        return

    manual_features = {
        "temperature_c": temp,
        "humidity_percent": humidity,
        "pressure_hpa": pressure,
        "wind_speed_kmh": wind,
        "cloud_cover_percent": cloud
    }
    prediction = predict_tomorrow_weather(results["models"], manual_features)

    print("\n" + "-" * 50)
    print("MANUAL WEATHER INPUT ML PREDICTION:")
    print(f"  - Decision Tree Prediction : {prediction['dt_label']}")
    print(f"  - Naive Bayes Prediction   : {prediction['nb_label']}")
    print("-" * 50)


def handle_visualizations(df: pd.DataFrame, results: dict):
    """Offers submenu for interactive Matplotlib visualizations."""
    if results is None:
        print("[Error] Please train the models first (Option 1).")
        return

    while True:
        print("\n--- VISUALIZATION MENU (Matplotlib) ---")
        print("1. Decision Tree vs Naive Bayes Accuracy & Recall Comparison")
        print("2. Confusion Matrices for Both Models")
        print("3. Historical Weather Trends (Monthly Temp & Rain)")
        print("4. Generate and Display All Visualizations")
        print("5. Return to Main Menu")

        vchoice = input("Enter visualization choice (1-5): ").strip()
        if vchoice == "1":
            plot_model_comparison(results, show=True)
        elif vchoice == "2":
            plot_confusion_matrices(results, show=True)
        elif vchoice == "3":
            plot_historical_trends(df, show=True)
        elif vchoice == "4":
            plot_all_visualizations(df, results, show=True, save_dir=PLOTS_DIR)
        elif vchoice == "5":
            break
        else:
            print("[Error] Invalid choice. Please enter a number between 1 and 5.")


def main():
    """Main CLI driver for Weather Prediction application."""
    print_banner()

    # Step 1: Ensure historical data is present
    df = None
    results = None

    try:
        df = load_or_fetch_historical_data(HISTORICAL_CSV_PATH, force_refresh=False)
        results = handle_train_models(df)
    except Exception as e:
        print(f"\n[Startup Warning] Could not load historical data: {e}")
        if os.path.exists(SAMPLE_CSV_PATH):
            print(f"[Fallback] Falling back to sample CSV: {SAMPLE_CSV_PATH}")
            df = pd.read_csv(SAMPLE_CSV_PATH).drop_duplicates().dropna()
            if "rain_tomorrow" in df.columns and "rain_today" not in df.columns:
                df["rain_today"] = df["rain_tomorrow"]
            results = handle_train_models(df)
        else:
            print("[Error] No dataset available. Please refresh dataset using Option 6.")

    # Interactive Loop
    while True:
        print("\n" + "=" * 50)
        print("                MAIN MENU")
        print("=" * 50)
        print("1. Train models using historical data")
        print("2. View model evaluation and performance comparison")
        print("3. Predict tomorrow's weather for a city (Live API)")
        print("4. Enter manual weather conditions for prediction")
        print("5. View visualizations (Matplotlib)")
        print("6. Refresh / re-download historical dataset")
        print("7. Exit")
        print("=" * 50)

        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            if df is not None:
                results = handle_train_models(df)
            else:
                print("[Error] No dataset loaded. Please refresh historical data (Option 6).")

        elif choice == "2":
            if results is not None:
                display_model_evaluation(results)
            else:
                print("[Error] Models not trained yet. Select Option 1 first.")

        elif choice == "3":
            handle_live_prediction(results)

        elif choice == "4":
            handle_manual_prediction(results)

        elif choice == "5":
            if df is not None and results is not None:
                handle_visualizations(df, results)
            else:
                print("[Error] Data or trained models not available. Select Option 1 first.")

        elif choice == "6":
            confirm = input("Re-download historical records from Open-Meteo Archive API? (y/n): ").strip().lower()
            if confirm in ["y", "yes"]:
                try:
                    df = fetch_historical_weather()
                    save_historical_data(df, HISTORICAL_CSV_PATH)
                    results = handle_train_models(df)
                except Exception as e:
                    print(f"[Error] Failed to refresh historical data: {e}")

        elif choice == "7":
            print("\nExiting Weather Prediction application. Goodbye!")
            sys.exit(0)

        else:
            print("[Error] Invalid choice. Please enter a number from 1 to 7.")


if __name__ == "__main__":
    main()
