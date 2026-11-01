"""
test_verification.py
====================
End-to-end automated verification script for the Weather Prediction Project.
Tests all pipeline components:
  1. Data verification (historical CSV integrity & columns)
  2. Model training & evaluation (Decision Tree & Gaussian Naive Bayes)
  3. Live API geocoding & tomorrow forecast retrieval (Open-Meteo)
  4. Real-time ML inference
  5. Matplotlib chart generation (saved to /plots)

Usage:
  python test_verification.py
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
HISTORICAL_CSV_PATH = DATA_DIR / "weather_historical.csv"

from data_collection import FEATURE_COLUMNS, TARGET_COLUMN, load_or_fetch_historical_data
from model_training import train_and_evaluate_models, display_model_evaluation
from live_forecast import geocode_city, fetch_tomorrow_forecast, predict_tomorrow_weather, display_forecast_and_prediction
from visualization import plot_all_visualizations


def run_all_tests():
    print("=" * 65)
    print(" WEATHER PREDICTION SYSTEM - COMPLETE VERIFICATION SUITE")
    print("=" * 65)

    # -------------------------------------------------------------
    # TEST 1: Historical Data Integrity
    # -------------------------------------------------------------
    print("\n[TEST 1/5] Checking Historical Meteorological Dataset...")
    assert HISTORICAL_CSV_PATH.exists(), f"Historical CSV not found at {HISTORICAL_CSV_PATH}"
    df = pd.read_csv(HISTORICAL_CSV_PATH)
    
    print(f"  - File path        : {HISTORICAL_CSV_PATH.name}")
    print(f"  - Total records    : {len(df)}")
    print(f"  - Missing values   : {df.isnull().sum().sum()}")
    
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Missing required column: {col}"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values!"
    print(f"  - Required columns : {FEATURE_COLUMNS + [TARGET_COLUMN]} -> ALL PRESENT")
    print(f"  - Class Balance    : No Rain (0)={sum(df[TARGET_COLUMN] == 0)}, Rain (1)={sum(df[TARGET_COLUMN] == 1)}")
    print("  -> TEST 1 PASSED: Dataset integrity verified.")

    # -------------------------------------------------------------
    # TEST 2: Model Training & Evaluation
    # -------------------------------------------------------------
    print("\n[TEST 2/5] Training Decision Tree and Gaussian Naive Bayes...")
    results = train_and_evaluate_models(df)
    dt_metrics = results["metrics"]["Decision Tree"]
    nb_metrics = results["metrics"]["Gaussian Naive Bayes"]

    print(f"  - Decision Tree Accuracy      : {dt_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Gaussian Naive Bayes Accuracy: {nb_metrics['accuracy'] * 100:.2f}%")
    print(f"  - DT Rainy Precision / Recall : {dt_metrics['rain_precision'] * 100:.2f}% / {dt_metrics['rain_recall'] * 100:.2f}%")
    print(f"  - NB Rainy Precision / Recall : {nb_metrics['rain_precision'] * 100:.2f}% / {nb_metrics['rain_recall'] * 100:.2f}%")

    assert dt_metrics["accuracy"] >= 0.70, "Decision Tree accuracy below expected threshold (70%)"
    assert nb_metrics["accuracy"] >= 0.70, "Gaussian Naive Bayes accuracy below expected threshold (70%)"
    print("  -> TEST 2 PASSED: Models trained and evaluated successfully.")

    # -------------------------------------------------------------
    # TEST 3: Live Geocoding & Tomorrow Forecast APIs
    # -------------------------------------------------------------
    test_cities = ["Delhi", "London", "Tokyo"]
    print(f"\n[TEST 3/5] Testing Live APIs for cities: {test_cities}...")
    for city in test_cities:
        geo = geocode_city(city)
        print(f"  - Geocoded '{city}': {geo['name']}, {geo['country']} (Lat: {geo['latitude']}, Lon: {geo['longitude']})")
        assert "latitude" in geo and "longitude" in geo

        forecast = fetch_tomorrow_forecast(geo["latitude"], geo["longitude"], timezone=geo["timezone"])
        for feat in FEATURE_COLUMNS:
            assert feat in forecast, f"Missing feature {feat} in forecast"
        print(f"    Forecast for {forecast['date']}: Temp={forecast['temperature_c']:.1f}°C, Humidity={forecast['humidity_percent']:.1f}%, Cloud={forecast['cloud_cover_percent']:.1f}%")
    print("  -> TEST 3 PASSED: Open-Meteo Geocoding and Forecast APIs verified.")

    # -------------------------------------------------------------
    # TEST 4: Real-time ML Inference on Live Data
    # -------------------------------------------------------------
    print("\n[TEST 4/5] Running ML Inference on Live City Forecast...")
    geo_delhi = geocode_city("Delhi")
    delhi_forecast = fetch_tomorrow_forecast(geo_delhi["latitude"], geo_delhi["longitude"], timezone=geo_delhi["timezone"])
    prediction = predict_tomorrow_weather(results["models"], delhi_forecast)

    print(f"  - Tomorrow's Forecast (Delhi): {delhi_forecast['date']}")
    print(f"    Decision Tree prediction : {prediction['dt_label']}")
    print(f"    Naive Bayes prediction   : {prediction['nb_label']}")
    assert prediction["dt_label"] in ["Rain likely", "No rain likely"]
    assert prediction["nb_label"] in ["Rain likely", "No rain likely"]
    print("  -> TEST 4 PASSED: Live inference executed successfully.")

    # -------------------------------------------------------------
    # TEST 5: Matplotlib Visualization Generation
    # -------------------------------------------------------------
    print("\n[TEST 5/5] Generating Matplotlib Plots to /plots directory...")
    plot_all_visualizations(df, results, show=False, save_dir=str(PLOTS_DIR))

    expected_plots = [
        "model_comparison.png",
        "confusion_matrices.png",
        "historical_trends.png"
    ]
    for p in expected_plots:
        plot_path = PLOTS_DIR / p
        assert plot_path.exists(), f"Plot not found: {plot_path}"
        assert plot_path.stat().st_size > 0, f"Plot is empty: {plot_path}"
        print(f"  - Verified chart: {p} ({plot_path.stat().st_size // 1024} KB)")

    print("  -> TEST 5 PASSED: All visualizations generated and verified.")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("  Project is fully operational and lab compliant.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()

