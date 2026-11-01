"""
model_training.py
-----------------
Module for training and evaluating Decision Tree and Gaussian Naive Bayes models
on historical weather observations.

Part of: Weather Prediction Using Data Mining Techniques
Lab: Data Mining and Warehousing Lab (CSAD2500)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "pressure_hpa",
    "wind_speed_kmh",
    "cloud_cover_percent"
]
TARGET_COLUMN = "rain_today"


def prepare_features_and_target(df: pd.DataFrame):
    """
    Validates DataFrame and extracts feature matrix X and target vector y.
    
    Checks:
    - Required feature columns exist
    - Target column exists
    - Insufficient record count (< 20)
    - Single class anomaly in target
    """
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing required feature column: '{col}'")

    if TARGET_COLUMN not in df.columns:
        if "rain_tomorrow" in df.columns:
            df[TARGET_COLUMN] = df["rain_tomorrow"]
        else:
            raise KeyError(f"Missing target column: '{TARGET_COLUMN}'")

    cleaned_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if len(cleaned_df) < 20:
        raise ValueError(
            f"Insufficient training records: only {len(cleaned_df)} records found. "
            f"At least 20 records are required to train and evaluate ML models."
        )

    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN].astype(int)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset target '{TARGET_COLUMN}' contains only one class ({unique_classes}). "
            f"Both class 0 (No Rain) and class 1 (Rain) are required for classification."
        )

    return X, y


def train_and_evaluate_models(df: pd.DataFrame) -> dict:
    """
    Trains Decision Tree Classifier and Gaussian Naive Bayes using train_test_split.
    Calculates genuine evaluation metrics from the actual historical dataset.
    
    Returns:
        dict: Trained models, test data, predictions, and evaluation metrics.
    """
    X, y = prepare_features_and_target(df)

    # 70% Training, 30% Testing with stratification and fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Model 1: Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    # Model 2: Gaussian Naive Bayes Classifier
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    # Calculate metrics
    dt_acc = accuracy_score(y_test, dt_preds)
    nb_acc = accuracy_score(y_test, nb_preds)

    dt_cm = confusion_matrix(y_test, dt_preds)
    nb_cm = confusion_matrix(y_test, nb_preds)

    dt_report = classification_report(y_test, dt_preds, zero_division=0)
    nb_report = classification_report(y_test, nb_preds, zero_division=0)

    # Rainy class (class 1) recall: TP / (TP + FN)
    # confusion_matrix format: [[TN, FP], [FN, TP]]
    dt_rain_recall = dt_cm[1, 1] / (dt_cm[1, 1] + dt_cm[1, 0]) if (dt_cm[1, 1] + dt_cm[1, 0]) > 0 else 0.0
    nb_rain_recall = nb_cm[1, 1] / (nb_cm[1, 1] + nb_cm[1, 0]) if (nb_cm[1, 1] + nb_cm[1, 0]) > 0 else 0.0

    dt_rain_precision = dt_cm[1, 1] / (dt_cm[1, 1] + dt_cm[0, 1]) if (dt_cm[1, 1] + dt_cm[0, 1]) > 0 else 0.0
    nb_rain_precision = nb_cm[1, 1] / (nb_cm[1, 1] + nb_cm[0, 1]) if (nb_cm[1, 1] + nb_cm[0, 1]) > 0 else 0.0

    return {
        "features": FEATURE_COLUMNS,
        "models": {
            "Decision Tree": dt_model,
            "Gaussian Naive Bayes": nb_model
        },
        "test_data": {
            "X_test": X_test,
            "y_test": y_test
        },
        "predictions": {
            "Decision Tree": dt_preds,
            "Gaussian Naive Bayes": nb_preds
        },
        "metrics": {
            "Decision Tree": {
                "accuracy": dt_acc,
                "confusion_matrix": dt_cm,
                "classification_report": dt_report,
                "rain_recall": dt_rain_recall,
                "rain_precision": dt_rain_precision
            },
            "Gaussian Naive Bayes": {
                "accuracy": nb_acc,
                "confusion_matrix": nb_cm,
                "classification_report": nb_report,
                "rain_recall": nb_rain_recall,
                "rain_precision": nb_rain_precision
            }
        }
    }


def display_model_evaluation(results: dict) -> None:
    """Prints clear, human-readable evaluation reports and comparisons."""
    metrics = results["metrics"]

    print("\n" + "=" * 60)
    print("      MACHINE LEARNING MODEL EVALUATION RESULTS")
    print("=" * 60)

    # Decision Tree
    dt_m = metrics["Decision Tree"]
    print("\n>>> MODEL 1: DECISION TREE CLASSIFIER")
    print(f"Overall Accuracy : {dt_m['accuracy'] * 100:.2f}%")
    print(f"Rainy-Class (1) Precision : {dt_m['rain_precision'] * 100:.2f}%")
    print(f"Rainy-Class (1) Recall    : {dt_m['rain_recall'] * 100:.2f}%")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(dt_m["confusion_matrix"])
    print("\nClassification Report:")
    print(dt_m["classification_report"])

    # Gaussian Naive Bayes
    nb_m = metrics["Gaussian Naive Bayes"]
    print("-" * 60)
    print(">>> MODEL 2: GAUSSIAN NAIVE BAYES CLASSIFIER")
    print(f"Overall Accuracy : {nb_m['accuracy'] * 100:.2f}%")
    print(f"Rainy-Class (1) Precision : {nb_m['rain_precision'] * 100:.2f}%")
    print(f"Rainy-Class (1) Recall    : {nb_m['rain_recall'] * 100:.2f}%")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(nb_m["confusion_matrix"])
    print("\nClassification Report:")
    print(nb_m["classification_report"])

    # Comparison Table
    print("\n" + "=" * 78)
    print("MODEL PERFORMANCE COMPARISON".center(78))
    print("=" * 78)
    print(f"{'Metric':<34} | {'Decision Tree':^16} | {'Gaussian Naive Bayes':^22}")
    print("-" * 78)

    dt_acc_str = f"{dt_m['accuracy'] * 100:.2f}%"
    nb_acc_str = f"{nb_m['accuracy'] * 100:.2f}%"
    dt_rec_str = f"{dt_m['rain_recall'] * 100:.2f}%"
    nb_rec_str = f"{nb_m['rain_recall'] * 100:.2f}%"
    dt_prec_str = f"{dt_m['rain_precision'] * 100:.2f}%"
    nb_prec_str = f"{nb_m['rain_precision'] * 100:.2f}%"

    print(f"{'Overall Accuracy':<34} | {dt_acc_str:^16} | {nb_acc_str:^22}")
    print(f"{'Rainy-Class Recall (Sensitivity)':<34} | {dt_rec_str:^16} | {nb_rec_str:^22}")
    print(f"{'Rainy-Class Precision':<34} | {dt_prec_str:^16} | {nb_prec_str:^22}")
    print("=" * 78)


if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "weather_historical.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        res = train_and_evaluate_models(df)
        display_model_evaluation(res)
    else:
        print("Historical CSV not found. Run data_collection.py first.")
