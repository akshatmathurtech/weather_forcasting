# AI/ML-Based Weather Forecasting with Historical Data and Live Predictions

A practical Artificial Intelligence and Machine Learning weather forecasting application developed for the **Data Mining and Warehousing Lab (CSAD2500)**.

The project downloads verified historical meteorological observations from the **Open-Meteo Historical Archive API**, trains **Decision Tree** and **Gaussian Naive Bayes** classifiers on real atmospheric parameters, geocodes user-queried cities globally, fetches live tomorrow forecast data via the **Open-Meteo Forecast API**, and predicts precipitation without training/inference feature mismatches.

**GitHub Repository:** [https://github.com/akshatmathurtech/weather_forcasting](https://github.com/akshatmathurtech/weather_forcasting)

---

## Table of Contents
- [1. Project Introduction & Problem Statement](#1-project-introduction--problem-statement)
- [2. Project Objectives](#2-project-objectives)
- [3. Strict Academic & Lab Restrictions](#3-strict-academic--lab-restrictions)
- [4. System Architecture & Directory Structure](#4-system-architecture--directory-structure)
- [5. Data Sources & Variables](#5-data-sources--variables)
- [6. Model Evaluation Results](#6-model-evaluation-results)
- [7. Installation & Setup](#7-installation--setup)
- [8. Running the Application](#8-running-the-application)
- [9. Automated Verification Suite](#9-automated-verification-suite)
- [10. Visualization Overview](#10-visualization-overview)
- [11. Limitations & Future Scope](#11-limitations--future-scope)

---

## 1. Project Introduction & Problem Statement

Conventional Numerical Weather Prediction (NWP) involves solving complex physical fluid dynamics and thermodynamic equations on supercomputers. Machine Learning provides an empirical, data-driven approach that learns non-linear relationships between atmospheric features (temperature, relative humidity, atmospheric pressure, wind speed, cloud cover) and precipitation events.

Most educational projects rely on synthetic CSVs or suffer from a **training/inference mismatch** (e.g., training on today's measurements to predict tomorrow's rainfall, but feeding tomorrow's forecast at inference time). This project solves both issues:
1. It trains on **verified multi-year historical observations (ERA5 Reanalysis)** from the Open-Meteo Archive API (dynamic date range from 2023 through the current date in November 2026).
2. It formulates the ML task consistently: given daily atmospheric measurements, predict whether rainfall occurs on that day (`rain_today`). At inference time, tomorrow's forecasted atmospheric measurements are fed into the model to predict tomorrow's rainfall.

---

## 2. Project Objectives

- **Data Warehousing & Collection:** Download verified daily historical meteorological records from the Open-Meteo Historical Archive API (Delhi, 2023–Nov 2026).
- **Data Preprocessing & Cleaning:** Remove duplicate date entries, drop null records, validate data types, and engineer the binary classification target `rain_today = 1 if precipitation > 0 else 0`.
- **Classification Modeling:** Implement Decision Tree Classifier and Gaussian Naive Bayes using Scikit-learn with a stratified 70/30 train/test split.
- **Model Evaluation:** Calculate genuine Accuracy, Confusion Matrix, and Classification Reports (Precision, Recall, F1-score) from real data with aligned comparison reporting.
- **Real-Time Forecasting:** Integrate the Open-Meteo Geocoding and Forecast APIs to accept any global city name, retrieve tomorrow's forecasted atmospheric parameters, and predict rainfall likelihood.
- **Simplified Visualizations:** Generate clear, readable Matplotlib charts (performance comparison, confusion matrix heatmaps, and historical monthly trends) designed for lab viva presentation.
- **Automated Verification:** Include an end-to-end test suite (`test_verification.py`) testing data integrity, model training, live API connectivity, and chart generation.

---

## 3. Strict Academic & Lab Restrictions

To strictly comply with the **Data Mining and Warehousing Lab Manual (CSAD2500)**, the project adheres to the following constraints:

### Permitted External Libraries Only:
- **Pandas**: Tabular data manipulation, cleaning, and preprocessing.
- **NumPy**: Numerical operations, array handling, and matrix indexing.
- **Matplotlib**: 2D plotting, distribution charts, and confusion matrix visualization.
- **Scikit-learn**: Classification models and evaluation functions.

### Permitted ML Algorithms:
- `DecisionTreeClassifier(random_state=42)`
- `GaussianNB()`

### Permitted Scikit-learn Functions:
- `train_test_split`
- `accuracy_score`
- `confusion_matrix`
- `classification_report`

*No forbidden frameworks (Random Forest, XGBoost, TensorFlow, PyTorch, Seaborn, Flask, Django, Streamlit, Joblib) are used. Standard Python modules (`json`, `urllib`, `datetime`, `time`, `os`, `pathlib`) handle networking and file I/O.*

---

## 4. System Architecture & Directory Structure

```
Weather_Prediction_Project_A_Lab_Compatible/
│
├── data/
│   ├── weather_historical.csv    # Real historical records from Open-Meteo Archive API (Delhi 2023-2026)
│   └── weather_sample.csv        # Fallback sample dataset
│
├── plots/                        # Generated Matplotlib charts (saved as PNG)
│   ├── model_comparison.png      # Grouped bar chart: Accuracy & Rain Recall
│   ├── confusion_matrices.png    # Side-by-side Confusion Matrix heatmaps
│   └── historical_trends.png     # Stacked subplots: Monthly Temp & Rainfall trends
│
├── weather_prediction.py         # Main CLI application with streamlined interactive menu
├── data_collection.py            # Historical data fetching, cleaning, and CSV caching
├── model_training.py             # Feature selection, train/test split, DT & NB training & metrics
├── live_forecast.py              # Geocoding, live API retrieval with retry logic, and inference
├── visualization.py              # Clean Matplotlib visualizer (simple, readable code)
├── test_verification.py          # Complete 5-stage automated verification test suite
├── requirements.txt              # Project library dependencies
├── README.md                     # Comprehensive project documentation
├── REPORT_OUTLINE.md             # Formal lab report outline with actual evaluation figures
└── VIVA.md                       # Frequently asked lab viva questions & answers
```

---

## 5. Data Sources & Variables

### Historical Dataset Source
- **API**: Open-Meteo Historical Weather Archive (`https://archive-api.open-meteo.com/v1/archive`)
- **Default Location**: Delhi, India (Lat: `28.6139`, Lon: `77.2090`)
- **Date Range**: 1 January 2023 to current date (dynamically retrieved using `datetime.date.today().isoformat()`, covering 1,096+ verified daily records)
- **Quality**: ERA5 atmospheric reanalysis with zero artificial generation or synthetic imputation.

### Input Features & Target
| Feature | Column Name | Source Variable (Open-Meteo) | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Temperature** | `temperature_c` | `temperature_2m_mean` | °C | Mean daily 2-meter air temperature |
| **Relative Humidity** | `humidity_percent` | `relative_humidity_2m_mean` | % | Mean daily relative humidity |
| **Atmospheric Pressure** | `pressure_hpa` | `surface_pressure_mean` | hPa | Mean daily surface atmospheric pressure |
| **Wind Speed** | `wind_speed_kmh` | `wind_speed_10m_max` | km/h | Maximum daily 10-meter wind speed |
| **Cloud Cover** | `cloud_cover_percent` | `cloud_cover_mean` | % | Mean daily cloud cover percentage |
| **Target Variable** | `rain_today` | `precipitation_sum > 0` | 1 / 0 | Binary target: 1 if precipitation occurred, 0 otherwise |

### Live APIs (Open-Meteo)
1. **Geocoding API**: Resolves city names into coordinates and timezones:
   `https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json`
2. **Forecast API**: Fetches tomorrow's 5 atmospheric features with built-in retry logic:
   `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=...`

---

## 6. Model Evaluation Results

Trained on 1,096 real historical daily observations with a 70/30 stratified train/test split (Training: 767, Testing: 329):

```text
==============================================================================
                         MODEL PERFORMANCE COMPARISON                         
==============================================================================
Metric                             |  Decision Tree   |  Gaussian Naive Bayes 
------------------------------------------------------------------------------
Overall Accuracy                   |      84.19%      |         88.15%        
Rainy-Class Recall (Sensitivity)   |      77.50%      |         80.00%        
Rainy-Class Precision              |      78.81%      |         86.49%        
==============================================================================
```

### Detailed Classification Metrics:
- **Decision Tree Classifier (`random_state=42`)**:
  - Accuracy: **84.19%**
  - Rainy Class (1) Precision: **78.81%** | Recall: **77.50%** | F1-Score: **0.78**
  - No-Rain Class (0) Precision: **87.20%** | Recall: **88.04%** | F1-Score: **0.88**
  - Confusion Matrix ($N=329$): $\text{TN}=184,\; \text{FP}=25,\; \text{FN}=27,\; \text{TP}=93$

- **Gaussian Naive Bayes (`GaussianNB()`)**:
  - Accuracy: **88.15%**
  - Rainy Class (1) Precision: **86.49%** | Recall: **80.00%** | F1-Score: **0.83**
  - No-Rain Class (0) Precision: **88.99%** | Recall: **92.82%** | F1-Score: **0.91**
  - Confusion Matrix ($N=329$): $\text{TN}=194,\; \text{FP}=15,\; \text{FN}=24,\; \text{TP}=96$

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8+ (tested on Python 3.11 on Windows)
- Active internet connection (for live geocoding and forecast queries)

### Step 1: Clone the Repository
```bash
git clone https://github.com/akshatmathurtech/weather_forcasting.git
cd weather_forcasting
```

### Step 2: Install Permitted Dependencies
```bash
pip install -r requirements.txt
```
*(Or install manually: `pip install pandas numpy matplotlib scikit-learn`)*

---

## 8. Running the Application

Execute the main interactive CLI driver:
```bash
python weather_prediction.py
```

### Main Menu Interface:
```text
==================================================
                MAIN MENU
==================================================
1. View model evaluation and performance comparison
2. Predict tomorrow's weather for a city (Live API)
3. View visualizations (Matplotlib)
4. Exit
==================================================
```

### Menu Options Explained:
1. **Option 1 (View Model Evaluation)**:
   Prints detailed confusion matrices, classification reports, and a formatted comparison table for Decision Tree and Gaussian Naive Bayes.
2. **Option 2 (Predict Tomorrow's Weather for a City)**:
   Prompts for any global city name (e.g., `Delhi`, `Mumbai`, `London`, `Tokyo`, `New York`), geocodes the coordinates, retrieves tomorrow's atmospheric measurements via Open-Meteo Forecast API, and displays classification predictions (`Rain likely` / `No rain likely`).
3. **Option 3 (View Visualizations)**:
   Opens an interactive Matplotlib submenu:
   ```text
   --- VISUALIZATION MENU (Matplotlib) ---
   1. Decision Tree vs Naive Bayes Accuracy & Recall Comparison
   2. Confusion Matrices for Both Models
   3. Historical Weather Trends (Monthly Temp & Rain)
   4. Generate and Display All Visualizations
   5. Return to Main Menu
   ```
4. **Option 4 (Exit)**:
   Gracefully exits the program.

---

## 9. Automated Verification Suite

An automated verification test script is included to validate all modules without requiring manual interaction:

```bash
python test_verification.py
```

### Verification Pipeline:
- **Test 1**: Verifies historical CSV dataset integrity (all 5 features + target, 0 missing values, date continuity).
- **Test 2**: Trains both Decision Tree and Gaussian Naive Bayes models and verifies accuracy thresholds (>70%).
- **Test 3**: Tests Open-Meteo Geocoding and Tomorrow Forecast APIs across multiple cities (`Delhi`, `London`, `Tokyo`).
- **Test 4**: Executes real-time ML inference on live forecast values and verifies prediction labels.
- **Test 5**: Generates all 3 Matplotlib charts and verifies file existence and non-zero byte size in `/plots`.

---

## 10. Visualization Overview

The visualization module ([`visualization.py`](file:///visualization.py)) is designed to be concise, readable, and easy to explain in a viva examination:

1. **Model Performance Comparison (`plots/model_comparison.png`)**:
   A side-by-side grouped bar chart comparing Overall Accuracy and Rainy-Class Recall between Decision Tree and Gaussian Naive Bayes with percentage labels.
2. **Confusion Matrices (`plots/confusion_matrices.png`)**:
   Side-by-side $2 \times 2$ heatmaps with numerical cell annotations displaying True Negatives, False Positives, False Negatives, and True Positives.
3. **Historical Weather Trends (`plots/historical_trends.png`)**:
   A stacked two-panel subplot sharing the monthly timeline:
   - **Top Panel**: Monthly average temperature trend curve (°C).
   - **Bottom Panel**: Total monthly precipitation bar chart (mm).

---

## 11. Limitations & Future Scope

### Limitations
- **Atmospheric Model Resolution**: Open-Meteo uses global NWP models (ECMWF, GFS, ICON) at ~11 km resolution; hyper-local microclimates may exhibit slight variance.
- **Geographic Generalization**: Training on Delhi's continental monsoonal climate means the model performs best in similar climate regimes.
- **Academic Scope**: Constrained to Decision Tree and Gaussian Naive Bayes as required by syllabus guidelines without ensemble methods (Random Forest, Gradient Boosting).

### Future Scope
- Multi-city distributed training where localized models are trained per climate zone.
- Continuous online learning as new daily records pass to dynamically update historical records.
- Integration of hourly precipitation prediction models for short-term nowcasting.
