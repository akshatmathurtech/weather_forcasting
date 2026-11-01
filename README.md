# AI/ML-Based Weather Forecasting with Historical Data and Live Predictions

A practical Artificial Intelligence and Machine Learning weather forecasting application developed for the **Data Mining and Warehousing Lab (CSAD2500)**.

The project downloads and cleans verified historical weather observations, trains **Decision Tree** and **Gaussian Naive Bayes** classifiers on real atmospheric parameters, geocodes user-queried cities, fetches live tomorrow forecast data, and predicts same-day/next-day precipitation without training/inference feature mismatches.

---

## 1. Project Introduction & Problem Statement

Conventional numerical weather prediction (NWP) involves solving complex physical fluid dynamics equations on supercomputers. Machine Learning provides an empirical, data-driven approach that learns non-linear relationships between atmospheric features (temperature, relative humidity, atmospheric pressure, wind speed, cloud cover) and precipitation events.

Most educational projects rely on synthetic CSVs or suffer from a **training/inference mismatch** (e.g., training on today's measurements to predict tomorrow's rainfall, but feeding tomorrow's forecast at inference time). This project fixes both issues:
1. It trains on **verified multi-year historical observations (ERA5 Reanalysis)** from the Open-Meteo Archive API.
2. It formulates the ML task consistently: given atmospheric measurements of a given day, predict whether rainfall occurs on that day (`rain_today`). At inference time, tomorrow's forecasted atmospheric measurements are fed into the model to predict tomorrow's rainfall.

---

## 2. Project Objectives

- **Data Warehousing & Collection:** Download verified daily historical meteorological records from the Open-Meteo Historical Archive API (Delhi, 2023–2025).
- **Data Preprocessing & Cleaning:** Remove duplicate date entries, drop null records, validate data types, and engineer the binary classification target `rain_today = 1 if precipitation > 0 else 0`.
- **Classification Modeling:** Implement Decision Tree Classifier and Gaussian Naive Bayes using Scikit-learn with stratified 70/30 train/test splitting.
- **Model Evaluation:** Calculate genuine Accuracy, Confusion Matrix, and Classification Reports (Precision, Recall, F1-score) from real data.
- **Real-Time Forecasting:** Integrate the Open-Meteo Geocoding and Forecast APIs to accept any global city name, retrieve tomorrow's forecasted features, and predict rainfall likelihood.
- **Visualization:** Generate clear Matplotlib visualizations (class distributions, confusion matrices, performance comparison, and historical trends).

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

*No forbidden frameworks (Random Forest, XGBoost, TensorFlow, PyTorch, Seaborn, Flask, Django, Streamlit, Joblib) are used. Standard Python modules (`json`, `urllib`, `datetime`, `os`, `pathlib`) handle networking and file I/O.*

---

## 4. System Architecture & Directory Structure

```
Weather_Prediction_Project_A_Lab_Compatible/
│
├── data/
│   ├── weather_historical.csv    # Real historical records from Open-Meteo Archive API (Delhi 2023-2025)
│   └── weather_sample.csv        # Retained synthetic sample CSV (backward compatibility fallback)
│
├── plots/                        # Generated Matplotlib charts (saved as PNG)
│   ├── rain_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── historical_trends.png
│
├── weather_prediction.py         # Main CLI application and interactive menu driver
├── data_collection.py            # Historical data fetching, cleaning, and CSV caching
├── model_training.py             # Feature selection, train/test split, DT & NB training & metrics
├── live_forecast.py              # Geocoding, tomorrow forecast API retrieval, and ML inference
├── visualization.py              # Matplotlib visualizer for distributions, CMs, and trends
├── README.md                     # Comprehensive project documentation
├── REPORT_OUTLINE.md             # Formal lab report outline with actual evaluation figures
└── VIVA.md                       # Frequently asked lab viva questions & answers
```

---

## 5. Data Sources & Variables

### Historical Dataset Source
- **API**: Open-Meteo Historical Weather Archive (`https://archive-api.open-meteo.com/v1/archive`)
- **Default Location**: Delhi, India (Lat: 28.6139, Lon: 77.2090)
- **Date Range**: 1 January 2023 to 31 December 2025 (1,096 verified daily records)
- **Quality**: ERA5 atmospheric reanalysis with zero artificial generation or synthetic imputation.

### Input Features & Target
| Feature | Column Name | Source Variable (Open-Meteo) | Unit |
| :--- | :--- | :--- | :--- |
| **Temperature** | `temperature_c` | `temperature_2m_mean` | °C |
| **Relative Humidity** | `humidity_percent` | `relative_humidity_2m_mean` | % |
| **Atmospheric Pressure** | `pressure_hpa` | `surface_pressure_mean` | hPa |
| **Wind Speed** | `wind_speed_kmh` | `wind_speed_10m_max` | km/h |
| **Cloud Cover** | `cloud_cover_percent` | `cloud_cover_mean` | % |
| **Target Variable** | `rain_today` | `precipitation_sum > 0` | 1 (Rain) / 0 (No Rain) |

### Live Forecast APIs
1. **Geocoding API**: `https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json`
2. **Forecast API**: `https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=...`

---

## 6. Model Evaluation Results (Actual Figures)

Trained on 1,096 real historical daily records with a 70/30 stratified split (Training: 767, Testing: 329):

| Evaluation Metric | Decision Tree Classifier | Gaussian Naive Bayes |
| :--- | :---: | :---: |
| **Overall Accuracy** | **84.19%** | **88.15%** |
| **Rainy-Class (1) Precision** | 78.81% | 86.49% |
| **Rainy-Class (1) Recall** | 77.50% | 80.00% |
| **Rainy-Class (1) F1-Score** | 0.78 | 0.83 |
| **No-Rain Class (0) Precision** | 87.20% | 88.99% |
| **No-Rain Class (0) Recall** | 88.04% | 92.82% |

### Confusion Matrices (Test Size = 329):
- **Decision Tree**:
  - TN: 184 | FP: 25
  - FN: 27  | TP: 93
- **Gaussian Naive Bayes**:
  - TN: 194 | FP: 15
  - FN: 24  | TP: 96

---

## 7. Installation & Setup

### Prerequisites
- Python 3.8+ (tested on Python 3.11 on Windows)
- Active internet connection (to fetch archive/forecast API data)

### Installation Command
Install the permitted external libraries in your terminal or VS Code Command Prompt:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 8. Running the Application

From the project root directory in VS Code or Terminal, execute:
```bash
python weather_prediction.py
```

### Main Menu Options:
```text
==================================================
                MAIN MENU
==================================================
1. Train models using historical data
2. View model evaluation and performance comparison
3. Predict tomorrow's weather for a city (Live API)
4. Enter manual weather conditions for prediction
5. View visualizations (Matplotlib)
6. Refresh / re-download historical dataset
7. Exit
==================================================
```

### Live City Prediction Workflow:
1. Select Option `3`.
2. Enter city name (e.g., `Delhi`, `Mumbai`, `London`, `Tokyo`).
3. The system geocodes the city, queries Open-Meteo for tomorrow's forecast, extracts the 5 features, and feeds them into both trained models.
4. Output displays tomorrow's forecasted atmospheric conditions alongside predictions:
   - Decision Tree: `Rain likely` or `No rain likely`
   - Naive Bayes: `Rain likely` or `No rain likely`

---

## 9. Limitations & Future Scope

### Limitations
- **Atmospheric Model Resolution**: Open-Meteo uses global NWP models (ECMWF, GFS, ICON) at ~11 km resolution; hyper-local microclimates may exhibit slight variance.
- **Geographic Generalization**: Training on Delhi's continental monsoonal climate means the model performs best in similar semi-arid/monsoon climate regimes.
- **Academic Scope**: Constrained to Decision Tree and Naive Bayes as required by syllabus guidelines without ensemble methods (Random Forest, Gradient Boosting).

### Future Scope
- Multi-city distributed training where localized models are trained per climate zone.
- Continuous online learning as new days pass to dynamically update historical CSV records.
