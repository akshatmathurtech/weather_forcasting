# Mini Project Report Outline

## 1. Title
**Weather Prediction Using Data Mining Techniques with Real Historical Observations and Live Meteorological Forecasting**

---

## 2. Objective
To design, implement, and evaluate an empirical Machine Learning weather forecasting system that:
1. Downloads and warehouses verified multi-year historical atmospheric observations from the Open-Meteo Historical Archive API.
2. Cleans and transforms the dataset into a standardized feature set with a same-day precipitation target (`rain_today`).
3. Trains and benchmarks Decision Tree and Gaussian Naive Bayes classifiers following Data Mining Lab specifications.
4. Integrates live geocoding and tomorrow weather forecasting to classify rain occurrence for any specified city.
5. Visualizes class distributions, model performance, confusion matrices, and meteorological trends using Matplotlib.

---

## 3. Tools and Libraries
- **Language**: Python 3.11+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (`train_test_split`, `DecisionTreeClassifier`, `GaussianNB`, `accuracy_score`, `confusion_matrix`, `classification_report`)
- **Visualization**: Matplotlib
- **Standard Library Modules**: `json`, `urllib.request`, `urllib.parse`, `datetime`, `os`, `pathlib`
- **External Data Sources**: Open-Meteo Historical Weather API & Open-Meteo Forecast & Geocoding APIs

---

## 4. Methodology
1. **Data Collection & Warehousing**:
   - Source: Open-Meteo Archive (Delhi: Lat 28.6139, Lon 77.2090).
   - Records: 1,096 verified daily observations (1 January 2023 to 31 December 2025).
2. **Data Cleaning & Transformation**:
   - Duplicates removal, missing value handling (`dropna()`).
   - Feature Engineering: Extract 5 atmospheric attributes: Mean Temperature (`temperature_c`), Relative Humidity (`humidity_percent`), Surface Pressure (`pressure_hpa`), Max Wind Speed (`wind_speed_kmh`), Mean Cloud Cover (`cloud_cover_percent`).
   - Target Creation: Binary indicator `rain_today = 1 if precipitation_mm > 0 else 0`.
3. **Model Training & Evaluation**:
   - 70% Training (767 records), 30% Testing (329 records) stratified by target class.
   - Algorithms: Decision Tree Classifier (`random_state=42`) and Gaussian Naive Bayes (`GaussianNB()`).
4. **Live Inference Pipeline**:
   - City input geocoded via Open-Meteo Geocoding API.
   - Tomorrow's daily forecast extracted via Open-Meteo Forecast API.
   - Prediction emitted by both models with "Rain likely" / "No rain likely" labels.
5. **Data Visualization**:
   - Matplotlib charts for class counts, comparative accuracy/recall bar plots, confusion matrices, and time-series trends.

---

## 5. Actual Experimental Results
*Results obtained from the verified 1,096-day dataset (No fabricated numbers):*

- **Dataset Breakdown**:
  - Class 0 (No Rain): 696 records (63.5%)
  - Class 1 (Rain): 400 records (36.5%)
  - Test Set (329 records): 209 No Rain, 120 Rain

- **Performance Metrics**:
| Metric | Decision Tree | Gaussian Naive Bayes |
| :--- | :---: | :---: |
| Overall Accuracy | **84.19%** | **88.15%** |
| Rainy-Class Precision | 78.81% | 86.49% |
| Rainy-Class Recall | 77.50% | 80.00% |
| Macro F1-Score | 0.83 | 0.87 |

- **Confusion Matrices**:
  - Decision Tree: TN = 184, FP = 25, FN = 27, TP = 93
  - Gaussian Naive Bayes: TN = 194, FP = 15, FN = 24, TP = 96

---

## 6. Analysis & Observations
1. **Gaussian Naive Bayes** achieved higher overall accuracy (88.15% vs 84.19%) and higher sensitivity/recall on rainy days (80.00% vs 77.50%). The continuous features follow roughly unimodal bell-shaped distributions within classes, fulfilling Gaussian assumptions reasonably well.
2. **Decision Tree** provides transparent rule-based partitioning (e.g., higher humidity and cloud cover paired with pressure drops trigger rain splits).
3. Addressing the training/inference feature alignment resolved past model drift and provided consistent predictions for live forecast inputs.

---

## 7. Limitations & Future Scope
- **Regional Dependence**: Training on Delhi data reflects North Indian climate patterns; training local models for other geographic regions would improve global accuracy.
- **Ensemble Techniques**: Future iterations can incorporate ensemble methods once syllabus constraints allow.
