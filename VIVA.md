# Data Mining & Warehousing Lab: Viva Questions & Answers

### 1. What is classification in data mining?
Classification is a supervised learning technique where a model learns from historical labelled training data to predict discrete categorical class labels for new, unseen feature observations.

### 2. What is the target variable in this project, and how is it defined?
The target variable is `rain_today`, which is a binary indicator:
- `1` (Rain likely) if measured daily precipitation is greater than `0.0 mm`.
- `0` (No rain likely) if measured daily precipitation equals `0.0 mm`.

### 3. What input features are used?
Five continuous meteorological measurements:
1. `temperature_c`: Mean 2-meter air temperature (°C).
2. `humidity_percent`: Mean 2-meter relative humidity (%).
3. `pressure_hpa`: Mean surface atmospheric pressure (hPa).
4. `wind_speed_kmh`: Maximum 10-meter wind speed (km/h).
5. `cloud_cover_percent`: Mean total cloud cover (%).

### 4. How does the Decision Tree Classifier work?
The Decision Tree builds an inverted tree structure by recursively splitting the dataset based on feature thresholds that maximize class purity (minimizing Gini Impurity or Entropy). At inference, a sample traverses root-to-leaf decision nodes to determine its final class.

### 5. How does Gaussian Naive Bayes work?
Gaussian Naive Bayes applies Bayes' Theorem with the "naive" assumption of conditional feature independence given the class. For continuous numerical features, it models the likelihood $P(x_i | C_k)$ using a Gaussian (Normal) distribution defined by the mean $\mu$ and variance $\sigma^2$ calculated from training instances.

### 6. Why did we replace the synthetic CSV with Open-Meteo Historical data?
Synthetic data contains artificial mathematical distributions that do not reflect true meteorological thermodynamics or atmospheric correlations. The Open-Meteo Archive provides verified ERA5 atmospheric reanalysis records covering 2023–2025 (1,096 days), allowing genuine model training and realistic evaluation.

### 7. What is the "training/inference mismatch" and how was it resolved?
In naive implementations, models are trained on today's weather to predict tomorrow's rain, but then during deployment, users pass *tomorrow's* forecast values. This misaligns the semantic meaning of the inputs. In this project, the model learns the relationship between atmospheric conditions on day $D$ and rain on day $D$. At inference time, tomorrow's forecasted atmospheric conditions are fed into the model to predict whether day $D+1$ will experience rain.

### 8. What is the purpose of `train_test_split` with stratification?
`train_test_split` divides data into 70% training and 30% testing. Stratification ensures that the proportion of rainy days (36.5%) and dry days (63.5%) is preserved identically in both training and testing subsets, preventing class imbalance bias during evaluation.

### 9. What is a Confusion Matrix and what do TP, FP, TN, and FN mean here?
A Confusion Matrix cross-tabulates actual vs predicted classes:
- **True Positive (TP)**: Actual Rain, correctly predicted as Rain.
- **True Negative (TN)**: Actual No Rain, correctly predicted as No Rain.
- **False Positive (FP)**: Actual No Rain, incorrectly predicted as Rain (Type I error).
- **False Negative (FN)**: Actual Rain, incorrectly predicted as No Rain (Type II error).

### 10. How does live city prediction work without third-party web frameworks?
Using Python's standard library `urllib` and `json`:
1. The user inputs a city name.
2. The Open-Meteo Geocoding API converts the city name to latitude and longitude.
3. The Open-Meteo Forecast API fetches tomorrow's weather metrics for those coordinates.
4. The trained Decision Tree and Naive Bayes models take those 5 metrics and output the prediction.
