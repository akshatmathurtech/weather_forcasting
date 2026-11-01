"""
live_forecast.py
----------------
Module for geocoding user-entered cities, fetching live tomorrow forecast data
from Open-Meteo, and predicting rainfall using trained ML models.

Part of: Weather Prediction Using Data Mining Techniques
Lab: Data Mining and Warehousing Lab (CSAD2500)
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import datetime
import time
import pandas as pd

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "pressure_hpa",
    "wind_speed_kmh",
    "cloud_cover_percent"
]


def geocode_city(city_name: str, timeout_sec: int = 15) -> dict:
    """
    Converts a human-readable city name into latitude, longitude, and metadata
    using the Open-Meteo Geocoding API.
    
    Returns:
        dict: Geocoding details (name, country, latitude, longitude, timezone).
    Raises:
        ValueError: If city name is invalid or not found.
        ConnectionError: If network/API connection fails.
    """
    clean_name = city_name.strip()
    if not clean_name:
        raise ValueError("City name cannot be empty.")

    encoded_name = urllib.parse.quote(clean_name)
    url = f"{GEOCODING_API_URL}?name={encoded_name}&count=1&language=en&format=json"

    headers = {"User-Agent": "WeatherPredictionLab/2.0 (Academic Project)"}
    req = urllib.request.Request(url, headers=headers)

    last_err = None
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                if response.status != 200:
                    raise RuntimeError(f"Geocoding API returned HTTP status {response.status}")
                data = json.loads(response.read().decode("utf-8"))
                break
        except urllib.error.URLError as e:
            last_err = ConnectionError(f"Network error while connecting to Geocoding API: {e.reason}")
        except TimeoutError:
            last_err = TimeoutError("Geocoding API request timed out. Please check your connection.")
        except Exception as e:
            last_err = e
        time.sleep(1)

    if data is None:
        raise last_err or RuntimeError("Could not retrieve geocoding data.")

    results = data.get("results")
    if not results or len(results) == 0:
        raise ValueError(
            f"City '{city_name}' could not be found. "
            f"Please check the spelling and try again (e.g., Delhi, Mumbai, London, Tokyo)."
        )

    first = results[0]
    return {
        "name": first.get("name", clean_name),
        "country": first.get("country", "Unknown"),
        "admin1": first.get("admin1", ""),
        "latitude": round(float(first["latitude"]), 4),
        "longitude": round(float(first["longitude"]), 4),
        "timezone": first.get("timezone", "auto")
    }


def fetch_tomorrow_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    timeout_sec: int = 15
) -> dict:
    """
    Fetches the live tomorrow weather forecast for the specified coordinates
    from the Open-Meteo Forecast API.
    
    Extracts the exact 5 daily features matching model training:
    - temperature_2m_mean -> temperature_c
    - relative_humidity_2m_mean -> humidity_percent
    - surface_pressure_mean -> pressure_hpa
    - wind_speed_10m_max -> wind_speed_kmh
    - cloud_cover_mean -> cloud_cover_percent
    """
    daily_vars = (
        "temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "surface_pressure_mean,"
        "wind_speed_10m_max,"
        "cloud_cover_mean,"
        "precipitation_sum"
    )
    tz_param = urllib.parse.quote(timezone) if timezone else "auto"
    url = (
        f"{FORECAST_API_URL}?"
        f"latitude={latitude}&longitude={longitude}&"
        f"daily={daily_vars}&timezone={tz_param}"
    )

    headers = {"User-Agent": "WeatherPredictionLab/2.0 (Academic Project)"}
    req = urllib.request.Request(url, headers=headers)

    last_err = None
    data = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                if response.status != 200:
                    raise RuntimeError(f"Forecast API returned HTTP status {response.status}")
                data = json.loads(response.read().decode("utf-8"))
                break
        except urllib.error.URLError as e:
            last_err = ConnectionError(f"Network error while connecting to Forecast API: {e.reason}")
        except TimeoutError:
            last_err = TimeoutError("Forecast API request timed out. Please try again.")
        except Exception as e:
            last_err = e
        time.sleep(1)

    if data is None:
        raise last_err or RuntimeError("Could not retrieve forecast data.")

    daily = data.get("daily")
    if not daily or "time" not in daily or len(daily["time"]) < 2:
        raise ValueError("Forecast API did not return sufficient daily forecast entries.")

    # Determine tomorrow's index
    dates = daily["time"]
    today_str = datetime.date.today().isoformat()
    tomorrow_str = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    if tomorrow_str in dates:
        idx = dates.index(tomorrow_str)
    elif len(dates) >= 2:
        # Index 1 corresponds to tomorrow if today is index 0
        idx = 1
        tomorrow_str = dates[idx]
    else:
        idx = 0
        tomorrow_str = dates[0]

    # Validate that none of tomorrow's feature values are null
    temp = daily["temperature_2m_mean"][idx]
    humidity = daily["relative_humidity_2m_mean"][idx]
    pressure = daily["surface_pressure_mean"][idx]
    wind = daily["wind_speed_10m_max"][idx]
    cloud = daily["cloud_cover_mean"][idx]
    precip = daily.get("precipitation_sum", [None])[idx]

    if any(v is None for v in [temp, humidity, pressure, wind, cloud]):
        raise ValueError(
            f"Forecast API returned incomplete data for tomorrow ({tomorrow_str})."
        )

    return {
        "date": tomorrow_str,
        "temperature_c": float(temp),
        "humidity_percent": float(humidity),
        "pressure_hpa": float(pressure),
        "wind_speed_kmh": float(wind),
        "cloud_cover_percent": float(cloud),
        "expected_precipitation_mm": float(precip) if precip is not None else 0.0
    }


def predict_tomorrow_weather(models_dict: dict, forecast_features: dict) -> dict:
    """
    Passes tomorrow's forecast features into both trained ML models.
    
    Prediction labels:
    - Class 1: "Rain likely"
    - Class 0: "No rain likely"
    """
    dt_model = models_dict["Decision Tree"]
    nb_model = models_dict["Gaussian Naive Bayes"]

    input_df = pd.DataFrame([{
        "temperature_c": forecast_features["temperature_c"],
        "humidity_percent": forecast_features["humidity_percent"],
        "pressure_hpa": forecast_features["pressure_hpa"],
        "wind_speed_kmh": forecast_features["wind_speed_kmh"],
        "cloud_cover_percent": forecast_features["cloud_cover_percent"]
    }], columns=FEATURE_COLUMNS)

    dt_pred = int(dt_model.predict(input_df)[0])
    nb_pred = int(nb_model.predict(input_df)[0])

    dt_label = "Rain likely" if dt_pred == 1 else "No rain likely"
    nb_label = "Rain likely" if nb_pred == 1 else "No rain likely"

    return {
        "dt_prediction": dt_pred,
        "dt_label": dt_label,
        "nb_prediction": nb_pred,
        "nb_label": nb_label,
        "features_used": input_df.to_dict(orient="records")[0]
    }


def display_forecast_and_prediction(
    geo: dict,
    forecast: dict,
    prediction: dict
) -> None:
    """Displays formatted live forecast data and Machine Learning predictions."""
    print("\n" + "=" * 60)
    print("        LIVE WEATHER FORECAST & ML PREDICTION")
    print("=" * 60)
    region = f", {geo['admin1']}" if geo.get("admin1") else ""
    print(f"Selected City       : {geo['name']}{region}, {geo['country']}")
    print(f"Geographic Coords   : Lat {geo['latitude']}, Lon {geo['longitude']}")
    print(f"Tomorrow's Date     : {forecast['date']}")
    print("-" * 60)
    print("TOMORROW'S FORECASTED ATMOSPHERIC MEASUREMENTS:")
    print(f"  - Temperature     : {forecast['temperature_c']:.1f} °C")
    print(f"  - Humidity        : {forecast['humidity_percent']:.1f} %")
    print(f"  - Surface Pressure: {forecast['pressure_hpa']:.1f} hPa")
    print(f"  - Max Wind Speed  : {forecast['wind_speed_kmh']:.1f} km/h")
    print(f"  - Cloud Cover     : {forecast['cloud_cover_percent']:.1f} %")
    print("-" * 60)
    print("MACHINE LEARNING CLASSIFICATION PREDICTIONS:")
    print(f"  [1] Decision Tree Classifier : {prediction['dt_label']}")
    print(f"  [2] Gaussian Naive Bayes     : {prediction['nb_label']}")
    print("=" * 60)
    print("NOTE:")
    print("  These results are machine learning classifications trained on")
    print("  historical weather data and fed with tomorrow's forecast values.")
    print("  They demonstrate data mining algorithms and are NOT official")
    print("  meteorological warnings or severe weather advisories.")
    print("=" * 60)
