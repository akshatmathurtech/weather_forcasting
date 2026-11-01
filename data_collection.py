"""
data_collection.py
------------------
Module for downloading, preprocessing, and managing historical weather data
from the Open-Meteo Historical Weather Archive API.

Part of: Weather Prediction Using Data Mining Techniques
Lab: Data Mining and Warehousing Lab (CSAD2500)
"""

import os
import json
import urllib.request
import urllib.error
import pandas as pd
import datetime

# Default parameters
DEFAULT_CITY = "Delhi"
DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090
DEFAULT_START_DATE = "2023-01-01"
# End date will be computed dynamically to the current date (e.g., up to Nov 2026)
DEFAULT_END_DATE = datetime.date.today().isoformat()



# Default parameters
DEFAULT_CITY = "Delhi"
DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2025-12-31"

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Feature definitions
FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "pressure_hpa",
    "wind_speed_kmh",
    "cloud_cover_percent"
]
TARGET_COLUMN = "rain_today"


def fetch_historical_weather(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    location_name: str = DEFAULT_CITY,
    timeout_sec: int = 30
) -> pd.DataFrame:
    """
    Downloads historical daily weather observations from Open-Meteo Archive API.
    
    Parameters:
        latitude (float): Latitude coordinate of the location.
        longitude (float): Longitude coordinate of the location.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        location_name (str): Human-readable name for logging.
        timeout_sec (int): Network timeout in seconds.
        
    Returns:
        pd.DataFrame: Cleaned historical dataset with standard feature names.
    """
    daily_vars = (
        "temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "surface_pressure_mean,"
        "wind_speed_10m_max,"
        "cloud_cover_mean,"
        "precipitation_sum"
    )
    url = (
        f"{ARCHIVE_API_URL}?"
        f"latitude={latitude}&longitude={longitude}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"daily={daily_vars}&timezone=auto"
    )

    print(f"\n[Data Collection] Contacting Open-Meteo Historical Archive API...")
    print(f"Location    : {location_name} (Lat: {latitude}, Lon: {longitude})")
    print(f"Date Range  : {start_date} to {end_date}")
    print(f"API Endpoint: {ARCHIVE_API_URL}")

    headers = {"User-Agent": "WeatherPredictionLab/2.0 (Academic Project)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            if response.status != 200:
                raise RuntimeError(f"Open-Meteo API returned HTTP status {response.status}")
            raw_bytes = response.read()
            data = json.loads(raw_bytes.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ConnectionError(f"HTTP Error while connecting to Open-Meteo API: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Network Error: Unable to reach Open-Meteo Historical API. "
            f"Please check your internet connection. Reason: {e.reason}"
        )
    except TimeoutError:
        raise TimeoutError("Connection to Open-Meteo API timed out. Please try again.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching historical weather data: {e}")

    if "daily" not in data or not data["daily"]:
        raise ValueError("API response did not contain daily weather records.")

    daily = data["daily"]
    required_keys = [
        "time",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "wind_speed_10m_max",
        "cloud_cover_mean",
        "precipitation_sum"
    ]
    for key in required_keys:
        if key not in daily:
            raise KeyError(f"Expected variable '{key}' missing from API response.")

    # Construct DataFrame with verified observations
    df = pd.DataFrame({
        "date": daily["time"],
        "temperature_c": daily["temperature_2m_mean"],
        "humidity_percent": daily["relative_humidity_2m_mean"],
        "pressure_hpa": daily["surface_pressure_mean"],
        "wind_speed_kmh": daily["wind_speed_10m_max"],
        "cloud_cover_percent": daily["cloud_cover_mean"],
        "precipitation_mm": daily["precipitation_sum"]
    })

    # Data Cleaning and Preprocessing:
    # 1. Remove duplicate date entries
    initial_len = len(df)
    df = df.drop_duplicates(subset=["date"])

    # 2. Handle missing/null values without inventing artificial data
    df = df.dropna()

    # 3. Create target variable for the same day:
    # rain_today = 1 if precipitation > 0, else 0
    df[TARGET_COLUMN] = (df["precipitation_mm"] > 0.0).astype(int)

    final_len = len(df)
    print(f"\n[Data Collection Summary]")
    print(f"- Source       : Open-Meteo Historical Weather Archive (Reanalysis ERA5)")
    print(f"- Location     : {location_name} (Lat: {latitude}, Lon: {longitude})")
    print(f"- Date Range   : {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"- Total Records: {final_len} valid daily records (dropped {initial_len - final_len} invalid/duplicate)")
    print(f"- Target Dist  : {df[TARGET_COLUMN].value_counts().to_dict()} (0=No Rain, 1=Rain)")

    return df


def save_historical_data(df: pd.DataFrame, filepath: str) -> None:
    """Saves DataFrame to CSV file ensuring parent directories exist."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"- Saved CSV to : {filepath}")


def load_or_fetch_historical_data(
    filepath: str,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Loads historical dataset from disk if available, otherwise downloads it
    from the Open-Meteo API. If force_refresh is True, always downloads fresh data.
    """
    if not force_refresh and os.path.exists(filepath):
        print(f"\n[Data Loading] Loading verified historical data from {filepath}...")
        df = pd.read_csv(filepath)
        df = df.drop_duplicates(subset=["date"] if "date" in df.columns else None).dropna()

        # Validate that necessary features exist
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                print(f"[Warning] Column '{col}' missing. Re-fetching dataset from API...")
                df = fetch_historical_weather()
                save_historical_data(df, filepath)
                return df

        if TARGET_COLUMN not in df.columns:
            if "precipitation_mm" in df.columns:
                df[TARGET_COLUMN] = (df["precipitation_mm"] > 0.0).astype(int)
            elif "rain_tomorrow" in df.columns:
                # Fallback if an old CSV format is present
                df[TARGET_COLUMN] = df["rain_tomorrow"]

        print(f"- Records loaded : {len(df)}")
        print(f"- Date range     : {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"- Class counts   : {df[TARGET_COLUMN].value_counts().to_dict()}")
        return df

    # Otherwise fetch from Open-Meteo Archive
    df = fetch_historical_weather()
    save_historical_data(df, filepath)
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "data", "weather_historical.csv")
    df = load_or_fetch_historical_data(target_csv, force_refresh=True)
    print("\nFirst 5 rows:")
    print(df.head())
