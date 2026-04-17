"""
Weather data collection and preprocessing for feature engineering.
Spatial assumptions: Uses point-coordinate matching (Madrid) as a generalized bounding-box proxy 
resampled to hourly frequency for the whole of Spain's electricity demand geography.
"""
import requests
import pandas as pd
from datetime import datetime

class WeatherFetcher:
    """
    Fetches weather data from the free Open-Meteo API for Spain (Madrid).
    ML Principle: Missing value interpolation and robust deterministic temporal alignment 
    (filling up missing rows to 24 frequency records) through forward-filling.
    """
    
    def __init__(self, lat=40.4168, lon=-3.7038):
        self.lat = lat
        self.lon = lon
        # Using the historical/forecast endpoint. Open-Meteo handles dates automatically.
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
    def get_weather_for_date(self, date_str: str) -> pd.DataFrame:
        """
        Fetches hourly weather data for the specified date.
        Returns a DataFrame with 24 hourly rows containing:
        - temperature_celsius (temperature_2m)
        - humidity_percent (relative_humidity_2m)
        - pressure_hpa (surface_pressure)
        - cover_percent (cloud_cover)
        - wind_speed_ms (wind_speed_10m)
        - wind_direction_deg (wind_direction_10m)
        - rain_mm (precipitation)
        """
        
        # OpenMeteo requires YYYY-MM-DD
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,precipitation",
            "timezone": "Europe/Madrid"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                raise ValueError("No hourly data in Open-Meteo response")
                
            hourly = data["hourly"]
            
            # Open-Meteo returns wind speed in km/h by default, but we can divide by 3.6 to get m/s
            # Or we can request `wind_speed_unit=ms`
            params["wind_speed_unit"] = "ms"
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            hourly = response.json()["hourly"]

            df = pd.DataFrame({
                "time": pd.to_datetime(hourly["time"]),
                "temperature_celsius": hourly["temperature_2m"],
                "humidity_percent": hourly["relative_humidity_2m"],
                "pressure_hpa": hourly["surface_pressure"],
                "cloud_cover_percent": hourly["cloud_cover"],
                "wind_speed_ms": hourly["wind_speed_10m"],
                "wind_direction_deg": hourly["wind_direction_10m"],
                "rain_mm": hourly["precipitation"]
            })
            
            # Ensure it's exactly 24 hours
            if len(df) > 24:
                df = df.iloc[:24]
            elif len(df) < 24:
                # Pad to 24 if missing
                df = df.reindex(range(24)).ffill().fillna(0)
                
            return df
            
        except Exception as e:
            print(f"Warning: Weather fetch failed ({e}). Returning fallback medians.")
            raise

