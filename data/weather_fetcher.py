"""
Weather data fetcher for match venues.
Uses Open-Meteo API for historical weather data (free, no API key needed).
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import time


class WeatherCache:
    """Simple in-memory cache for weather data to avoid repeated API calls."""
    
    def __init__(self):
        self._cache = {}
    
    def get_key(self, lat, lon, date):
        """Generate cache key from location and date."""
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
        return f"{lat:.4f},{lon:.4f},{date_str}"
    
    def get(self, lat, lon, date):
        """Get cached weather data."""
        return self._cache.get(self.get_key(lat, lon, date))
    
    def set(self, lat, lon, date, data):
        """Store weather data in cache."""
        self._cache[self.get_key(lat, lon, date)] = data


# Global cache instance
_weather_cache = WeatherCache()


def fetch_historical_weather(lat: float, lon: float, date: datetime, 
                            use_cache: bool = True) -> Optional[Dict]:
    """
    Fetch historical weather data for a specific location and date.
    
    Args:
        lat: Latitude of venue
        lon: Longitude of venue
        date: Match date (datetime object)
        use_cache: Whether to use cached results
    
    Returns:
        Dict with weather features or None if fetch fails:
        {
            'temperature': float (°C),
            'precipitation': float (mm),
            'wind_speed': float (km/h),
            'cloud_cover': float (0-100%),
            'humidity': float (0-100%),
        }
    """
    # Check cache first
    if use_cache:
        cached = _weather_cache.get(lat, lon, date)
        if cached is not None:
            return cached
    
    # Convert datetime to date string
    date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
    
    # Open-Meteo API endpoint for historical data
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,cloud_cover_mean,relative_humidity_2m_mean",
        "timezone": "auto",
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract daily data
        if "daily" in data and data["daily"]:
            daily = data["daily"]
            
            weather_data = {
                "temperature": daily.get("temperature_2m_mean", [None])[0],
                "precipitation": daily.get("precipitation_sum", [None])[0],
                "wind_speed": daily.get("wind_speed_10m_max", [None])[0],
                "cloud_cover": daily.get("cloud_cover_mean", [None])[0],
                "humidity": daily.get("relative_humidity_2m_mean", [None])[0],
            }
            
            # Cache the result
            if use_cache:
                _weather_cache.set(lat, lon, date, weather_data)
            
            return weather_data
        
        return None
    
    except requests.RequestException as e:
        print(f"Warning: Failed to fetch weather for {lat},{lon} on {date_str}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error parsing weather data: {e}")
        return None


def get_weather_features(lat: float, lon: float, date: datetime, 
                        venue_type: str = "open",
                        use_cache: bool = True) -> Dict[str, float]:
    """
    Get weather features for a match venue, with fallback defaults.
    
    Args:
        lat: Latitude of venue
        lon: Longitude of venue  
        date: Match date
        venue_type: 'open', 'retractable', or 'dome'
        use_cache: Whether to use cached results
    
    Returns:
        Dict with normalized weather features (0.0 if dome/fetch fails):
        {
            'temperature_norm': float,      # Normalized -10°C to 30°C
            'precipitation': float,         # mm
            'wind_speed_norm': float,       # Normalized 0-50 km/h
            'cloud_cover_norm': float,      # 0-1 scale
            'humidity_norm': float,         # 0-1 scale
            'is_dome': float,               # 1.0 if dome, 0.0 otherwise
        }
    """
    # Default features (for dome or failed fetch)
    default_features = {
        "temperature_norm": 0.0,
        "precipitation": 0.0,
        "wind_speed_norm": 0.0,
        "cloud_cover_norm": 0.0,
        "humidity_norm": 0.0,
        "is_dome": 1.0 if venue_type == "dome" else 0.0,
    }
    
    # For dome stadiums, weather doesn't matter
    if venue_type == "dome":
        return default_features
    
    # Fetch weather data
    weather = fetch_historical_weather(lat, lon, date, use_cache=use_cache)
    
    if weather is None:
        return default_features
    
    # Normalize features to reasonable ranges
    features = {
        "is_dome": 0.0,
    }
    
    # Temperature: normalize -10°C to 30°C to 0-1 range
    if weather["temperature"] is not None:
        temp = max(-10, min(30, weather["temperature"]))
        features["temperature_norm"] = (temp + 10) / 40.0
    else:
        features["temperature_norm"] = 0.5  # Default to mild
    
    # Precipitation: keep as mm (0-50mm typical)
    features["precipitation"] = weather["precipitation"] if weather["precipitation"] is not None else 0.0
    
    # Wind speed: normalize 0-50 km/h to 0-1 range
    if weather["wind_speed"] is not None:
        wind = max(0, min(50, weather["wind_speed"]))
        features["wind_speed_norm"] = wind / 50.0
    else:
        features["wind_speed_norm"] = 0.0
    
    # Cloud cover: already 0-100, normalize to 0-1
    if weather["cloud_cover"] is not None:
        features["cloud_cover_norm"] = weather["cloud_cover"] / 100.0
    else:
        features["cloud_cover_norm"] = 0.5
    
    # Humidity: already 0-100, normalize to 0-1
    if weather["humidity"] is not None:
        features["humidity_norm"] = weather["humidity"] / 100.0
    else:
        features["humidity_norm"] = 0.5
    
    return features


def clear_cache():
    """Clear the weather data cache."""
    global _weather_cache
    _weather_cache = WeatherCache()
