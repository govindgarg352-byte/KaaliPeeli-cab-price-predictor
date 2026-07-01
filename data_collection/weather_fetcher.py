"""
weather_fetcher.py
Fetches REAL historical weather (last 4 weeks) for all 5 cities using
Open-Meteo's free Archive API (no key required).
Maps WMO weather codes -> our 5 categories: Sunny, Cloudy, Rainy, Foggy, Stormy
"""

import requests
import json
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CITIES, WEATHER_BIAS  # CITIES[city]["weather_coords"] = (lat, lng)

RAW_DIR = "data/raw/weather_forecasts"
MASTER_LOOKUP_PATH = "data/raw/weather_lookup.json"

os.makedirs(RAW_DIR, exist_ok=True)

# ---- Date window: last 4 weeks ending today ----
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(weeks=4)


# ---- WMO Weather Code -> Our 5 Categories ----
def map_wmo_code(code: int) -> str:
    if code == 0:
        return "Sunny"
    elif code in [1, 2]:
        return "Sunny"
    elif code == 3:
        return "Cloudy"
    elif code in [45, 48]:
        return "Foggy"
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
        return "Rainy"
    elif code in [95, 96, 99]:
        return "Stormy"
    else:
        return "Cloudy"


def fetch_city_history(city: str, lat: float, lon: float) -> dict:
    """Fetch hourly weather codes for the date window and return {timestamp: condition}."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": "weathercode",
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    raw_path = os.path.join(RAW_DIR, f"{city.lower()}_history_raw.json")
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)

    times = data["hourly"]["time"]
    codes = data["hourly"]["weathercode"]

    lookup = {}
    for t, code in zip(times, codes):
        lookup[t] = map_wmo_code(code)

    parsed_path = os.path.join(RAW_DIR, f"{city.lower()}_lookup.json")
    with open(parsed_path, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"  {city}: {len(lookup)} hourly slots parsed ({START_DATE} -> {END_DATE})")
    if lookup:
        sample_ts = times[12]
        print(f"  Sample -> {sample_ts} : {lookup[sample_ts]}")

    return lookup


def save_master_lookup(all_city_lookups: dict):
    with open(MASTER_LOOKUP_PATH, "w") as f:
        json.dump(all_city_lookups, f, indent=2)
    print(f"\n💾 Master lookup saved to {MASTER_LOOKUP_PATH}")


def load_weather_lookup() -> dict:
    """Used by pipeline.py — no API call, just reads saved JSON."""
    with open(MASTER_LOOKUP_PATH, "r") as f:
        return json.load(f)


def fallback_weather_by_bias(city: str) -> str:
    """Probabilistic fallback using WEATHER_BIAS already defined in config.py."""
    bias = WEATHER_BIAS.get(
        city, {"Sunny": 0.4, "Cloudy": 0.3, "Rainy": 0.2, "Stormy": 0.05, "Foggy": 0.05}
    )
    return random.choices(list(bias.keys()), weights=list(bias.values()))[0]


def get_weather(city: str, timestamp: str, lookup: dict = None) -> str:
    """
    Lookup weather for a given city + timestamp.
    timestamp must match Open-Meteo's hourly format: 'YYYY-MM-DDTHH:00'
    Falls back to bias sampling if timestamp not found.
    """
    if lookup is None:
        lookup = load_weather_lookup()

    city_data = lookup.get(city, {})
    if timestamp in city_data:
        return city_data[timestamp]

    return fallback_weather_by_bias(city)


def get_all_timestamps(city: str, lookup: dict = None) -> list:
    if lookup is None:
        lookup = load_weather_lookup()
    return list(lookup.get(city, {}).keys())


if __name__ == "__main__":
    all_lookups = {}
    for city, info in CITIES.items():
        print(f"📍 Fetching historical weather for {city}...")
        lat, lon = info["weather_coords"]  # tuple stored directly in config.py
        all_lookups[city] = fetch_city_history(city, lat, lon)
        print()

    save_master_lookup(all_lookups)
    print("\n✅ Weather fetcher complete (historical, 4-week window)!")