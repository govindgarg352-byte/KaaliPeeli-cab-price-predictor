# =============================================================================
# traffic_fetcher.py — Fetch Sub-Region-Level Traffic Baseline via TomTom
#
# Makes ~76 API calls (one per sub-region) → saves freeFlowSpeed per sub-region
# Each sub-region gets its own structural traffic baseline.
# Nearby places share the same sub-region's baseline.
#
# Usage:
#   python data_collection/traffic_fetcher.py
# =============================================================================

import os
import json
import requests
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
    API_KEYS, ZONES, TRAFFIC_MATRIX, WEEKEND_TRAFFIC_FACTOR,
    WEATHER_TIME_IMPACT, TRAFFIC_LEVEL_THRESHOLDS, TIME_SLOT_HOURS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
RAW_DIR         = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'tomtom_traffic')
BASELINE_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'traffic_baseline.json')

# =============================================================================
# HELPER: Derive time_of_day from hour
# =============================================================================

def get_time_of_day(hour: int) -> str:
    """Returns time slot category from hour (0-23)."""
    if 6 <= hour < 10:
        return "Morning"
    elif 10 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


# =============================================================================
# FETCH TRAFFIC FLOW FOR ONE SUB-REGION
# =============================================================================

def fetch_subregion_traffic(city: str, subregion: str, lat: float, lon: float) -> dict:
    """
    Calls TomTom Flow Segment Data API for a specific sub-region coordinate.
    Returns raw API response with freeFlowSpeed.

    TomTom endpoint:
        https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json

    Parameters:
        point = "lat,lon" (comma-separated, WGS84)
        key   = API key
        unit  = kmph (kilometers per hour)

    Response (JSON):
        {
          "flowSegmentData": {
            "freeFlowSpeed": 70,
            "currentSpeed": 41,
            "confidence": 0.59,
            ...
          }
        }
    """
    params = {
        "point": f"{lat},{lon}",
        "key":   API_KEYS["tomtom"],
        "unit":  "kmph",
    }

    response = requests.get(TOMTOM_FLOW_URL, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(
            f"TomTom API error for {city}/{subregion}: {response.status_code} — {response.text}"
        )

    return response.json()


# =============================================================================
# PARSE RESPONSE → EXTRACT freeFlowSpeed
# =============================================================================

def parse_subregion_traffic(city: str, subregion: str, zone: str, lat: float, lon: float, raw: dict) -> dict:
    """
    Extracts freeFlowSpeed from TomTom response for a sub-region.

    We ONLY use freeFlowSpeed (structural baseline), not currentSpeed.
    freeFlowSpeed tells us how fast traffic moves under ideal conditions
    for this specific sub-region — CSMT (dense city center) naturally slower
    than Borivali (outer suburb).

    Returns:
        {
            "free_flow_speed": float,     # km/h under ideal conditions
            "zone": str,                   # parent zone (Airport/Railway/City Center/Suburb)
            "query_lat": float,            # coordinate used for query
            "query_lon": float,
            "city": str,
            "subregion": str
        }
    """
    flow_data = raw.get("flowSegmentData", {})

    free_flow_speed = flow_data.get("freeFlowSpeed")
    if free_flow_speed is None:
        raise KeyError(f"freeFlowSpeed missing in TomTom response for {city}/{subregion}")

    result = {
        "free_flow_speed": float(free_flow_speed),
        "zone": zone,
        "query_lat": lat,
        "query_lon": lon,
        "city": city,
        "subregion": subregion,
    }

    print(f"  {city}/{subregion} ({zone}): freeFlowSpeed = {free_flow_speed} km/h")
    return result


# =============================================================================
# SAVE PER-CITY RAW + PARSED
# =============================================================================

def save_city_traffic(city: str, subregion_data: dict) -> None:
    """Saves raw traffic data for all sub-regions in a city."""
    os.makedirs(RAW_DIR, exist_ok=True)

    raw_path = os.path.join(RAW_DIR, f"{city.lower()}_traffic.json")
    with open(raw_path, "w") as f:
        json.dump({
            "city": city,
            "fetched_at": datetime.now().isoformat(),
            "subregions": subregion_data,
        }, f, indent=2)


# =============================================================================
# SAVE MASTER BASELINE
# =============================================================================

def save_master_baseline(master: dict) -> None:
    """
    Saves master traffic baseline with structure:
    {city: {subregion: {free_flow_speed, zone, query_lat, query_lon}}}

    This is what pipeline.py and get_traffic_level() import.
    """
    output = {
        "fetched_at": datetime.now().isoformat(),
        "cities": master,
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Master baseline saved → {BASELINE_PATH}")


# =============================================================================
# LOAD BASELINE (used by pipeline.py)
# =============================================================================

def load_traffic_baseline() -> dict:
    """
    Loads master traffic baseline.

    Returns:
        {
            "Mumbai": {
                "CSMT": {"free_flow_speed": 25, "zone": "Railway", ...},
                "Dadar Station": {"free_flow_speed": 28, "zone": "Railway", ...},
                ...
            },
            ...
        }

    Usage in pipeline.py:
        from data_collection.traffic_fetcher import load_traffic_baseline, get_traffic_level
        BASELINE = load_traffic_baseline()
        level = get_traffic_level("Mumbai", "CSMT", "2026-06-29 17:00:00", "Rainy", BASELINE)
    """
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"Traffic baseline not found at {BASELINE_PATH}\n"
            f"Run: python data_collection/traffic_fetcher.py"
        )

    with open(BASELINE_PATH, "r") as f:
        data = json.load(f)

    return data["cities"]


# =============================================================================
# GET TRAFFIC LEVEL — Apply time + day + weather adjustments
# =============================================================================

def get_traffic_level(city, subregion, timestamp, weather, baseline=None):
    if baseline is None:
        baseline = load_traffic_baseline()

    city_baseline = baseline.get(city, {})
    subregion_data = city_baseline.get(subregion, {})
    free_flow_speed = subregion_data.get("free_flow_speed")

    if free_flow_speed is None:
        raise KeyError(f"No baseline data for {city}/{subregion}. Run traffic_fetcher.py first.")

    # NEW: normalize this subregion's speed against the city's average freeFlowSpeed
    city_avg_speed = sum(d["free_flow_speed"] for d in city_baseline.values()) / len(city_baseline)
    relative_speed_factor = free_flow_speed / city_avg_speed  # >1 = naturally faster road, <1 = naturally slower

    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    weekday = dt.weekday()
    time_of_day = get_time_of_day(hour)

    time_factor = TRAFFIC_MATRIX[city][time_of_day]
    weekend_factor = WEEKEND_TRAFFIC_FACTOR if weekday >= 5 else 1.0
    weather_factor = WEATHER_TIME_IMPACT.get(weather, 1.0)

    if weather == "Stormy":
        return "High"

    # subregion's structural speed now actually influences the result
    congestion_ratio = relative_speed_factor / (time_factor * weekend_factor * weather_factor)

    for level, (low, high) in TRAFFIC_LEVEL_THRESHOLDS.items():
        if low <= congestion_ratio < high:
            return level

    return "Medium"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TRAFFIC FETCHER — Sub-Region-Level Baseline (~76 calls)")
    print("=" * 60)

    master_baseline = {}
    total_calls = 0

    for city in ZONES:
        print(f"\n📍 Fetching traffic baseline for {city}...")
        city_subregions = {}

        for zone in ZONES[city]:
            sub_regions = ZONES[city][zone]["sub_regions"]

            for subregion, (lat, lon) in sub_regions.items():
                try:
                    # Fetch from TomTom for this specific sub-region
                    raw = fetch_subregion_traffic(city, subregion, lat, lon)

                    # Parse — extract only freeFlowSpeed
                    parsed = parse_subregion_traffic(city, subregion, zone, lat, lon, raw)

                    # Store in city data
                    city_subregions[subregion] = parsed
                    total_calls += 1

                except Exception as e:
                    print(f"  ❌ Error fetching {city}/{subregion}: {e}")

        # Save per-city raw
        save_city_traffic(city, city_subregions)

        # Add to master
        master_baseline[city] = city_subregions

        print(f"  ✅ {city}: {len(city_subregions)} sub-regions saved")

    # Save master baseline
    save_master_baseline(master_baseline)

    print(f"\n📊 Total TomTom API calls made: {total_calls}")

    # Sanity check — test get_traffic_level with different scenarios
    print("\n📅 Sanity Check — Testing traffic level derivation:")
    baseline = load_traffic_baseline()

    test_cases = [
        # Same city, same time/weather, different sub-regions → different outcomes?
        ("Mumbai", "CSMT", "2026-06-29 08:00:00", "Sunny"),           # Morning weekday
        ("Mumbai", "Borivali Station", "2026-06-29 08:00:00", "Sunny"), # Same time, different subregion
        ("Mumbai", "CSMT", "2026-06-29 18:00:00", "Rainy"),           # Evening rainy
        ("Delhi", "IGI Terminal 3", "2026-06-27 14:00:00", "Sunny"),  # Saturday afternoon
        ("Chandigarh", "Mohali", "2026-06-29 23:00:00", "Foggy"),    # Night foggy
    ]

    for city, subregion, ts, weather in test_cases:
        try:
            level = get_traffic_level(city, subregion, ts, weather, baseline)
            print(f"  {city}/{subregion} @ {ts} ({weather}) → {level}")
        except Exception as e:
            print(f"  ❌ Error testing {city}/{subregion}: {e}")

    print("\n✅ Traffic fetcher complete!")
    print("   Import in other scripts with:")
    print("   from data_collection.traffic_fetcher import load_traffic_baseline, get_traffic_level")
