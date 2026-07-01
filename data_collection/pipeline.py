# =============================================================================
# pipeline.py — Master Orchestrator: Build Final cab_dataset.csv
#
# Ties together all data collection modules:
#   - Routes (ORS distance + base_time)
#   - Weather (timestamp → weather_condition)
#   - Holidays (timestamp → is_holiday)
#   - Traffic (subregion + timestamp + weather → traffic_level)
#   - Fare formula (6 noise layers)
#
# For each route, samples N random timestamps from city's weather forecast,
# derives all features, calculates time + fare, and saves clean CSV.
#
# Usage:
#   python data_collection/pipeline.py
# =============================================================================

import os
import json
import random
import csv
from datetime import datetime
from typing import List, Dict, Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import (
    CITIES, ZONES, CAB_TYPES, DATASET_CONFIG, get_time_of_day
)

# Data collection modules
from data_collection.holiday_fetcher import load_holidays, is_holiday
from data_collection.weather_fetcher import load_weather_lookup, get_weather
from data_collection.traffic_fetcher import load_traffic_baseline, get_traffic_level
from data_collection.route_fetcher import load_routes
from data_collection.fare_calculator import (
    calculate_time, calculate_fare, calculate_surge, get_cab_availability
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
OUTPUT_CSV = os.path.join(PROCESSED_DIR, 'cab_dataset.csv')
PROGRESS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'pipeline_progress.json')

TIMESTAMPS_PER_ROUTE = 5  # Variations per route
RANDOM_SEED = 42

# =============================================================================
# RANDOM SEED for reproducibility
# =============================================================================

random.seed(RANDOM_SEED)

# =============================================================================
# LOAD ALL RAW DATA (once, at startup)
# =============================================================================

def load_all_data() -> Dict[str, Any]:
    """
    Loads all raw data files into memory.
    Returns dict with all lookups for fast access during row generation.
    """
    print("📂 Loading raw data files...")

    data = {
        "holidays": load_holidays(),
        "weather": load_weather_lookup(),
        "traffic": load_traffic_baseline(),
        "routes": load_routes(),
    }

    print(f"  ✅ Holidays: {len(data['holidays'])} dates")
    print(f"  ✅ Weather: {sum(len(v) for v in data['weather'].values())} timestamps")
    print(f"  ✅ Traffic: {sum(len(v) for v in data['traffic'].values())} subregions")
    print(f"  ✅ Routes: {len(data['routes'])} routes")

    return data


# =============================================================================
# GET ALL TIMESTAMPS FOR A CITY
# =============================================================================

def get_city_timestamps(city: str, weather_lookup: dict) -> List[str]:
    """Returns all available timestamps from weather forecast for a city."""
    return list(weather_lookup.get(city, {}).keys())


# =============================================================================
# GENERATE ONE ROW
# =============================================================================

def generate_row(
    route: dict,
    timestamp: str,
    data: dict
) -> dict:
    """
    Generates one complete dataset row from a route + timestamp.

    Args:
        route: Route dict with distance_km, base_time_min, source_subregion, etc.
        timestamp: Raw timestamp string AS STORED in weather_lookup keys —
                   this is Open-Meteo's format: "YYYY-MM-DDTHH:00" (ISO, no seconds).
                   Do NOT assume "YYYY-MM-DD HH:MM:SS" here — that mismatch
                   was the original bug. We parse with fromisoformat() instead
                   of a hardcoded strptime pattern, since it handles Open-Meteo's
                   format natively.
        data: Master data dict with holidays, weather, traffic, routes

    Returns:
        Complete row dict with all features + targets
    """
    city = route["city"]
    source_subregion = route["source_subregion"]
    source_zone = route["source_zone"]

    # Parse timestamp robustly — handles Open-Meteo's "YYYY-MM-DDTHH:00" format
    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour
    day_of_week = dt.strftime("%A")

    # Time of day
    time_of_day = get_time_of_day(hour)

    # Weather — lookup MUST use the original raw timestamp string,
    # since that's the exact key stored in weather_lookup
    weather = get_weather(city, timestamp, data["weather"])

    # Holiday — first 10 chars (YYYY-MM-DD) work regardless of T/space separator
    holiday_flag = is_holiday(timestamp[:10], data["holidays"])

    # Traffic level — get_traffic_level() internally parses with
    # "%Y-%m-%d %H:%M:%S", so we must reformat the timestamp for this call
    # specifically (this is the fix for the original crash)
    traffic_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    traffic_level = get_traffic_level(
        city, source_subregion, traffic_timestamp, weather, data["traffic"]
    )

    # Cab availability (from zone)
    cab_availability = get_cab_availability(city, source_zone)

    # Surge multiplier
    surge = calculate_surge(
        city, source_zone, traffic_level, weather,
        holiday_flag, time_of_day, cab_availability
    )

    # Random cab type
    cab_type = random.choice(list(CAB_TYPES.keys()))

    # Calculate estimated time (Model 1 target)
    # NOTE: calculate_time() now only takes base_time_min + traffic_level —
    # weather/time_of_day are already baked into traffic_level upstream
    est_time = calculate_time(
        route["base_time_min"],
        traffic_level
    )

    # Calculate fare (Model 2 target)
    fare = calculate_fare(
        route["distance_km"],
        est_time,
        cab_type,
        surge,
        city,
        source_zone,
        time_of_day
    )

    # Build complete row
    row = {
        # Identifiers
        "route_id": route["route_id"],
        "city": city,
        "city_tier": CITIES[city]["tier"],

        # Location
        "source_lat": route["source_lat"],
        "source_lng": route["source_lng"],
        "dest_lat": route["dest_lat"],
        "dest_lng": route["dest_lng"],
        "source_subregion": source_subregion,
        "source_zone": source_zone,
        "dest_subregion": route["dest_subregion"],
        "dest_zone": route["dest_zone"],

        # Route
        "distance_km": route["distance_km"],
        "base_time_min": route["base_time_min"],

        # Time
        "timestamp": timestamp,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "hour": hour,
        "is_holiday": int(holiday_flag),

        # Weather
        "weather_condition": weather,

        # Traffic
        "traffic_level": traffic_level,

        # Cab
        "cab_type": cab_type,
        "cab_availability": cab_availability,

        # Pricing
        "surge_multiplier": surge,
        "base_price_per_km": CITIES[city]["base_price_per_km"],
        "per_minute_rate": CITIES[city]["per_minute_rate"],

        # Targets
        "estimated_time_min": est_time,
        "fare_amount": fare,
    }

    return row


# =============================================================================
# SAVE PROGRESS
# =============================================================================

def save_progress(completed_routes: int, total_rows: int) -> None:
    """Saves pipeline progress for resume capability."""
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "completed_routes": completed_routes,
            "total_rows": total_rows,
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)


def load_progress() -> tuple[int, int]:
    """Loads previous progress. Returns (completed_routes, total_rows)."""
    if not os.path.exists(PROGRESS_PATH):
        return 0, 0

    with open(PROGRESS_PATH, "r") as f:
        data = json.load(f)

    print(f"  📂 Resuming from route {data['completed_routes']}")
    return data["completed_routes"], data["total_rows"]


def clear_progress() -> None:
    """Removes progress file after successful completion."""
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(rows: List[dict], filepath: str) -> None:
    """Saves rows to CSV file."""
    if not rows:
        print("⚠️  No rows to save!")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  💾 Saved {len(rows)} rows → {filepath}")


def append_csv(rows: List[dict], filepath: str) -> None:
    """Appends rows to existing CSV file."""
    if not rows:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    fieldnames = list(rows[0].keys())

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PIPELINE — Building cab_dataset.csv")
    print("=" * 60)

    # Load all raw data
    data = load_all_data()
    routes = data["routes"]

    # Check for resume
    completed_routes, existing_rows = load_progress()

    if completed_routes > 0:
        print(f"\n🔄 Resuming from route {completed_routes} of {len(routes)}")
    else:
        print(f"\n🚀 Starting fresh — {len(routes)} routes × {TIMESTAMPS_PER_ROUTE} timestamps = ~{len(routes) * TIMESTAMPS_PER_ROUTE} rows")

    # Prepare CSV (create fresh if not resuming)
    if completed_routes == 0 and os.path.exists(OUTPUT_CSV):
        print(f"\n🗑️  Removing old CSV: {OUTPUT_CSV}")
        os.remove(OUTPUT_CSV)

    all_rows = []
    total_rows = existing_rows

    # Process routes
    for i in range(completed_routes, len(routes)):
        route = routes[i]
        city = route["city"]

        # Get available timestamps for this city
        city_timestamps = get_city_timestamps(city, data["weather"])

        if not city_timestamps:
            print(f"  ⚠️  No weather data for {city}, skipping route {i}")
            continue

        # Sample N random timestamps
        sampled_timestamps = random.sample(
            city_timestamps,
            min(TIMESTAMPS_PER_ROUTE, len(city_timestamps))
        )

        # Generate rows for each timestamp
        route_rows = []
        for timestamp in sampled_timestamps:
            try:
                row = generate_row(route, timestamp, data)
                route_rows.append(row)
            except Exception as e:
                print(f"  ❌ Error generating row for route {i}, timestamp {timestamp}: {e}")
                continue

        # Append to CSV immediately (batch of route rows)
        if route_rows:
            append_csv(route_rows, OUTPUT_CSV)
            total_rows += len(route_rows)
            all_rows.extend(route_rows)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  ✅ {i + 1}/{len(routes)} routes processed ({total_rows} rows so far)")
            save_progress(i + 1, total_rows)

    # Final save
    save_progress(len(routes), total_rows)

    print(f"\n📊 Pipeline Summary:")
    print(f"  Total routes processed: {len(routes)}")
    print(f"  Total rows generated: {total_rows}")
    print(f"  Variations per route: {TIMESTAMPS_PER_ROUTE}")
    print(f"  Output: {OUTPUT_CSV}")

    # Clear progress on success
    clear_progress()
    print(f"\n🧹 Progress file cleared")

    # Sanity check
    print(f"\n📅 Sanity Check — Loading CSV:")
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            sample_rows = list(reader)[:3]

        print(f"  Total rows in CSV: {total_rows}")
        print(f"  Columns: {len(sample_rows[0]) if sample_rows else 0}")

        if sample_rows:
            print(f"\n  Sample row:")
            for k, v in list(sample_rows[0].items())[:10]:
                print(f"    {k}: {v}")

    print(f"\n✅ Pipeline complete!")
    print(f"   Next step: EDA → eda/eda.ipynb")


if __name__ == "__main__":
    main()
