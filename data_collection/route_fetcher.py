# =============================================================================
# route_fetcher.py — Generate 2000 Unique Coordinate Pairs + Call OpenRouteService
#
# Generates 2000 random source-destination pairs within same city,
# calls OpenRouteService for real road distance + base travel time.
# Saves per-city route data + master routes.json.
#
# Also exposes helper to derive nearest subregion/zone from coordinates
# for use by pipeline.py.
#
# Usage:
#   python data_collection/route_fetcher.py
# =============================================================================

import os
import json
import random
import time
import requests
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import API_KEYS, CITIES, ZONES, DATASET_CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================

ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'route_data')
ROUTES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'routes.json')
PROGRESS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'route_progress.json')

COORDINATE_OFFSET = DATASET_CONFIG["coordinate_offset"]  # 0.005
TOTAL_ROUTES = DATASET_CONFIG["total_rows"]  # 2000
random.seed(DATASET_CONFIG["random_seed"])  # 42

# Rate limiting: ORS free tier 2000/day, 40/minute
# Sleep 1.6s between calls = ~37 calls/minute (safe under 40/min)
ORS_DELAY_SECONDS = 1.6
MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 60]  # seconds for retry 1, 2, 3

# On 404 (no routable point), retry with smaller offset
RETRY_OFFSETS_404 = [0.003, 0.001, 0.0005]  # progressively smaller jitter

# Safety margin: stop when remaining quota is this low
QUOTA_SAFETY_MARGIN = 50

# =============================================================================
# HELPER: Get all subregions for a city
# =============================================================================

def get_all_subregions(city: str) -> list:
    """
    Returns list of (subregion_name, lat, lng, zone) for a city.

    Usage:
        subregions = get_all_subregions("Mumbai")
        # [("CSIA Terminal 1", 19.0928, 72.8571, "Airport"), ...]
    """
    result = []
    for zone, zone_data in ZONES[city].items():
        for name, (lat, lng) in zone_data["sub_regions"].items():
            result.append((name, lat, lng, zone))
    return result


# =============================================================================
# HELPER: Find nearest subregion from coordinates
# =============================================================================

def find_nearest_subregion(city: str, lat: float, lng: float) -> tuple[str, str]:
    """
    Finds the nearest subregion to given coordinates using Euclidean distance.
    Returns (subregion_name, zone_name).

    This is used by:
      - route_fetcher.py → to tag source/dest subregions for generated routes
      - pipeline.py → to tag pickup_zone for any user-selected coordinate

    Args:
        city: City name e.g. "Mumbai"
        lat:  Latitude
        lng:  Longitude

    Returns:
        (subregion_name, zone_name) e.g. ("CSMT", "Railway")

    Usage in pipeline.py:
        from data_collection.route_fetcher import find_nearest_subregion
        subregion, zone = find_nearest_subregion("Mumbai", 19.093, 72.858)
    """
    subregions = get_all_subregions(city)

    min_dist = float('inf')
    nearest_subregion = None
    nearest_zone = None

    for name, sub_lat, sub_lng, zone in subregions:
        # Euclidean distance (sufficient for small distances)
        dist = ((lat - sub_lat) ** 2 + (lng - sub_lng) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_subregion = name
            nearest_zone = zone

    return nearest_subregion, nearest_zone


# =============================================================================
# HELPER: Get zone for a subregion (wrapper for config helper)
# =============================================================================

def get_zone_for_subregion(city: str, subregion: str) -> str:
    """
    Returns the zone name for a given subregion.
    Wrapper around config's helper for convenience.

    Usage in pipeline.py:
        from data_collection.route_fetcher import get_zone_for_subregion
        zone = get_zone_for_subregion("Mumbai", "CSMT")  # → "Railway"
    """
    for zone, zone_data in ZONES[city].items():
        if subregion in zone_data["sub_regions"]:
            return zone
    return "City Center"


# =============================================================================
# GENERATE RANDOM COORDINATE PAIR
# =============================================================================

def generate_random_coordinates(city: str) -> tuple[float, float]:
    """
    Generates random coordinates within a city's bounding box.
    Used for destination when we want variety beyond sub-regions.
    """
    bbox = CITIES[city]["bounding_box"]
    lat = random.uniform(bbox["lat_min"], bbox["lat_max"])
    lng = random.uniform(bbox["lng_min"], bbox["lng_max"])
    return lat, lng


def generate_route_pair(city: str, offset: float = None) -> dict:
    """
    Generates one source-destination pair for a city.

    Strategy:
      1. Pick random source subregion from city
      2. Add ±offset jitter for variety (default: COORDINATE_OFFSET)
      3. Pick random destination subregion from SAME city
      4. Add ±offset jitter for variety
      5. Return coordinates + subregion metadata

    Args:
        city: City name
        offset: Coordinate jitter in degrees (default: COORDINATE_OFFSET from config)

    Returns:
        {
            "city": city,
            "source_lat": float,
            "source_lng": float,
            "dest_lat": float,
            "dest_lng": float,
            "source_subregion": str,
            "source_zone": str,
            "dest_subregion": str,
            "dest_zone": str,
        }
    """
    if offset is None:
        offset = COORDINATE_OFFSET

    subregions = get_all_subregions(city)

    # Pick source
    src_name, src_lat, src_lng, src_zone = random.choice(subregions)
    source_lat = src_lat + random.uniform(-offset, offset)
    source_lng = src_lng + random.uniform(-offset, offset)

    # Pick destination (same city, can be same or different subregion)
    dest_name, dest_lat, dest_lng, dest_zone = random.choice(subregions)
    dest_lat = dest_lat + random.uniform(-offset, offset)
    dest_lng = dest_lng + random.uniform(-offset, offset)

    return {
        "city": city,
        "source_lat": round(source_lat, 6),
        "source_lng": round(source_lng, 6),
        "dest_lat": round(dest_lat, 6),
        "dest_lng": round(dest_lng, 6),
        "source_subregion": src_name,
        "source_zone": src_zone,
        "dest_subregion": dest_name,
        "dest_zone": dest_zone,
    }


# =============================================================================
# CALL OPENROUTESERVICE FOR ONE ROUTE (with retry + rate limit + 404 handling)
# =============================================================================

def fetch_route_distance_time(route: dict, attempt_offset: float = None) -> tuple[dict, dict]:
    """
    Calls OpenRouteService Directions API for a source-destination pair.
    Returns (updated_route, headers) with distance (km) and base_time (minutes).

    Handles:
      - Rate limiting (429) with exponential backoff retries
      - No routable point (404) by retrying with smaller coordinate offset
      - Timeouts with retry
      - Quota tracking via response headers

    OpenRouteService endpoint:
        https://api.openrouteservice.org/v2/directions/driving-car

    Request body (JSON):
        {
            "coordinates": [
                [source_lng, source_lat],
                [dest_lng, dest_lat]
            ]
        }

    Response:
        {
            "routes": [{
                "summary": {
                    "distance": 12450.5,   # meters
                    "duration": 1852.3     # seconds
                }
            }]
        }

    Args:
        route: Dict with source_lat, source_lng, dest_lat, dest_lng
        attempt_offset: Override coordinate offset for this attempt (for 404 retries)

    Returns:
        (updated_route_dict, response_headers_dict)
    """
    headers = {
        "Authorization": API_KEYS["openrouteservice"],
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": [
            [route["source_lng"], route["source_lat"]],
            [route["dest_lng"], route["dest_lat"]]
        ]
    }

    response_headers = {}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(ORS_DIRECTIONS_URL, headers=headers, json=body, timeout=15)

            # Capture rate limit headers
            response_headers = dict(response.headers)

            # Rate limit hit (minutely) — wait and retry
            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF[attempt]
                    print(f"    ⏳ Rate limit (429) — waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("ORS API error: 429 — Rate Limit Exceeded (max retries reached)")

            # Quota exceeded (daily) — fatal, stop immediately
            if response.status_code == 403:
                raise Exception("ORS API error: 403 — Daily Quota Exceeded. Resume tomorrow.")

            # No routable point nearby — regenerate coordinates with smaller offset
            if response.status_code == 404:
                if attempt < len(RETRY_OFFSETS_404):
                    new_offset = RETRY_OFFSETS_404[attempt]
                    print(f"    🔄 No road found (404) — retrying with smaller offset {new_offset}...")

                    # Regenerate route with smaller offset
                    city = route["city"]
                    new_route = generate_route_pair(city, offset=new_offset)
                    new_route["route_id"] = route.get("route_id")

                    # Update body with new coordinates
                    body["coordinates"] = [
                        [new_route["source_lng"], new_route["source_lat"]],
                        [new_route["dest_lng"], new_route["dest_lat"]]
                    ]

                    # Copy metadata from new route
                    route.update(new_route)

                    # Small delay before retry
                    time.sleep(1)
                    continue
                else:
                    raise Exception("ORS API error: 404 — No routable point found (max offset retries reached)")

            # Other error
            if response.status_code != 200:
                raise Exception(
                    f"ORS API error: {response.status_code} — {response.text}"
                )

            # Success
            data = response.json()

            # Extract from first route's summary
            route_summary = data["routes"][0]["summary"]
            distance_m = route_summary["distance"]
            duration_sec = route_summary["duration"]

            route["distance_km"] = round(distance_m / 1000, 2)
            route["base_time_min"] = round(duration_sec / 60, 2)

            return route, response_headers

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF[attempt]
                print(f"    ⏳ Timeout — waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait_time)
            else:
                raise Exception("ORS API timeout (max retries reached)")

    return route, response_headers


# =============================================================================
# SAVE PROGRESS
# =============================================================================

def save_progress(routes: list, completed: int) -> None:
    """Saves intermediate progress every 100 routes for resume capability."""
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "completed": completed,
            "routes": routes,
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)


def load_progress() -> tuple[int, list]:
    """Loads previous progress if exists. Returns (completed_count, routes)."""
    if not os.path.exists(PROGRESS_PATH):
        return 0, []

    with open(PROGRESS_PATH, "r") as f:
        data = json.load(f)

    print(f"  📂 Resuming from {data['completed']} previously completed routes")
    return data["completed"], data["routes"]


def clear_progress() -> None:
    """Removes progress file after successful completion."""
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


# =============================================================================
# SAVE FINAL ROUTES
# =============================================================================

def save_city_routes(city: str, routes: list) -> None:
    """Saves routes for one city to raw directory."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{city.lower()}_routes.json")
    with open(path, "w") as f:
        json.dump({
            "city": city,
            "count": len(routes),
            "fetched_at": datetime.now().isoformat(),
            "routes": routes,
        }, f, indent=2)


def save_master_routes(all_routes: list) -> None:
    """Saves combined master routes file."""
    output = {
        "fetched_at": datetime.now().isoformat(),
        "total": len(all_routes),
        "routes": all_routes,
    }
    with open(ROUTES_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Master routes saved → {ROUTES_PATH}")


# =============================================================================
# LOAD ROUTES (used by pipeline.py)
# =============================================================================

def load_routes() -> list[dict]:
    """
    Loads all routes from master file.

    Returns list of route dicts with keys:
        route_id, city, source_lat, source_lng, dest_lat, dest_lng,
        distance_km, base_time_min, source_subregion, source_zone,
        dest_subregion, dest_zone

    Usage in pipeline.py:
        from data_collection.route_fetcher import load_routes
        ROUTES = load_routes()
        for route in ROUTES:
            ...
    """
    if not os.path.exists(ROUTES_PATH):
        raise FileNotFoundError(
            f"Routes file not found at {ROUTES_PATH}\n"
            f"Run: python data_collection/route_fetcher.py"
        )

    with open(ROUTES_PATH, "r") as f:
        data = json.load(f)

    return data["routes"]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ROUTE FETCHER — 2000 Unique Coordinate Pairs via ORS")
    print("=" * 60)
    print(f"  Rate limit: {ORS_DELAY_SECONDS}s delay between calls (~37 calls/min)")
    print(f"  ORS free tier: 2000 directions/day, 40/minute")
    print(f"  Estimated time: ~{TOTAL_ROUTES * ORS_DELAY_SECONDS / 60:.0f} minutes")
    print(f"  Quota safety margin: stop when {QUOTA_SAFETY_MARGIN} calls remaining")

    # Check for resume
    completed, routes = load_progress()

    if completed > 0:
        print(f"\n🔄 Resuming from route {completed} of {TOTAL_ROUTES}")
    else:
        print(f"\n🚀 Starting fresh — generating {TOTAL_ROUTES} routes")

    # Generate all route pairs first (cheap, no API call)
    if completed == 0:
        print("\n📍 Generating coordinate pairs...")
        all_pairs = []

        # Distribute evenly across cities (400 per city)
        cities = list(CITIES.keys())
        per_city = TOTAL_ROUTES // len(cities)

        for city in cities:
            for _ in range(per_city):
                pair = generate_route_pair(city)
                pair["route_id"] = len(all_pairs)
                all_pairs.append(pair)

        print(f"  Generated {len(all_pairs)} coordinate pairs")
        routes = all_pairs

    # Call ORS for remaining routes
    print(f"\n🌐 Calling OpenRouteService (starting from route {completed})...")

    success_count = sum(1 for r in routes[:completed] if r.get("distance_km") is not None)
    fail_count = 0
    quota_exceeded = False

    for i in range(completed, len(routes)):
        route = routes[i]

        # Check if we're running low on quota (from previous response headers)
        if i > completed and 'x-ratelimit-remaining' in locals():
            remaining = int(locals().get('x-ratelimit-remaining', 2000))
            if remaining < QUOTA_SAFETY_MARGIN:
                print(f"\n⚠️  Quota running low ({remaining} remaining). Stopping at route {i}.")
                print(f"    Resume tomorrow by re-running this script.")
                quota_exceeded = True
                break

        try:
            routes[i], headers = fetch_route_distance_time(route)
            success_count += 1

            # Track quota from headers
            if 'x-ratelimit-remaining' in headers:
                remaining = int(headers['x-ratelimit-remaining'])
                if remaining % 100 == 0:  # Print every 100
                    print(f"    📊 ORS quota remaining: {remaining}")

            # Progress every 10 routes
            if (i + 1) % 10 == 0:
                print(f"  ✅ {i + 1}/{len(routes)} routes completed")

            # Save progress every 100 routes
            if (i + 1) % 100 == 0:
                save_progress(routes, i + 1)
                print(f"  💾 Progress saved at route {i + 1}")

            # Rate limit delay between ALL calls
            time.sleep(ORS_DELAY_SECONDS)

        except Exception as e:
            error_msg = str(e)

            # Quota exceeded — save progress and stop
            if "403" in error_msg or "Quota Exceeded" in error_msg:
                print(f"\n🛑 {error_msg}")
                print(f"    Saving progress at route {i}...")
                save_progress(routes, i)
                quota_exceeded = True
                break

            # Other error
            fail_count += 1
            print(f"  ❌ Route {i} failed: {e}")
            # Mark with nulls so we can retry later
            routes[i]["distance_km"] = None
            routes[i]["base_time_min"] = None

            # Save progress even on failure
            if (i + 1) % 100 == 0:
                save_progress(routes, i + 1)

            # Still delay after failure
            time.sleep(ORS_DELAY_SECONDS)

    # Final save
    final_completed = i if quota_exceeded else len(routes)
    save_progress(routes, final_completed)

    print(f"\n📊 Summary:")
    print(f"  Total routes: {len(routes)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Completed: {final_completed}")
    if quota_exceeded:
        print(f"  ⚠️  Stopped early due to quota. Resume tomorrow!")

    # Save per-city
    print(f"\n💾 Saving per-city route files...")
    for city in CITIES:
        city_routes = [r for r in routes if r["city"] == city and r.get("distance_km") is not None]
        save_city_routes(city, city_routes)
        print(f"  {city}: {len(city_routes)} routes saved")

    # Save master
    valid_routes = [r for r in routes if r.get("distance_km") is not None]
    save_master_routes(valid_routes)

    # Clear progress on full success
    if not quota_exceeded and fail_count == 0:
        clear_progress()
        print(f"\n🧹 Progress file cleared (all routes successful)")

    # Sanity check
    print(f"\n📅 Sanity Check — Loading back from master:")
    loaded = load_routes()
    print(f"  Total routes loaded: {len(loaded)}")

    if loaded:
        sample = loaded[0]
        print(f"  Sample route:")
        print(f"    {sample['city']}: {sample['source_subregion']} → {sample['dest_subregion']}")
        print(f"    Distance: {sample['distance_km']} km, Base time: {sample['base_time_min']} min")

    print(f"\n✅ Route fetcher complete!")
    print(f"   Import in other scripts with:")
    print(f"   from data_collection.route_fetcher import load_routes, find_nearest_subregion, get_zone_for_subregion")
