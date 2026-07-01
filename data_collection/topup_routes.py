# =============================================================================
# topup_routes.py — Fetch ONLY the missing routes to reach 2000 total
#
# Loads existing routes.json (1887 routes), figures out how many each city
# is short of its target (400/city), generates + fetches just those, and
# merges the result back into routes.json. Does NOT touch existing data.
#
# Usage:
#   python data_collection/topup_routes.py
# =============================================================================

import os
import sys
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import CITIES, DATASET_CONFIG
from data_collection.route_fetcher import (
    generate_route_pair,
    fetch_route_distance_time,
    save_master_routes,
    save_city_routes,
    load_routes,
    ORS_DELAY_SECONDS,
)

TOTAL_ROUTES = DATASET_CONFIG["total_rows"]  # 2000
cities = list(CITIES.keys())
per_city_target = TOTAL_ROUTES // len(cities)  # 400

print("=" * 60)
print("  ROUTE TOP-UP — Filling gap to reach 2000 total")
print("=" * 60)

# Load existing routes (your 1887, untouched)
existing_routes = load_routes()
print(f"\n📂 Loaded {len(existing_routes)} existing routes")

# Count existing routes per city
existing_by_city = {city: 0 for city in cities}
for r in existing_routes:
    if r["city"] in existing_by_city:
        existing_by_city[r["city"]] += 1

# Figure out how many each city is short
to_fetch = {}
for city in cities:
    have = existing_by_city[city]
    need = max(0, per_city_target - have)
    to_fetch[city] = need
    print(f"  {city}: have {have}/{per_city_target} → need {need} more")

total_needed = sum(to_fetch.values())
print(f"\n🎯 Total routes to fetch: {total_needed}")

if total_needed == 0:
    print("\n✅ Already at target — nothing to do!")
    sys.exit(0)

# Find the highest existing route_id to continue numbering safely
max_id = max((r.get("route_id", -1) for r in existing_routes), default=-1)
next_id = max_id + 1

# Generate the missing pairs
new_pairs = []
for city in cities:
    for _ in range(to_fetch[city]):
        pair = generate_route_pair(city)
        pair["route_id"] = next_id
        next_id += 1
        new_pairs.append(pair)

print(f"\n🌐 Calling OpenRouteService for {len(new_pairs)} new routes...")
print(f"  Estimated time: ~{len(new_pairs) * ORS_DELAY_SECONDS / 60:.1f} minutes")

success_count = 0
fail_count = 0

for i, route in enumerate(new_pairs):
    try:
        new_pairs[i], headers = fetch_route_distance_time(route)
        success_count += 1

        if (i + 1) % 10 == 0:
            print(f"  ✅ {i + 1}/{len(new_pairs)} new routes completed")

        time.sleep(ORS_DELAY_SECONDS)

    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Quota Exceeded" in error_msg:
            print(f"\n🛑 {error_msg}")
            print(f"    Stopping — keeping {success_count} successfully fetched routes.")
            break

        fail_count += 1
        print(f"  ❌ Route {i} ({route['city']}) failed: {e}")
        new_pairs[i]["distance_km"] = None
        new_pairs[i]["base_time_min"] = None
        time.sleep(ORS_DELAY_SECONDS)

print(f"\n📊 Top-up Summary:")
print(f"  New routes attempted: {len(new_pairs)}")
print(f"  Successful: {success_count}")
print(f"  Failed: {fail_count}")

# Merge: existing + newly successful routes
valid_new = [r for r in new_pairs if r.get("distance_km") is not None]
merged_routes = existing_routes + valid_new

print(f"\n💾 Merging {len(existing_routes)} existing + {len(valid_new)} new = {len(merged_routes)} total")

# Save per-city files
for city in cities:
    city_routes = [r for r in merged_routes if r["city"] == city]
    save_city_routes(city, city_routes)
    print(f"  {city}: {len(city_routes)} routes saved")

# Save master
save_master_routes(merged_routes)

print(f"\n✅ Top-up complete! Total routes now: {len(merged_routes)}")
