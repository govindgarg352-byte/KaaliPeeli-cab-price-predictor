# =============================================================================
# fare_calculator.py — Apply Realistic Fare Formula + Noise Layers
#
# Takes raw route data (distance, base_time) and derives:
#   - estimated_time_min (Model 1 target)
#   - fare_amount (Model 2 target)
#
# Uses independent noise layers to simulate real-world fare variation.
#
# Usage:
#   from data_collection.fare_calculator import calculate_fare, calculate_time
# =============================================================================

import os
import json
import random
import numpy as np
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import (
    CITIES, ZONES, CAB_TYPES, NOISE_CONFIG,
    SURGE_CONFIG, TRAFFIC_MATRIX, WEATHER_TIME_IMPACT,
    CITY_TOLL_CONFIG, AVAILABILITY_SURGE_MAP,
    get_zone_for_subregion, get_time_of_day, get_availability_surge
)

# =============================================================================
# RANDOM SEED for reproducibility
# =============================================================================

random.seed(42)
np.random.seed(42)

# =============================================================================
# TRAFFIC LEVEL → TIME MULTIPLIER (inline, not in config)
# =============================================================================

TRAFFIC_TIME_MULTIPLIER = {
    "Low":    1.0,
    "Medium": 1.25,
    "High":   1.55,
}


# =============================================================================
# MODEL 1: Calculate estimated_time_min
# =============================================================================

def calculate_time(
    base_time_min: float,
    traffic_level: str
) -> float:
    """
    Calculates estimated ride time with noise layers.

    NOTE: traffic_level (Low/Medium/High) already encodes the combined
    structural effect of time_of_day, weekend, and weather — it's derived
    upstream in traffic_fetcher.get_traffic_level() using TRAFFIC_MATRIX,
    WEEKEND_TRAFFIC_FACTOR, and WEATHER_TIME_IMPACT against the sub-region's
    TomTom baseline. So we do NOT re-apply weather/time-of-day multipliers
    here — doing so would double-count the same real-world effect twice
    and distort estimated_time_min, especially in bad-condition scenarios.

    Formula:
        estimated_time = base_time
                       × traffic_multiplier
                       + signal_wait
                       + pickup_wait
                       (with route_deviation noise applied as a final multiplier)

    Args:
        base_time_min: Base travel time from ORS (no traffic)
        traffic_level: "Low" / "Medium" / "High" (already encodes time/weather)

    Returns:
        Estimated time in minutes (float)
    """
    # Traffic multiplier — sole driver of congestion-based time inflation
    traffic_mult = TRAFFIC_TIME_MULTIPLIER.get(traffic_level, 1.0)

    # Base time with traffic applied
    adjusted_time = base_time_min * traffic_mult

    # Noise: Signal wait (traffic lights, junctions)
    signal_wait = random.uniform(
        NOISE_CONFIG["signal_wait_min"]["min"],
        NOISE_CONFIG["signal_wait_min"]["max"]
    )

    # Noise: Route deviation (driver takes non-optimal route)
    route_deviation = np.random.normal(
        NOISE_CONFIG["route_deviation"]["mean"],
        NOISE_CONFIG["route_deviation"]["std"]
    )
    # Apply deviation as multiplier (e.g., +6% = 1.06x)
    route_deviation = 1.0 + route_deviation
    route_deviation = max(0.85, min(1.15, route_deviation))

    # Noise: Pickup wait (driver reaching pickup point)
    pickup_wait = random.uniform(
        NOISE_CONFIG["pickup_wait_min"]["min"],
        NOISE_CONFIG["pickup_wait_min"]["max"]
    )

    # Final estimated time
    estimated_time = (adjusted_time * route_deviation) + signal_wait + pickup_wait

    return round(max(estimated_time, 3.0), 2)  # Minimum 3 minutes


# =============================================================================
# MODEL 2: Calculate fare_amount
# =============================================================================

def calculate_fare(
    distance_km: float,
    estimated_time_min: float,
    cab_type: str,
    surge_multiplier: float,
    city: str,
    pickup_zone: str,
    time_of_day: str
) -> float:
    """
    Calculates ride fare with noise layers.

    Formula:
        fare = (
            (base_price_per_km × distance_km × route_efficiency)
          + (per_minute_rate × (estimated_time_min + idle_time))
          + toll_charge
          + zone_premium
          + night_premium
        )
        × surge_multiplier
        × micro_surge
        × cab_type_multiplier
        × discount

    Args:
        distance_km: Road distance from ORS
        estimated_time_min: Output from calculate_time()
        cab_type: "Mini" / "Sedan" / "SUV"
        surge_multiplier: Derived from availability + traffic + weather + holiday
        city: City name
        pickup_zone: "Airport" / "City Center" / "Railway" / "Suburb"
        time_of_day: "Morning" / "Afternoon" / "Evening" / "Night"

    Returns:
        Fare amount in ₹ (float, rounded to 2 decimals)
    """
    city_config = CITIES[city]
    base_price_per_km = city_config["base_price_per_km"]
    per_minute_rate = city_config["per_minute_rate"]

    # Cab type multiplier
    cab_mult = CAB_TYPES[cab_type]["fare_multiplier"]

    # Zone premium
    zone_premium = ZONES[city][pickup_zone].get("zone_premium", 0)

    # Noise: Route efficiency (driver doesn't always take optimal route)
    route_efficiency = np.random.normal(
        NOISE_CONFIG["route_efficiency"]["mean"],
        NOISE_CONFIG["route_efficiency"]["std"]
    )
    route_efficiency = max(0.85, min(1.15, route_efficiency))

    # Noise: Idle time (waiting at signals, slow zones)
    idle_time = random.uniform(
        NOISE_CONFIG["idle_time_min"]["min"],
        NOISE_CONFIG["idle_time_min"]["max"]
    )

    # Noise: Toll charge (some routes have tolls)
    toll_config = NOISE_CONFIG["toll"]
    city_toll = CITY_TOLL_CONFIG[city]
    if random.random() < toll_config["probability"]:
        toll = random.uniform(city_toll["min"], city_toll["max"])
    else:
        toll = 0

    # Noise: Micro-surge (small random platform-level demand spike)
    micro_surge = np.random.normal(
        NOISE_CONFIG["micro_surge"]["mean"],
        NOISE_CONFIG["micro_surge"]["std"]
    )
    micro_surge = max(0.95, min(1.10, micro_surge))

    # Noise: Discount / promo (random platform discounts)
    discount_config = NOISE_CONFIG["discount"]
    if random.random() < discount_config["probability"]:
        discount = random.uniform(discount_config["min_discount"], discount_config["max_discount"])
    else:
        discount = 1.0

    # Noise: Night safety premium
    night_premium = 0
    if time_of_day == "Night":
        night_config = NOISE_CONFIG["night_premium"]
        night_premium = random.uniform(night_config["min"], night_config["max"])

    # Base fare components
    distance_component = base_price_per_km * distance_km * route_efficiency
    time_component = per_minute_rate * (estimated_time_min + idle_time)

    # Subtotal before multipliers
    subtotal = distance_component + time_component + toll + zone_premium + night_premium

    # Apply multipliers
    fare = subtotal * surge_multiplier * micro_surge * cab_mult * discount

    # Ensure minimum fare
    min_fare = city_config["minimum_fare"]
    fare = max(fare, min_fare)

    # Apply surge cap (MoRTH guidelines: max 2.0x)
    # NOTE: calculate_surge() already caps surge before it reaches here, so this
    # branch is a defensive safety net for direct/manual calls, not normally triggered.
    max_surge = SURGE_CONFIG[city]["max_surge"]
    if surge_multiplier > max_surge:
        fare = fare * (max_surge / surge_multiplier)

    return round(fare, 2)


# =============================================================================
# DERIVE SURGE MULTIPLIER
# =============================================================================

def calculate_surge(
    city: str,
    pickup_zone: str,
    traffic_level: str,
    weather: str,
    is_holiday: bool,
    time_of_day: str,
    cab_availability: int
) -> float:
    """
    Derives surge multiplier from multiple factors.

    Uses config's AVAILABILITY_SURGE_MAP and SURGE_CONFIG.

    Args:
        city: City name
        pickup_zone: Zone name
        traffic_level: "Low" / "Medium" / "High"
        weather: Weather condition
        is_holiday: True/False
        time_of_day: Time slot
        cab_availability: Numeric availability (from ZONES config)

    Returns:
        Surge multiplier (float, e.g., 1.0 = no surge, 2.0 = max)
    """
    # Base surge from availability (primary driver)
    base_surge = get_availability_surge(cab_availability)

    # Traffic boost
    traffic_boost = {"Low": 1.0, "Medium": 1.1, "High": 1.2}.get(traffic_level, 1.0)

    # Weather boost
    weather_boost = {"Sunny": 1.0, "Cloudy": 1.0, "Rainy": 1.15, "Foggy": 1.1, "Stormy": 1.25}.get(weather, 1.0)

    # Holiday boost
    holiday_boost = 1.2 if is_holiday else 1.0

    # Time boost
    time_boost = {"Morning": 1.1, "Afternoon": 1.0, "Evening": 1.15, "Night": 1.1}.get(time_of_day, 1.0)

    # Combine multiplicatively
    surge = base_surge * traffic_boost * weather_boost * holiday_boost * time_boost

    # Add randomness (±10% variation)
    surge = surge * random.uniform(0.9, 1.1)

    # Cap at city max
    max_surge = SURGE_CONFIG[city]["max_surge"]
    surge = min(surge, max_surge)

    return round(max(surge, 1.0), 2)


# =============================================================================
# GET CAB AVAILABILITY (from ZONES config)
# =============================================================================

def get_cab_availability(city: str, zone: str) -> int:
    """
    Returns numeric cab availability for a city/zone from config.

    Args:
        city: City name
        zone: Zone name (Airport, City Center, Railway, Suburb)

    Returns:
        Integer availability value (used by get_availability_surge)
    """
    return ZONES[city][zone]["cab_availability"]


# =============================================================================
# MAIN: Test the functions
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  FARE CALCULATOR — Testing Formula + Noise Layers")
    print("=" * 60)

    # Test case: Mumbai, City Center, Evening, Rainy, Weekday
    test_cases = [
        {
            "city": "Mumbai",
            "distance_km": 12.5,
            "base_time_min": 28.0,
            "traffic_level": "High",
            "weather": "Rainy",
            "time_of_day": "Evening",
            "is_holiday": False,
            "pickup_zone": "City Center",
            "cab_type": "Sedan",
        },
        {
            "city": "Delhi",
            "distance_km": 8.2,
            "base_time_min": 18.0,
            "traffic_level": "Medium",
            "weather": "Sunny",
            "time_of_day": "Afternoon",
            "is_holiday": False,
            "pickup_zone": "Airport",
            "cab_type": "SUV",
        },
        {
            "city": "Chandigarh",
            "distance_km": 5.0,
            "base_time_min": 10.0,
            "traffic_level": "Low",
            "weather": "Sunny",
            "time_of_day": "Night",
            "is_holiday": False,
            "pickup_zone": "Suburb",
            "cab_type": "Mini",
        },
    ]

    for case in test_cases:
        print(f"\n📍 {case['city']} — {case['pickup_zone']} ({case['time_of_day']}, {case['weather']})")

        # Calculate time — only base_time_min + traffic_level now
        # (weather/time_of_day already baked into traffic_level upstream)
        est_time = calculate_time(
            case["base_time_min"],
            case["traffic_level"]
        )
        print(f"  Base time: {case['base_time_min']} min → Estimated: {est_time} min")

        # Get availability
        avail = get_cab_availability(case["city"], case["pickup_zone"])
        print(f"  Cab availability: {avail}")

        # Calculate surge
        surge = calculate_surge(
            case["city"], case["pickup_zone"], case["traffic_level"],
            case["weather"], case["is_holiday"], case["time_of_day"], avail
        )
        print(f"  Surge multiplier: {surge}x")

        # Calculate fare
        fare = calculate_fare(
            case["distance_km"], est_time, case["cab_type"],
            surge, case["city"], case["pickup_zone"], case["time_of_day"]
        )
        print(f"  Fare ({case['cab_type']}): ₹{fare}")

    print(f"\n✅ Fare calculator ready!")
    print(f"   Import in pipeline with:")
    print(f"   from data_collection.fare_calculator import calculate_fare, calculate_time, calculate_surge, get_cab_availability")
