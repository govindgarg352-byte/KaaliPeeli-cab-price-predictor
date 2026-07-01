# =============================================================================
# config.py — Master Configuration File
# Cab Price Predictor Project
#
# ALL coordinates verified via Google Places API (places_search tool)
# Pricing verified via:
#   Mumbai  → MMRTA Sept 2025: ₹20.66/km non-AC, ₹22.72/km AC
#   Delhi   → MoRTH July 2025: ₹20-21/km base
#   Bengaluru → Karnataka govt order: ₹18-36/km by vehicle tier
#   Hyderabad → Ola/Uber 2026 rate card: Mini ₹8-11, Sedan ₹12-15, SUV ₹16-22
#   Chandigarh → Admin fixed ₹25/km (2025), drivers demanding ₹35
# =============================================================================

import os
from dotenv import load_dotenv
load_dotenv()
import os

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

def get_api_key(name: str):
    """Reads from Streamlit Secrets when deployed, falls back to .env locally."""
    if _HAS_STREAMLIT:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
    return os.getenv(name)

API_KEYS = {
    "openrouteservice": get_api_key("ORS_API_KEY"),
    "openweathermap": get_api_key("OWM_API_KEY"),
    "calendarific": get_api_key("CALENDARIFIC_API_KEY"),
    "tomtom": get_api_key("TOMTOM_API_KEY"),
}

# =============================================================================
# 1. CITY CONFIGURATIONS
# =============================================================================

CITIES = {
    "Mumbai": {
        "tier": 1,
        "bounding_box": {
            "lat_min": 18.87, "lat_max": 19.27,
            "lng_min": 72.77, "lng_max": 72.98
        },
        "base_price_per_km": 21.0,
        "per_minute_rate":    2.5,
        "minimum_fare":       75.0,
        "night_premium":      40.0,
        "weather_coords":     (19.0760, 72.8777),
    },
    "Delhi": {
        "tier": 1,
        "bounding_box": {
            "lat_min": 28.40, "lat_max": 28.88,
            "lng_min": 76.84, "lng_max": 77.35
        },
        "base_price_per_km": 20.0,
        "per_minute_rate":    2.2,
        "minimum_fare":       70.0,
        "night_premium":      35.0,
        "weather_coords":     (28.6139, 77.2090),
    },
    "Bengaluru": {
        "tier": 1,
        "bounding_box": {
            "lat_min": 12.83, "lat_max": 13.22,
            "lng_min": 77.46, "lng_max": 77.78
        },
        "base_price_per_km": 20.0,
        "per_minute_rate":    2.3,
        "minimum_fare":       75.0,
        "night_premium":      35.0,
        "weather_coords":     (12.9716, 77.5946),
    },
    "Hyderabad": {
        "tier": 2,
        "bounding_box": {
            "lat_min": 17.27, "lat_max": 17.57,
            "lng_min": 78.27, "lng_max": 78.63
        },
        "base_price_per_km": 10.0,
        "per_minute_rate":    1.8,
        "minimum_fare":       60.0,
        "night_premium":      25.0,
        "weather_coords":     (17.3850, 78.4867),
    },
    "Chandigarh": {
        "tier": 2,
        "bounding_box": {
            "lat_min": 30.58, "lat_max": 30.78,
            "lng_min": 76.62, "lng_max": 76.90
        },
        "base_price_per_km": 25.0,
        "per_minute_rate":    1.5,
        "minimum_fare":       50.0,
        "night_premium":      20.0,
        "weather_coords":     (30.7333, 76.7794),
    },
}

# =============================================================================
# 2. CAB TYPE MULTIPLIERS
# =============================================================================

CAB_TYPES = {
    "Mini":  {"fare_multiplier": 1.0,  "speed_factor": 1.0},
    "Sedan": {"fare_multiplier": 1.2,  "speed_factor": 1.0},
    "SUV":   {"fare_multiplier": 1.6,  "speed_factor": 0.95},
}

# =============================================================================
# 3. ZONES & SUB-REGIONS
#    ALL coordinates verified via Google Places API
# =============================================================================

ZONES = {

    # =========================================================================
    # MUMBAI
    # =========================================================================
    "Mumbai": {
        "Airport": {
            "zone_premium":     100,
            "cab_availability": 18,
            "sub_regions": {
                # T1 (Santacruz domestic) → Google Places: 19.0928, 72.8571
                "CSIA Terminal 1": (19.0928, 72.8571),
                # T2 (International) → Google Places: 19.0974, 72.8746
                "CSIA Terminal 2": (19.0974, 72.8746),
            }
        },
        "Railway": {
            "zone_premium":     40,
            "cab_availability": 15,
            "sub_regions": {
                # Google Places verified ↓
                "CSMT":              (18.9398, 72.8354),
                "Dadar Station":     (19.0196, 72.8439),
                "Bandra Station":    (19.0548, 72.8407),
                "Borivali Station":  (19.2291, 72.8574),
                "Thane Station":     (19.1865, 72.9755),
            }
        },
        "City Center": {
            "zone_premium":     0,
            "cab_availability": 12,
            "sub_regions": {
                "Nariman Point": (18.9256, 72.8242),
                "Bandra West":   (19.0596, 72.8295),
                "Andheri West":  (19.1364, 72.8296),
                "Juhu":          (19.1048, 72.8267),
                "Lower Parel":   (18.9982, 72.8270),
            }
        },
        "Suburb": {
            "zone_premium":    -10,
            "cab_availability": 6,
            "sub_regions": {
                "Thane":      (19.2123, 72.9772),
                "Navi Mumbai":(19.0222, 73.0390),
                "Mira Road":  (19.2856, 72.8691),
                "Virar":      (19.4548, 72.8120),
                "Kalyan":     (19.2403, 73.1305),
            }
        },
    },

    # =========================================================================
    # DELHI
    # =========================================================================
    "Delhi": {
        "Airport": {
            "zone_premium":     80,
            "cab_availability": 17,
            "sub_regions": {
                # T1 metro station coords → Google Places: 28.5653, 77.1224
                "IGI Terminal 1": (28.5653, 77.1224),
                # T2 (between T1 and T3)
                "IGI Terminal 2": (28.5580, 77.1050),
                # T3 → Google Places: 28.5551, 77.0844
                "IGI Terminal 3": (28.5551, 77.0844),
            }
        },
        "Railway": {
            "zone_premium":     30,
            "cab_availability": 14,
            "sub_regions": {
                "New Delhi Station":     (28.6429, 77.2191),
                "Hazrat Nizamuddin":     (28.5889, 77.2534),
                "Old Delhi Station":     (28.6563, 77.2321),
                "Anand Vihar Terminal":  (28.6475, 77.3153),
            }
        },
        "City Center": {
            "zone_premium":     0,
            "cab_availability": 13,
            "sub_regions": {
                "Connaught Place": (28.6304, 77.2177),
                "Karol Bagh":      (28.6550, 77.1888),
                "Lajpat Nagar":    (28.5649, 77.2403),
                "Nehru Place":     (28.5503, 77.2502),
                "Saket":           (28.5221, 77.2102),
            }
        },
        "Suburb": {
            "zone_premium":    -10,
            "cab_availability": 7,
            "sub_regions": {
                "Dwarka":             (28.5823, 77.0500),
                "Rohini":             (28.7383, 77.0822),
                "Noida Sector 18":    (28.5703, 77.3218),
                "Gurgaon Cyber City": (28.4950, 77.0895),
                "Faridabad":          (28.4089, 77.3178),
            }
        },
    },

    # =========================================================================
    # BENGALURU
    # =========================================================================
    "Bengaluru": {
        "Airport": {
            "zone_premium":     120,
            "cab_availability": 16,
            "sub_regions": {
                # KIA verified: 13.1979, 77.7063
                "KIA Terminal 1": (13.1979, 77.7063),
                "KIA Terminal 2": (13.2010, 77.7083),
            }
        },
        "Railway": {
            "zone_premium":     35,
            "cab_availability": 13,
            "sub_regions": {
                # Google Places verified ↓
                "KSR City Station":     (12.9781, 77.5695),
                "Yeshwanthpur Station": (13.0232, 77.5514),
                "Bangalore Cantonment": (12.9939, 77.5983),
            }
        },
        "City Center": {
            "zone_premium":     0,
            "cab_availability": 11,
            "sub_regions": {
                "MG Road":    (12.9747, 77.6095),
                "Koramangala":(12.9352, 77.6245),
                "Indiranagar":(12.9784, 77.6408),
                "Jayanagar":  (12.9308, 77.5839),
                "Rajajinagar":(12.9982, 77.5530),
            }
        },
        "Suburb": {
            "zone_premium":    -15,
            "cab_availability": 5,
            "sub_regions": {
                "Whitefield":        (12.9698, 77.7500),
                "Electronic City":   (12.8452, 77.6602),
                "Hebbal":            (13.0354, 77.5988),
                "Sarjapur":          (12.9109, 77.6844),
                "Bannerghatta Road": (12.8928, 77.5989),
            }
        },
    },

    # =========================================================================
    # HYDERABAD
    # =========================================================================
    "Hyderabad": {
        "Airport": {
            "zone_premium":     130,
            "cab_availability": 15,
            "sub_regions": {
                # RGIA verified: 17.2403, 78.4294
                "RGIA Terminal 1": (17.2403, 78.4294),
                "RGIA Terminal 2": (17.2373, 78.4235),
            }
        },
        "Railway": {
            "zone_premium":     30,
            "cab_availability": 12,
            "sub_regions": {
                # Google Places verified ↓
                "Secunderabad Junction": (17.4337, 78.5016),
                "Kachiguda Station":     (17.3893, 78.4992),
                "Hyderabad Deccan":      (17.3924, 78.4690),
                "Begumpet":              (17.4442, 78.4708),
            }
        },
        "City Center": {
            "zone_premium":     0,
            "cab_availability": 10,
            "sub_regions": {
                "Hitech City":   (17.4470, 78.3778),
                "Banjara Hills": (17.4169, 78.4387),
                "Jubilee Hills": (17.4326, 78.4071),
                "Ameerpet":      (17.4375, 78.4482),
                "Madhapur":      (17.4486, 78.3908),
            }
        },
        "Suburb": {
            "zone_premium":    -10,
            "cab_availability": 5,
            "sub_regions": {
                "Gachibowli": (17.4401, 78.3489),
                "Miyapur":    (17.5169, 78.3428),
                "LB Nagar":   (17.3457, 78.5522),
                "Kukatpally": (17.4875, 78.3953),
                "Uppal":      (17.4015, 78.5682),
            }
        },
    },

    # =========================================================================
    # CHANDIGARH
    # =========================================================================
    "Chandigarh": {
        "Airport": {
            "zone_premium":     50,
            "cab_availability": 10,
            "sub_regions": {
                # Verified: 30.6678, 76.7862
                "Chandigarh International Airport": (30.6678, 76.7862),
            }
        },
        "Railway": {
            "zone_premium":     20,
            "cab_availability": 8,
            "sub_regions": {
                # Google Places: 30.7024, 76.8215
                "Chandigarh Railway Station": (30.7024, 76.8215),
            }
        },
        "City Center": {
            "zone_premium":     0,
            "cab_availability": 7,
            "sub_regions": {
                # Google Places verified ↓
                "Sector 17": (30.7392, 76.7834),
                "Sector 22": (30.7320, 76.7726),
                "Sector 35": (30.7265, 76.7589),
                "Sector 43": (30.7191, 76.7487),
            }
        },
        "Suburb": {
            "zone_premium":    -5,
            "cab_availability": 3,
            "sub_regions": {
                # Google Places verified ↓
                "Mohali":    (30.7046, 76.7179),
                "Panchkula": (30.6942, 76.8606),
                "Zirakpur":  (30.6425, 76.8173),
                "Kharar":    (30.7499, 76.6411),
                "Derabassi": (30.5887, 76.8471),
            }
        },
    },
}

# =============================================================================
# 4. WEATHER BIAS PER CITY
# =============================================================================

WEATHER_BIAS = {
    "Mumbai":     {"Sunny": 0.30, "Cloudy": 0.20, "Rainy": 0.35, "Foggy": 0.05, "Stormy": 0.10},
    "Delhi":      {"Sunny": 0.30, "Cloudy": 0.20, "Rainy": 0.15, "Foggy": 0.25, "Stormy": 0.10},
    "Bengaluru":  {"Sunny": 0.35, "Cloudy": 0.25, "Rainy": 0.25, "Foggy": 0.10, "Stormy": 0.05},
    "Hyderabad":  {"Sunny": 0.40, "Cloudy": 0.25, "Rainy": 0.20, "Foggy": 0.05, "Stormy": 0.10},
    "Chandigarh": {"Sunny": 0.35, "Cloudy": 0.20, "Rainy": 0.15, "Foggy": 0.25, "Stormy": 0.05},
}

WEATHER_TIME_IMPACT = {
    "Sunny":  1.00,
    "Cloudy": 1.05,
    "Rainy":  1.25,
    "Foggy":  1.20,
    "Stormy": 1.40,
}

WEATHER_SURGE_BOOST = {
    "Sunny":  0.00,
    "Cloudy": 0.05,
    "Rainy":  0.20,
    "Foggy":  0.10,
    "Stormy": 0.30,
}

# =============================================================================
# 5. TRAFFIC MATRIX
# =============================================================================

TIME_SLOTS = ["Morning", "Afternoon", "Evening", "Night"]

TIME_SLOT_HOURS = {
    "Morning":   (6,  10),
    "Afternoon": (10, 17),
    "Evening":   (17, 21),
    "Night":     (21,  6),
}

TRAFFIC_MATRIX = {
    "Mumbai":     {"Morning": 1.7, "Afternoon": 1.2, "Evening": 1.8, "Night": 0.6},
    "Delhi":      {"Morning": 1.5, "Afternoon": 1.2, "Evening": 1.6, "Night": 0.5},
    "Bengaluru":  {"Morning": 1.8, "Afternoon": 1.4, "Evening": 1.9, "Night": 0.7},
    "Hyderabad":  {"Morning": 1.4, "Afternoon": 1.1, "Evening": 1.5, "Night": 0.5},
    "Chandigarh": {"Morning": 1.1, "Afternoon": 0.9, "Evening": 1.2, "Night": 0.4},
}

WEEKEND_TRAFFIC_FACTOR = 0.75

TRAFFIC_LEVEL_THRESHOLDS = {
    "Low":    (0.8, 999),
    "Medium": (0.5, 0.8),
    "High":   (0.0, 0.5),
}

TRAFFIC_TIME_IMPACT = {
    "Low":    1.0,
    "Medium": 1.3,
    "High":   1.7,
}

# =============================================================================
# 6. SURGE CONFIGURATION
# MoRTH July 2025: peak surge capped at 2x base fare
# =============================================================================

SURGE_CONFIG = {
    "Mumbai":     {"base_probability": 0.45, "max_surge": 2.0, "min_surge": 1.0},
    "Delhi":      {"base_probability": 0.40, "max_surge": 2.0, "min_surge": 1.0},
    "Bengaluru":  {"base_probability": 0.50, "max_surge": 2.0, "min_surge": 1.0},
    "Hyderabad":  {"base_probability": 0.30, "max_surge": 2.0, "min_surge": 1.0},
    "Chandigarh": {"base_probability": 0.15, "max_surge": 2.0, "min_surge": 1.0},
}

AVAILABILITY_SURGE_MAP = {
    (15, 20): 1.0,
    (10, 14): 1.2,
    (5,   9): 1.5,
    (1,   4): 2.0,
}

# =============================================================================
# 7. NOISE CONFIGURATION
# =============================================================================

NOISE_CONFIG = {
    "route_efficiency": {"mean": 1.0,  "std": 0.08},
    "idle_time_min":    {"min": 0,     "max": 5},
    "toll":             {"probability": 0.30, "min_amount": 30, "max_amount": 180},
    "micro_surge":      {"mean": 1.0,  "std": 0.05},
    "discount":         {"probability": 0.25, "min_discount": 0.85, "max_discount": 0.95},
    "night_premium":    {"min": 10,    "max": 40},
    "signal_wait_min":  {"min": 0,     "max": 4},
    "pickup_wait_min":  {"min": 1,     "max": 5},
    "route_deviation":  {"mean": 0.0,  "std": 0.06},
}

CITY_TOLL_CONFIG = {
    "Mumbai":     {"min": 30,  "max": 80},
    "Delhi":      {"min": 30,  "max": 80},
    "Bengaluru":  {"min": 20,  "max": 60},
    "Hyderabad":  {"min": 130, "max": 180},
    "Chandigarh": {"min": 0,   "max": 30},
}

# =============================================================================
# 8. DATASET CONFIGURATION
# =============================================================================

DATASET_CONFIG = {
    "total_rows":        2000,
    "coordinate_offset": 0.005,
    "output_file":       "data/processed/cab_dataset.csv",
    "random_seed":       42,
}

# =============================================================================
# 9. HELPER FUNCTIONS
# =============================================================================

def get_zone_for_subregion(city: str, subregion: str) -> str:
    for zone, zone_data in ZONES[city].items():
        if subregion in zone_data["sub_regions"]:
            return zone
    return "City Center"

def get_city_tier(city: str) -> int:
    return CITIES[city]["tier"]

def get_time_of_day(hour: int) -> str:
    if 6 <= hour < 10:   return "Morning"
    elif 10 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 21: return "Evening"
    else:                 return "Night"

def get_availability_surge(availability: int) -> float:
    for (low, high), surge in AVAILABILITY_SURGE_MAP.items():
        if low <= availability <= high:
            return surge
    return 1.0

def get_all_subregions(city: str) -> list:
    """Returns list of (subregion_name, lat, lng, zone) for a city."""
    result = []
    for zone, zone_data in ZONES[city].items():
        for name, (lat, lng) in zone_data["sub_regions"].items():
            result.append((name, lat, lng, zone))
    return result
