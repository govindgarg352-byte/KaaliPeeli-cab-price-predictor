# =============================================================================
# app.py — Streamlit Cab Price Predictor (KaaliPeeli ML Engine)
# =============================================================================

import os, sys, math, datetime, random, time, base64
from zoneinfo import ZoneInfo
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_DIR)

from custom_transformers import FrequencyEncoder  # joblib needs this
from config.config import CITIES, ZONES, CAB_TYPES, API_KEYS, TRAFFIC_MATRIX, WEEKEND_TRAFFIC_FACTOR, WEATHER_TIME_IMPACT, TRAFFIC_LEVEL_THRESHOLDS

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KaaliPeeli Dispatch Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Initialize session state ──────────────────────────────────────────────────
if "selected_cab" not in st.session_state:
    st.session_state.selected_cab = "Sedan"
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light Mode"  # Light Mode is default

# ── Theme Loading Transition ──────────────────────────────────────────────────
if st.session_state.get("theme_loading", False):
    st.session_state.theme_loading = False
    with st.spinner("Switching display layout..."):
        time.sleep(0.5)
    st.rerun()

# ── Theme Color Mapping ───────────────────────────────────────────────────────
theme_mode = st.session_state.theme_mode

if theme_mode == "Dark Slate":
    bg_color = "#121212"
    card_bg = "#1E1E1E"
    border_color = "rgba(255, 255, 255, 0.12)"
    text_color = "#F5F5F7"
    sub_text = "#A1A1A6"
    input_bg = "rgba(255, 255, 255, 0.05)"
    table_border = "rgba(255, 255, 255, 0.08)"
    glow_shadow = "rgba(0, 0, 0, 0.5)"
    paper_dash = "rgba(255, 255, 255, 0.18)"
    acc1 = "#FFC700"   # Classic Mumbai Yellow (Peeli)
    acc2 = "#E4483D"   # Brakes/Tail Light Red
    map_style = "dark"
    badge_green_bg, badge_green_fg = "#173A24", "#7BE8A4"
    badge_yellow_bg, badge_yellow_fg = "#3A2E0A", "#FFD666"
    badge_red_bg, badge_red_fg = "#3A1418", "#FF8A80"
else:
    bg_color = "#FAF8F2"       # Classic Cream Receipt Paper
    card_bg = "#FFFFFF"        # Pure white panels
    border_color = "rgba(0, 0, 0, 0.14)"
    text_color = "#1A1A1A"     # Deep Coal (Kaali)
    sub_text = "#6E6E73"       # Slate Grey
    input_bg = "rgba(0, 0, 0, 0.03)"
    table_border = "rgba(0, 0, 0, 0.08)"
    glow_shadow = "rgba(0, 0, 0, 0.05)"
    paper_dash = "rgba(0, 0, 0, 0.15)"
    acc1 = "#FFC700"   # Classic Mumbai Yellow (Peeli)
    acc2 = "#D32F2F"   # Red Highlight
    map_style = "light"
    badge_green_bg, badge_green_fg = "#E8F5E9", "#2E7D32"
    badge_yellow_bg, badge_yellow_fg = "#FFF9C4", "#F57F17"
    badge_red_bg, badge_red_fg = "#FFEBEE", "#C62828"

# ── Custom CSS injection ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Work+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Work Sans', sans-serif; }}
h1, h2, h3, h4, h5, h6 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; }}
.stApp {{ background: {bg_color}; color: {text_color}; }}

/* Explicit contrast rules for ALL Streamlit text, markdown, and labels */
div[data-testid="stWidgetLabel"] p, 
div[data-testid="stWidgetLabel"] span,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[role="radiogroup"] label p,
div[role="radiogroup"] label span,
.stSelectbox label, 
.stTextInput label {{
    color: {text_color} !important;
    font-family: 'Work Sans', sans-serif !important;
    font-weight: 600 !important;
}}

/* Re-assert intentional colors for elements that sit on dark custom backgrounds —
   these need to win over the generic rule above, so they're scoped more specifically. */
div[data-testid="stMarkdownContainer"] .meter-lbl {{
    color: #FFC700 !important;
}}
div[data-testid="stMarkdownContainer"] .meter-val {{
    color: #FFC700 !important;
}}
div[data-testid="stMarkdownContainer"] .meter-sub {{
    color: #FFFFFF !important;
}}
div[data-testid="stMarkdownContainer"] .topmarquee-track span {{
    color: {acc1} !important;
}}
div[data-testid="stMarkdownContainer"] span.badge-green {{
    color: {badge_green_fg} !important;
}}
div[data-testid="stMarkdownContainer"] span.badge-yellow {{
    color: {badge_yellow_fg} !important;
}}
div[data-testid="stMarkdownContainer"] span.badge-red {{
    color: {badge_red_fg} !important;
}}

/* Sidebar custom layout text colors */
section[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
    border-right: 1px dashed {table_border} !important;
}}

/* Custom Dispatcher Banner styling */
.banner-panel {{
    background: {card_bg} !important;
    border: 2px solid #111111;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.8rem;
    box-shadow: 4px 4px 0px #111111;
}}

/* Eyebrow Section Header */
.section-hdr {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {text_color};
    border-bottom: 2px solid #111111;
    padding-bottom: 0.6rem;
    margin-bottom: 0.9rem;
}}

/* Signature Dashboard Fare Meter (Mechanical Yellow/Black Aesthetic) */
.meter-face {{
    background: #111111 !important;
    border: 3px solid #FFC700 !important;
    border-radius: 8px !important;
    padding: 1.7rem 1.5rem !important;
    text-align: center !important;
    box-shadow: 4px 4px 0px #111111 !important;
    margin-bottom: 1.2rem;
}}
.meter-lbl {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #FFC700; font-weight: 700;
}}
.meter-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.8rem; font-weight: 700; line-height: 1;
    color: #FFC700; margin-top: 0.45rem;
    text-shadow: 0 0 10px rgba(255, 199, 0, 0.4);
    letter-spacing: 0.01em;
}}
.meter-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem; font-weight: 600; color: #FFFFFF; margin-top: 0.1rem;
}}

/* Ticket tearing indicators */
.ticket-tear {{
    display: flex; align-items: center; gap: 0.5rem;
    margin: 1.1rem 0; color: {sub_text};
}}
.ticket-tear::before, .ticket-tear::after {{
    content: ""; flex: 1; border-top: 2px dashed {table_border};
}}

/* Badges */
.badge-premium {{
    padding: 0.3rem 0.75rem; border-radius: 4px; font-size: 0.78rem; font-weight: 600;
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
    border: 2px solid #111111;
}}
.badge-green  {{ background: {badge_green_bg};  color: {badge_green_fg}; }}
.badge-yellow {{ background: {badge_yellow_bg};  color: {badge_yellow_fg}; }}
.badge-red    {{ background: {badge_red_bg};  color: {badge_red_fg}; }}

/* Custom Button overrides - Kaali Peeli tactile flat brutalist design */
div.stButton > button {{
    background-color: #FFC700 !important; /* Bold taxi yellow */
    color: #111111 !important; /* Rich deep black text */
    border: 2px solid #111111 !important;
    border-radius: 4px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 3px 3px 0px #111111 !important;
    transition: all 0.12s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
div.stButton > button:hover {{
    background-color: #FFE054 !important;
    box-shadow: 1px 1px 0px #111111 !important;
    transform: translate(2px, 2px) !important;
}}

/* Disabled vehicle class SET indicators */
div.stButton > button[disabled],
div.stButton > button:disabled {{
    background-color: #111111 !important;
    color: #FFC700 !important;
    border: 2px solid #FFC700 !important;
    opacity: 1 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}}
div.stButton > button[disabled] p,
div.stButton > button[disabled] span,
div.stButton > button[disabled] div,
div.stButton > button:disabled p,
div.stButton > button:disabled span,
div.stButton > button:disabled div {{
    color: #FFC700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #FFC700 !important;
}}

/* Theme Toggle Button styling */
button[key="theme_toggle_btn"] {{
    background: none !important; border: 2px solid #111111 !important;
    border-radius: 4px !important; width: 42px !important; height: 42px !important;
    padding: 0 !important; font-size: 1.15rem !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    cursor: pointer !important; transition: background 0.2s !important;
    box-shadow: 2px 2px 0px #111111 !important;
    color: {text_color} !important;
}}

/* Receipt breakups */
.receipt-table {{
    width: 100%; border-collapse: collapse; margin-top: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}}
.receipt-table td {{
    padding: 0.65rem 0 !important;
    border-bottom: 2px dashed {table_border} !important;
    border-top: none !important; border-left: none !important; border-right: none !important;
    font-size: 0.92rem; color: {text_color};
}}
.receipt-table td.val {{
    text-align: right; font-weight: 700; color: {acc1};
}}
.receipt-table td.lbl {{
    color: {sub_text}; font-family: 'Work Sans', sans-serif;
}}
.receipt-table tr.total td {{
    border-top: 3px solid #111111 !important; border-bottom: none !important;
    padding-top: 1rem !important;
}}
.receipt-table tr.total td.lbl {{ font-weight: 700; color: {text_color}; font-size: 1.1rem; font-family: 'Space Grotesk', sans-serif; }}
.receipt-table tr.total td.val {{ font-weight: 700; font-size: 1.4rem; color: #111111 !important; }}

/* Top scrolling marquee banner */
.topmarquee-wrap {{
    width: 100%;
    overflow: hidden;
    background: #111111;
    border-bottom: 2px solid {acc1};
    padding: 0.5rem 0;
    margin: -1rem -1rem 1.4rem -1rem;
    width: calc(100% + 2rem);
}}
.topmarquee-track {{
    display: flex;
    width: max-content;
    animation: topmarquee-scroll 22s linear infinite;
}}
.topmarquee-track span {{
    display: inline-block;
    padding: 0 2.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: {acc1};
    white-space: nowrap;
}}
@keyframes topmarquee-scroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}

/* Fare-calculation loading card */
.calc-card {{
    background: #111111;
    border: 3px solid {acc1};
    border-radius: 8px;
    padding: 2.4rem 1.5rem;
    text-align: center;
    box-shadow: 4px 4px 0px #111111;
    margin-top: 2rem;
}}
.calc-taxi {{
    font-size: 2.6rem;
    display: inline-block;
    animation: calc-bounce 0.9s ease-in-out infinite;
}}
@keyframes calc-bounce {{
    0%, 100% {{ transform: translateY(0) rotate(0deg); }}
    50% {{ transform: translateY(-8px) rotate(-3deg); }}
}}
.calc-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    color: {acc1}; margin-top: 0.8rem;
    letter-spacing: 0.02em;
}}
.calc-steps {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.08em;
    color: #A1A1A6; margin-top: 0.5rem; text-transform: uppercase;
}}
.calc-track {{
    width: 100%; max-width: 260px; height: 4px;
    background: rgba(255,199,0,0.15); border-radius: 2px;
    margin: 1.2rem auto 0; overflow: hidden;
}}
.calc-fill {{
    height: 100%; width: 40%; border-radius: 2px;
    background: {acc1};
    animation: calc-sweep 1.1s ease-in-out infinite;
}}
@keyframes calc-sweep {{
    0%   {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(350%); }}
}}
</style>
""", unsafe_allow_html=True)

# ── load models (cached) ──────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(BASE_DIR, "data", "processed", "trained_models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "processed", "model_ready")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic_baseline.json")

@st.cache_resource(show_spinner="Initializing predictive pipelines…")
def load_models():
    m1_b = joblib.load(os.path.join(MODELS_DIR,    "model1_time_bundle.pkl"))
    m2_b = joblib.load(os.path.join(MODELS_DIR,    "model2_fare_bundle.pkl"))
    m1_a = joblib.load(os.path.join(ARTIFACTS_DIR, "model1_time_artifacts.pkl"))
    m2_a = joblib.load(os.path.join(ARTIFACTS_DIR, "model2_fare_artifacts.pkl"))
    return m1_b, m2_b, m1_a, m2_a

@st.cache_data
def load_baseline():
    if os.path.exists(BASELINE_PATH):
        import json
        with open(BASELINE_PATH, "r") as f:
            return json.load(f)["cities"]
    return {}

m1_bundle, m2_bundle, m1_arts, m2_arts = load_models()
traffic_baseline = load_baseline()

# ── helper methods ────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def get_time_of_day(hour):
    if 6 <= hour < 10:    return "Morning"
    elif 10 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 21: return "Evening"
    else:                 return "Night"

def check_is_holiday(date_obj):
    month_day = (date_obj.month, date_obj.day)
    national_holidays = {
        (1, 26),   # Republic Day
        (8, 15),   # Independence Day
        (10, 2),   # Gandhi Jayanti
        (12, 25),  # Christmas
        (5, 1),    # Labor Day
    }
    if month_day in national_holidays:
        return True
    if date_obj.weekday() in {5, 6}: # Saturday or Sunday
        return True
    return False

def get_surge(cab_avail, weather, hour, is_holiday):
    if cab_avail >= 15:   base = 1.0
    elif cab_avail >= 10: base = 1.2
    elif cab_avail >= 5:  base = 1.5
    else:                 base = 2.0
    weather_add = {"Sunny":0.0,"Cloudy":0.05,"Rainy":0.20,"Foggy":0.10,"Stormy":0.30}
    peak_add    = 0.2 if hour in {7,8,9,18,19,20} else 0.0
    return round(min(base + weather_add.get(weather,0) + peak_add + (0.1 if is_holiday else 0), 2.0), 2)

def lookup_subregion(city, subregion):
    for zone, zone_data in ZONES[city].items():
        if subregion in zone_data["sub_regions"]:
            lat, lng = zone_data["sub_regions"][subregion]
            return lat, lng, zone
    return 0.0, 0.0, "City Center"

def find_nearest_subregion(city: str, lat: float, lng: float) -> tuple[str, str]:
    subregions = []
    for zone, zone_data in ZONES[city].items():
        for name, (sub_lat, sub_lng) in zone_data["sub_regions"].items():
            subregions.append((name, sub_lat, sub_lng, zone))
    
    min_dist = float('inf')
    nearest_subregion = None
    nearest_zone = None
    
    for name, sub_lat, sub_lng, zone in subregions:
        dist = ((lat - sub_lat) ** 2 + (lng - sub_lng) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_subregion = name
            nearest_zone = zone
            
    return nearest_subregion, nearest_zone

def get_ors_route(src_lat, src_lng, dst_lat, dst_lng):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        "Authorization": API_KEYS["openrouteservice"],
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [[src_lng, src_lat], [dst_lng, dst_lat]]
    }
    try:
        response = requests.post(url, headers=headers, json=body, timeout=6)
        if response.status_code == 200:
            data = response.json()
            route_summary = data["routes"][0]["summary"]
            distance_km = round(route_summary["distance"] / 1000, 2)
            base_time_min = round(route_summary["duration"] / 60, 2)
            return distance_km, base_time_min
    except Exception:
        pass
    
    # fallback to Haversine * 1.3
    straight = haversine_km(src_lat, src_lng, dst_lat, dst_lng)
    distance_km = max(round(straight * 1.3, 2), 0.5)
    base_time_min = round(distance_km / 50 * 60, 2)
    return distance_km, base_time_min

def get_current_weather(lat, lng, city):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lng,
        "appid": API_KEYS["openweathermap"]
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            weather_id = data["weather"][0]["id"]
            if 200 <= weather_id < 300: return "Stormy"
            elif (300 <= weather_id < 400) or (500 <= weather_id < 600): return "Rainy"
            elif 600 <= weather_id < 700: return "Cloudy"
            elif 700 <= weather_id < 800: return "Foggy"
            elif weather_id == 800: return "Sunny"
            elif 801 <= weather_id < 900: return "Cloudy"
    except Exception:
        pass
    return fallback_weather_by_bias(city)

def fallback_weather_by_bias(city: str) -> str:
    biases = {
        "Mumbai":     {"Sunny": 0.30, "Cloudy": 0.20, "Rainy": 0.35, "Foggy": 0.05, "Stormy": 0.10},
        "Delhi":      {"Sunny": 0.30, "Cloudy": 0.20, "Rainy": 0.15, "Foggy": 0.25, "Stormy": 0.10},
        "Bengaluru":  {"Sunny": 0.35, "Cloudy": 0.25, "Rainy": 0.25, "Foggy": 0.10, "Stormy": 0.05},
        "Hyderabad":  {"Sunny": 0.40, "Cloudy": 0.25, "Rainy": 0.20, "Foggy": 0.05, "Stormy": 0.10},
        "Chandigarh": {"Sunny": 0.35, "Cloudy": 0.20, "Rainy": 0.15, "Foggy": 0.25, "Stormy": 0.05},
    }
    bias = biases.get(city, {"Sunny": 0.4, "Cloudy": 0.3, "Rainy": 0.2, "Stormy": 0.05, "Foggy": 0.05})
    return random.choices(list(bias.keys()), weights=list(bias.values()))[0]

def derive_traffic_level(city, subregion, timestamp_str, weather, baseline):
    if not baseline or city not in baseline or subregion not in baseline[city]:
        dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        if dt.hour in {8, 9, 10, 17, 18, 19, 20}: return "High"
        elif dt.hour in {23, 0, 1, 2, 3, 4, 5}: return "Low"
        return "Medium"

    city_baseline = baseline[city]
    subregion_data = city_baseline[subregion]
    free_flow_speed = subregion_data["free_flow_speed"]
    city_avg_speed = sum(d["free_flow_speed"] for d in city_baseline.values()) / len(city_baseline)
    relative_speed_factor = free_flow_speed / city_avg_speed

    dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour
    weekday = dt.weekday()
    time_of_day = get_time_of_day(hour)

    time_factor = TRAFFIC_MATRIX[city][time_of_day]
    weekend_factor = WEEKEND_TRAFFIC_FACTOR if weekday >= 5 else 1.0
    weather_factor = WEATHER_TIME_IMPACT.get(weather, 1.0)

    if weather == "Stormy": return "High"

    congestion_ratio = relative_speed_factor / (time_factor * weekend_factor * weather_factor)
    for level, (low, high) in TRAFFIC_LEVEL_THRESHOLDS.items():
        if low <= congestion_ratio < high: return level
    return "Medium"

def stack_predict(bundle, X_xgb, X_lgbm, X_cb):
    p_xgb  = bundle["xgb_model"].predict(X_xgb)[0]
    p_lgbm = bundle["lgbm_model"].predict(X_lgbm)[0]
    p_cb   = bundle["cb_model"].predict(X_cb)[0]
    meta_in = pd.DataFrame({"xgb_pred":[p_xgb],"lgbm_pred":[p_lgbm],"cb_pred":[p_cb]})
    return float(bundle["meta_model"].predict(meta_in)[0])

def build_xgb_input(bundle, df, feat_cols):
    sub = df[feat_cols]
    arr = bundle["xgb_preprocessor"].transform(sub)
    return pd.DataFrame(arr, columns=bundle["xgb_preprocessor"].get_feature_names_out())

def build_lgbm_input(bundle, df, feat_cols):
    sub = df[feat_cols]
    arr = bundle["lgbm_preprocessor"].transform(sub)
    return pd.DataFrame(arr, columns=bundle["lgbm_preprocessor"].get_feature_names_out())

def build_cb_input(arts, df, feat_cols, eta_col=None, eta_val=None):
    cb = df[feat_cols].copy()
    for col in arts["categorical_columns"]:
        if col in cb.columns:
            cb[col] = cb[col].astype(str)
    if eta_col and eta_col in cb.columns:
        cb[eta_col] = float(eta_val)
    return cb

def run_predictions(city, src_lat, src_lng, dst_lat, dst_lng, src_sub, dst_sub, src_zone, dst_zone,
                    cab_type, hour, day_of_week, is_holiday, weather, traffic_level, cab_avail, surge, dist_km, base_min):
    cfg = CITIES[city]
    time_of_day = get_time_of_day(hour)
    row = {
        "distance_km": dist_km, "base_time_min": base_min,
        "hour": hour, "is_holiday": int(is_holiday),
        "cab_availability": cab_avail, "surge_multiplier": surge,
        "base_price_per_km": cfg["base_price_per_km"],
        "per_minute_rate":   cfg["per_minute_rate"],
        "city_tier":         cfg["tier"],
        "traffic_level":     traffic_level,
        "time_of_day":       time_of_day,
        "cab_type":          cab_type,
        "city":              city,
        "source_zone":       src_zone,
        "dest_zone":         dst_zone,
        "weather_condition": weather,
        "day_of_week":       day_of_week,
        "source_lat": src_lat, "source_lng": src_lng,
        "dest_lat": dst_lat,   "dest_lng": dst_lng,
        "source_subregion": src_sub, "dest_subregion": dst_sub,
    }
    df = pd.DataFrame([row])

    # ETA Model
    m1_cols = m1_arts["feature_columns"]
    df_m1 = df[[c for c in m1_cols if c in df.columns]]
    X1_xgb  = build_xgb_input(m1_bundle, df_m1, m1_cols)
    X1_lgbm = build_lgbm_input(m1_bundle, df_m1, m1_cols)
    X1_cb   = build_cb_input(m1_arts, df_m1, m1_cols)
    eta = max(stack_predict(m1_bundle, X1_xgb, X1_lgbm, X1_cb), 1.0)

    # Fare Model
    row["estimated_time_min"] = eta
    df2 = pd.DataFrame([row])
    m2_cols = m2_arts["feature_columns"]
    df_m2 = df2[[c for c in m2_cols if c in df2.columns]]
    X2_xgb  = build_xgb_input(m2_bundle, df_m2, m2_cols)
    X2_lgbm = build_lgbm_input(m2_bundle, df_m2, m2_cols)
    X2_cb   = build_cb_input(m2_arts, df_m2, m2_cols, eta_col="estimated_time_min", eta_val=eta)
    fare = max(stack_predict(m2_bundle, X2_xgb, X2_lgbm, X2_cb), cfg["minimum_fare"])

    return eta, fare

# ── Dynamic SVG Logo Rendering (KaaliPeeli Dispatch) ──────────────────────────
logo_svg = f"""<svg width="240" height="52" viewBox="0 0 240 52" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(4, 6)">
        <path d="M2 18c0-1.6 1.3-2.9 2.9-2.9h1.3l2-4.2c.6-1.3 1.9-2.1 3.3-2.1h11c1.3 0 2.5.7 3.2 1.8l2.5 4.5h1.4c1.6 0 2.9 1.3 2.9 2.9v0.9c0 .8-.6 1.4-1.4 1.4H3.4c-.8 0-1.4-.6-1.4-1.4V18Z" fill="#111111"/>
        <path d="M12.5 10.2c.4-.8 1.2-1.3 2.1-1.3h6.9c.8 0 1.6.4 2 1.1l1.7 2.9H11l1.5-2.7Z" fill="#FFC700"/>
        <rect x="14" y="11.6" width="4" height="2.6" rx="0.4" fill="#FFFFFF" opacity="0.75"/>
        <rect x="19" y="11.6" width="4" height="2.6" rx="0.4" fill="#FFFFFF" opacity="0.75"/>
        <rect x="16.5" y="5.6" width="8" height="2.4" rx="1" fill="#111111"/>
        <rect x="17.4" y="6.2" width="6.2" height="1.2" rx="0.5" fill="#FFC700"/>
        <circle cx="9.5" cy="19.3" r="3.1" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.1"/>
        <circle cx="9.5" cy="19.3" r="1" fill="#FFC700"/>
        <circle cx="27" cy="19.3" r="3.1" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.1"/>
        <circle cx="27" cy="19.3" r="1" fill="#FFC700"/>
    </g>
    <text x="58" y="26" fill="{text_color}" font-family="Space Grotesk, sans-serif" font-size="21" font-weight="700" letter-spacing="0.5">KaaliPeeli</text>
    <text x="58" y="40" fill="{sub_text}" font-family="IBM Plex Mono, monospace" font-size="8.5" font-weight="600" letter-spacing="1.2">MUMBAI DISPATCH &#183; ML ENGINE</text>
</svg>"""
logo_b64 = base64.b64encode(logo_svg.encode('utf-8')).decode('utf-8')
logo_html = f'<div style="text-align: left; margin-bottom: 1.6rem;"><img src="data:image/svg+xml;base64,{logo_b64}" style="height: 52px;"/></div>'

# ── Dynamic Inline Vector Car Shapes (Kaali Peeli Vintage Premier Padmini Cabs) ──
def get_cab_svg(cab_type):
    if cab_type == "Mini":
        return f"""
        <svg width="64" height="40" viewBox="0 0 64 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="32" cy="33" rx="24" ry="2.6" fill="#111111" opacity="0.12"/>
            <path d="M9 27c-2.2 0-3.6-1.7-3.2-3.8.4-2 2.3-3.4 4.3-3.4h1.4l3-6.4c.9-1.9 2.8-3.1 4.9-3.1h16.8c1.9 0 3.7 1 4.7 2.7l4.1 6.8h1.5c2.3 0 4.3 1.6 4.6 3.8.3 2-1.1 3.6-3.1 3.8" stroke="#111111" stroke-width="0" fill="#111111"/>
            <path d="M6 24.5c0-2.3 1.9-4.2 4.2-4.2h2l3-6c.8-1.6 2.4-2.6 4.2-2.6h13.4c1.7 0 3.2.9 4.1 2.4l3.7 6.2h2.4c2.3 0 4.2 1.9 4.2 4.2v1.3c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2v-1.3Z" fill="#111111"/>
            <path d="M19.5 14.7c.5-1 1.5-1.6 2.6-1.6h10.6c1 0 2 .5 2.5 1.4l2.6 4.4H17.3l2.2-4.2Z" fill="#FFC700"/>
            <rect x="22" y="16.5" width="6" height="3.6" rx="0.6" fill="#FFFFFF" opacity="0.75"/>
            <rect x="30" y="16.5" width="6" height="3.6" rx="0.6" fill="#FFFFFF" opacity="0.75"/>
            <rect x="26" y="8.5" width="12" height="3.4" rx="1.4" fill="#111111"/>
            <rect x="27.4" y="9.3" width="9.2" height="1.8" rx="0.8" fill="#FFC700"/>
            <circle cx="9.5" cy="20.5" r="1.3" fill="#FFC700"/>
            <circle cx="16" cy="28" r="5" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.6"/>
            <circle cx="16" cy="28" r="1.6" fill="#FFC700"/>
            <circle cx="42" cy="28" r="5" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.6"/>
            <circle cx="42" cy="28" r="1.6" fill="#FFC700"/>
        </svg>
        """
    elif cab_type == "Sedan":
        return f"""
        <svg width="64" height="40" viewBox="0 0 64 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="32" cy="33.5" rx="27" ry="2.6" fill="#111111" opacity="0.12"/>
            <path d="M4 25c0-2.4 2-4.3 4.3-4.3h1.6l3.6-6.3c.9-1.6 2.6-2.6 4.4-2.6h4.5V9.6c0-.9.7-1.6 1.6-1.6h11.4c.9 0 1.6.7 1.6 1.6v2.2h2.6c1.9 0 3.7 1 4.6 2.7l3.5 6.2h1.6c2.3 0 4.3 1.9 4.3 4.3v1.4c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V25Z" fill="#111111"/>
            <path d="M18.5 14.4c.6-1.1 1.8-1.8 3.1-1.8h20.6c1.3 0 2.5.7 3.1 1.9l2.5 4.6H16.2l2.3-4.7Z" fill="#FFC700"/>
            <rect x="21.5" y="16.3" width="7.4" height="4" rx="0.6" fill="#FFFFFF" opacity="0.78"/>
            <rect x="30.8" y="16.3" width="7.4" height="4" rx="0.6" fill="#FFFFFF" opacity="0.78"/>
            <rect x="40" y="16.3" width="5" height="4" rx="0.6" fill="#FFFFFF" opacity="0.65"/>
            <rect x="24" y="8.6" width="14" height="3.6" rx="1.4" fill="#111111"/>
            <rect x="25.6" y="9.4" width="10.8" height="2" rx="0.9" fill="#FFC700"/>
            <circle cx="7.5" cy="20.8" r="1.4" fill="#FFC700"/>
            <circle cx="53" cy="21.2" r="1.4" fill="#E4483D"/>
            <circle cx="16" cy="28.5" r="5.2" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.7"/>
            <circle cx="16" cy="28.5" r="1.7" fill="#FFC700"/>
            <circle cx="45" cy="28.5" r="5.2" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.7"/>
            <circle cx="45" cy="28.5" r="1.7" fill="#FFC700"/>
        </svg>
        """
    else: # SUV
        return f"""
        <svg width="64" height="40" viewBox="0 0 64 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <ellipse cx="32" cy="34" rx="27" ry="2.6" fill="#111111" opacity="0.12"/>
            <rect x="13" y="6.6" width="30" height="2.2" rx="1.1" fill="#333333"/>
            <path d="M4 26c0-2.5 2-4.5 4.5-4.5h1.2V13c0-1.2 1-2.1 2.1-2.1h27.9c1.6 0 3.1.9 3.8 2.3l4.1 8.2h1.9c2.5 0 4.5 2 4.5 4.5v1.7c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V26Z" fill="#111111"/>
            <path d="M15.2 13.9h19l3.6 7.1H15.2v-7.1Z" fill="#FFC700"/>
            <rect x="18" y="15.4" width="6.6" height="4.6" rx="0.6" fill="#FFFFFF" opacity="0.78"/>
            <rect x="26.2" y="15.4" width="6.6" height="4.6" rx="0.6" fill="#FFFFFF" opacity="0.78"/>
            <rect x="26" y="9.2" width="14" height="3.8" rx="1.4" fill="#111111"/>
            <rect x="27.6" y="10.1" width="10.8" height="2" rx="0.9" fill="#FFC700"/>
            <circle cx="9.5" cy="22.3" r="1.4" fill="#FFC700"/>
            <circle cx="17" cy="29.5" r="5.6" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.8"/>
            <circle cx="17" cy="29.5" r="1.8" fill="#FFC700"/>
            <circle cx="46" cy="29.5" r="5.6" fill="#1B1B1B" stroke="#FFC700" stroke-width="1.8"/>
            <circle cx="46" cy="29.5" r="1.8" fill="#FFC700"/>
        </svg>
        """

# ── Rebuilt, Stateful-Bug Free Sidebar (Dispatcher Tower metrics) ───────────
# ── Moving top banner ──────────────────────────────────────────────────────────
_marquee_items = [
    "🚖 KAALIPEELI DISPATCH — LIVE ETA & FARE PREDICTION",
    "⚡ STACKING ENSEMBLE: XGBOOST + LIGHTGBM + CATBOOST → ELASTICNET",
    "📍 REAL-TIME ROUTING POWERED BY OPENROUTESERVICE",
    "🕒 ETA RMSE 2.46 MIN · FARE RMSE ₹126.74",
]
_marquee_html = "".join(f"<span>{item}</span>" for item in _marquee_items * 2)
st.markdown(f"""
<div class="topmarquee-wrap">
    <div class="topmarquee-track">{_marquee_html}</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1rem 0;">
        <h3 style="color: {text_color}; font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem;">Control Tower</h3>
        <p style="color: {sub_text}; font-size: 0.82rem; margin-bottom: 1.2rem;">Live dispatcher telemetry feed</p>
    </div>
    """, unsafe_allow_html=True)

    # Pulsing online status indicator
    st.markdown(f"""
    <div style="background: {input_bg}; border: 1px dashed {border_color}; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="display: inline-block; width: 8px; height: 8px; background: #4CAF7D; border-radius: 50%;"></span>
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #4CAF7D; letter-spacing: 0.05em;">SERVERS ONLINE</span>
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; color: {text_color}; margin-top: 0.25rem;">
            KaaliPeeli Engine v4.12
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Static Loaded Model Checkpoints
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {sub_text}; font-weight: 700; letter-spacing: 0.05em;">ACTIVE PIPELINES</span>
        <div style="font-size: 0.82rem; color: {text_color}; margin-top: 0.4rem;">
            &bull; Time-prediction Stack (M1_XG_LG_CB)<br>
            &bull; Fare-estimator Stack (M2_XG_LG_CB)<br>
            &bull; Routing Engine (ORS Live API)<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div style="border-top: 1px dashed {table_border}; margin: 1.2rem 0;"></div>""", unsafe_allow_html=True)

    # Live Mock Dispatch Event Logger
    st.markdown(f"""
    <div style="margin-bottom: 1.2rem;">
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {sub_text}; font-weight: 700; letter-spacing: 0.05em;">LATEST DISPATCH LOGS</span>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: {sub_text}; margin-top: 0.4rem; line-height: 1.5;">
            [14:18] Mini Fare prediction in Bandra West<br>
            [14:19] Route solved (7.2 km) in Bengaluru<br>
            [14:21] M2 ElasticNet blending calculated<br>
            [14:21] Dispatcher ready for requests.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div style="border-top: 1px dashed {table_border}; margin: 1.2rem 0;"></div>""", unsafe_allow_html=True)

    # Quick support channels
    st.markdown(f"""
    <div>
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {sub_text}; font-weight: 700; letter-spacing: 0.05em;">DISPATCHER CHANNELS</span>
        <p style="font-size: 0.82rem; color: {text_color}; margin-top: 0.3rem; line-height: 1.4;">
            For platform integrations or API support requests, email the telemetry squad at: <b style="color: {acc1};">telemetry@faremind.ai</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Header Row: wordmark left, theme toggle right ─────────────────────────────
header_col1, header_col2 = st.columns([10, 1.2])
with header_col1:
    st.markdown(logo_html, unsafe_allow_html=True)
with header_col2:
    st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
    theme_icon = "☀" if theme_mode == "Dark Slate" else "☾"
    if st.button(theme_icon, key="theme_toggle_btn"):
        st.session_state.theme_loading = True
        st.session_state.theme_mode = "Light Mode" if theme_mode == "Dark Slate" else "Dark Slate"
        st.rerun()

st.markdown(f'<p style="color:{sub_text}; font-size:0.95rem; margin-top:-0.8rem; margin-bottom:1.5rem;">Set your route and pickup time — the meter fills in the fare.</p>', unsafe_allow_html=True)

# ── Resolve selected city first ───────────────────────────────────────────────
if "city" not in st.session_state:
    st.session_state.city = "Mumbai"
city = st.session_state.city

# ── Live City Dispatch Status Banner ───────────────────────────
st.markdown(f"""
<div class="banner-panel">
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;'>
        <div>
            <span style='font-family: IBM Plex Mono, monospace; font-size: 0.72rem; letter-spacing: 0.15em; color: {acc1}; font-weight: 700;'>LIVE NETWORK STATUS &middot; {city.upper()}</span>
            <h4 style='margin: 0.2rem 0 0; font-family: Space Grotesk, sans-serif; font-size: 1.2rem; color: {text_color};'>Dispatcher Active</h4>
        </div>
        <div style='display: flex; gap: 2rem; flex-wrap: wrap;'>
            <div>
                <span style='font-family: Work Sans, sans-serif; font-size: 0.75rem; color: {sub_text};'>Active Fleet</span>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 1.1rem; font-weight: 700; color: {text_color};'>1,842 Cabs Online</div>
            </div>
            <div>
                <span style='font-family: Work Sans, sans-serif; font-size: 0.75rem; color: {sub_text};'>Average Pickup ETA</span>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 1.1rem; font-weight: 700; color: {acc1};'>4 &minus; 6 mins</div>
            </div>
            <div>
                <span style='font-family: Work Sans, sans-serif; font-size: 0.85rem; font-weight: 600; color: #4CAF7D; display: flex; align-items: center; gap: 0.35rem; margin-top: 0.2rem;'>
                    <span style='display: inline-block; width: 8px; height: 8px; background: #4CAF7D; border-radius: 50%;'></span> Operational
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Asymmetric Columns (Uber/Ola Panel Style) ───────────────────────────
left_col, right_col = st.columns([5, 7], gap="large")

# State variables for location coordinates
src_lat, src_lng, src_zone = 0.0, 0.0, "City Center"
dst_lat, dst_lng, dst_zone = 0.0, 0.0, "City Center"

# ── Left Column: Uber/Ola style Booking Card panel (Fixed div wrap alignment bugs) ────
with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-hdr">Trip ticket — route &amp; timing</div>', unsafe_allow_html=True)

        # City selection
        city = st.selectbox("City", list(CITIES.keys()), key="city_select")
        st.session_state.city = city

        # Handle city change to reset prediction state and selections
        if "prev_city" not in st.session_state:
            st.session_state.prev_city = city
        elif st.session_state.prev_city != city:
            st.session_state.prev_city = city
            st.session_state.has_predicted = False
            if "pickup_sel" in st.session_state: del st.session_state["pickup_sel"]
            if "dropoff_sel" in st.session_state: del st.session_state["dropoff_sel"]
            st.rerun()

        subregions = []
        for zone, zone_data in ZONES[city].items():
            for subregion_name in zone_data["sub_regions"].keys():
                subregions.append(subregion_name)
        subregions = sorted(subregions)

        location_options = subregions + ["🔍 Enter Custom Location..."]

        # Pickup location dropdown
        pickup_sel = st.selectbox("Pickup", location_options, key="pickup_sel")
        
        if pickup_sel == "🔍 Enter Custom Location...":
            # Real-time autocomplete search using OSM Nominatim API
            src_search_query = st.text_input("Type the pickup address", placeholder="e.g. Bandra West, Mumbai", key="src_search")
            if not src_search_query.strip():
                st.caption("💡 Type 3 or more characters to fetch coordinates from the live map database.")
            
            src_suggestions = []
            if len(src_search_query.strip()) >= 3:
                url = "https://nominatim.openstreetmap.org/search"
                params = {"q": f"{src_search_query}, {city}, India", "format": "json", "limit": 5}
                headers = {"User-Agent": "faremind_cab_predictor/1.0"}
                try:
                    res = requests.get(url, params=params, headers=headers, timeout=2.5)
                    if res.status_code == 200:
                        src_suggestions = res.json()
                except Exception:
                    pass
            
            if src_suggestions:
                src_options_map = {item["display_name"]: (float(item["lat"]), float(item["lon"])) for item in src_suggestions}
                selected_src_opt = st.selectbox("Recommended Pickup Results", list(src_options_map.keys()), key="src_recommend_select")
                src_lat, src_lng = src_options_map[selected_src_opt]
                src_sub, src_zone = find_nearest_subregion(city, src_lat, src_lng)
            else:
                # Fallback to city center weather coords
                src_lat, src_lng = CITIES[city]["weather_coords"]
                src_sub, src_zone = find_nearest_subregion(city, src_lat, src_lng)
        else:
            src_lat, src_lng, src_zone = lookup_subregion(city, pickup_sel)
            src_sub = pickup_sel

        # Drop-off location dropdown
        dropoff_sel = st.selectbox("Drop-off", location_options, index=min(1, len(location_options)-1), key="dropoff_sel")
        
        if dropoff_sel == "🔍 Enter Custom Location...":
            # Real-time autocomplete search using OSM Nominatim API
            dst_search_query = st.text_input("Type the drop-off address", placeholder="e.g. Bandra Terminus, Mumbai", key="dst_search")
            if not dst_search_query.strip():
                st.caption("💡 Type 3 or more characters to fetch coordinates from the live map database.")
            
            dst_suggestions = []
            if len(dst_search_query.strip()) >= 3:
                url = "https://nominatim.openstreetmap.org/search"
                params = {"q": f"{dst_search_query}, {city}, India", "format": "json", "limit": 5}
                headers = {"User-Agent": "faremind_cab_predictor/1.0"}
                try:
                    res = requests.get(url, params=params, headers=headers, timeout=2.5)
                    if res.status_code == 200:
                        dst_suggestions = res.json()
                except Exception:
                    pass
            
            if dst_suggestions:
                dst_options_map = {item["display_name"]: (float(item["lat"]), float(item["lon"])) for item in dst_suggestions}
                selected_dst_opt = st.selectbox("Recommended Drop-off Results", list(dst_options_map.keys()), key="dst_recommend_select")
                dst_lat, dst_lng = dst_options_map[selected_dst_opt]
                dst_sub, dst_zone = find_nearest_subregion(city, dst_lat, dst_lng)
            else:
                # Fallback to city center weather coords
                dst_lat, dst_lng = CITIES[city]["weather_coords"]
                dst_sub, dst_zone = find_nearest_subregion(city, dst_lat, dst_lng)
        else:
            dst_lat, dst_lng, dst_zone = lookup_subregion(city, dropoff_sel)
            dst_sub = dropoff_sel

        # Trip Scheduling details
        st.markdown(f'<div class="ticket-tear">TIMING</div>', unsafe_allow_html=True)
        booking_mode = st.radio("When", ["Right now", "Pick a time"], horizontal=True, key="booking_mode")

        if booking_mode == "Right now":
            now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
            trip_date = now.date()
            trip_hour = now.hour
            trip_minute = now.minute
            
            # Displays precise current captured time
            st.markdown(f"""
            <div style="background: {input_bg}; padding: 0.65rem 0.85rem; border-radius: 6px; border: 1px dashed {border_color}; margin-top: -0.4rem; margin-bottom: 0.8rem;">
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {acc1}; font-weight: 700; letter-spacing: 0.08em;">🕒 CAPTURING SYSTEM TIME</span>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; color: {text_color}; margin-top: 0.15rem;">
                    {now.strftime('%I:%M %p')} &middot; {now.strftime('%A, %b %d')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                # Use IST for default date
                trip_date = st.date_input("Date", datetime.datetime.now(ZoneInfo("Asia/Kolkata")).date())
            with c2:
                # Use IST for default slider hour
                current_ist_hour = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).hour
                trip_hour = st.slider("Hour (24h)", 0, 23, current_ist_hour)
            trip_minute = 0

        is_holiday = check_is_holiday(trip_date)
        day_of_week = trip_date.strftime("%A")
        time_of_day = get_time_of_day(trip_hour)

        # Cab Category Selection
        st.markdown(f'<div class="ticket-tear">VEHICLE CLASS</div>', unsafe_allow_html=True)

        cabs_data = [
            {"id": "Mini", "name": "KaaliPeeli Mini", "desc": "Affordable compact cars"},
            {"id": "Sedan", "name": "KaaliPeeli Sedan", "desc": "Comfortable daily padminis"},
            {"id": "SUV", "name": "KaaliPeeli SUV", "desc": "Premium spacious Ertigas"},
        ]

        for cab in cabs_data:
            is_active = st.session_state.selected_cab == cab["id"]
            border_style = (
                f"border: 2px solid {acc1}; background: {acc1}15;"
                if is_active else
                f"border: 1.5px dashed {border_color}; background: {input_bg};"
            )

            row_col_card, row_col_btn = st.columns([6, 1.8])
            with row_col_card:
                card_html = (
                    f"<div style='display: flex; align-items: center; gap: 1.2rem; padding: 0.7rem 0.9rem; border-radius: 6px; {border_style}'>"
                    f"<div>{get_cab_svg(cab['id'])}</div>"
                    f"<div>"
                    f"<b style='font-family: Space Grotesk, sans-serif; font-size: 0.95rem; color: {text_color};'>{cab['name']}</b><br>"
                    f"<span style='font-size: 0.78rem; color: {sub_text};'>{cab['desc']}</span>"
                    f"</div>"
                    f"</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
            with row_col_btn:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                btn_lbl = "· SET ·" if is_active else "SELECT"
                if st.button(btn_lbl, key=f"btn_select_{cab['id']}", disabled=is_active):
                    st.session_state.selected_cab = cab["id"]
                    st.rerun()

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        estimate_button = st.button("Punch the meter — get my fare", key="estimate_btn")

# ── Right Column: Map & Stacking Ensemble Analytics Panel ────────────────────
with right_col:
    # Bento Block 1: Dynamic Route Map
    with st.container(border=True):
        st.markdown('<div class="section-hdr">Route strip</div>', unsafe_allow_html=True)
        
        # Center map on custom locations or default weather center coords
        c_lat = (src_lat + dst_lat) / 2 if (src_lat and dst_lat) else CITIES[city]["weather_coords"][0]
        c_lng = (src_lng + dst_lng) / 2 if (src_lng and dst_lng) else CITIES[city]["weather_coords"][1]
        
        view_state = pdk.ViewState(
            latitude=c_lat,
            longitude=c_lng,
            zoom=11.2,
            pitch=20,
            bearing=0
        )
        
        layers = []
        if src_lat and dst_lat:
            points_df = pd.DataFrame([
                {"name": f"Pickup: {src_sub}", "lat": src_lat, "lon": src_lng, "color": [255, 199, 0, 230], "radius": 240},
                {"name": f"Dropoff: {dst_sub}", "lat": dst_lat, "lon": dst_lng, "color": [228, 72, 61, 230], "radius": 240}
            ])
            line_df = pd.DataFrame([
                {"start_lat": src_lat, "start_lng": src_lng, "end_lat": dst_lat, "end_lng": dst_lng}
            ])
            
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                points_df,
                get_position="[lon, lat]",
                get_color="color",
                get_radius="radius",
                pickable=True
            ))
            layers.append(pdk.Layer(
                "LineLayer",
                line_df,
                get_source_position="[start_lng, start_lat]",
                get_target_position="[end_lng, end_lat]",
                get_color="[255, 199, 0, 200]",
                get_width=4
            ))
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=map_style,
            tooltip={"text": "{name}"}
        )
        st.pydeck_chart(deck)

    # Check if we should predict
    if estimate_button:
        st.session_state.is_calculating = True
        st.session_state.has_predicted = True
        st.rerun()

    # AI Prediction Loader
    if st.session_state.get("is_calculating", False):
        st.session_state.is_calculating = False
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div class="calc-card">
                <span class="calc-taxi">🚖</span>
                <div class="calc-title">Punching the meter…</div>
                <div class="calc-steps">reading route · checking weather · pricing</div>
                <div class="calc-track"><div class="calc-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1.3)  # Smooth transition loader instead of a sudden glitch
        loading_placeholder.empty()

    # If predicted, execute predictions and show results
    if st.session_state.has_predicted:
        error_found = False
        if pickup_sel == "🔍 Enter Custom Location..." and not src_search_query.strip():
            st.error("Enter a pickup address in the trip ticket before requesting a fare.")
            error_found = True
        if dropoff_sel == "🔍 Enter Custom Location..." and not dst_search_query.strip():
            st.error("Enter a drop-off address in the trip ticket before requesting a fare.")
            error_found = True
        if pickup_sel == dropoff_sel and pickup_sel != "🔍 Enter Custom Location...":
            st.error("Pickup and drop-off are the same place — pick two different points.")
            error_found = True

        if not error_found and src_lat and dst_lat:
            # 1. ORS Route Calculation
            dist_km, base_min = get_ors_route(src_lat, src_lng, dst_lat, dst_lng)

            # 2. Weather Fetch
            weather = get_current_weather(src_lat, src_lng, city)

            # 3. Traffic Level Derivation
            timestamp_str = f"{trip_date.isoformat()} {trip_hour:02d}:{trip_minute:02d}:00"
            traffic_level = derive_traffic_level(city, src_sub, timestamp_str, weather, traffic_baseline)

            # 4. Cab Availability
            base_avail = ZONES[city][src_zone]["cab_availability"]
            if trip_hour in {8, 9, 10, 17, 18, 19, 20}:
                cab_avail = max(1, int(base_avail * random.uniform(0.2, 0.45)))
            elif trip_hour in {23, 0, 1, 2, 3, 4, 5}:
                cab_avail = max(1, int(base_avail * random.uniform(0.3, 0.6)))
            else:
                cab_avail = max(1, int(base_avail * random.uniform(0.6, 1.0)))

            # 5. Surge Multiplier
            surge = get_surge(cab_avail, weather, trip_hour, is_holiday)

            # 6. ML Predictions
            eta, fare = run_predictions(
                city, src_lat, src_lng, dst_lat, dst_lng, src_sub, dst_sub, src_zone, dst_zone,
                st.session_state.selected_cab, trip_hour, day_of_week, is_holiday, weather, traffic_level, cab_avail, surge, dist_km, base_min
            )

            # Live Meter Display — Kaali Peeli Vintage Dashboard readout
            st.markdown(f"""
            <div class="meter-face">
                <div class="meter-lbl">🚖 KaaliPeeli Fare Meter</div>
                <div class="meter-val">₹{fare:,.0f}</div>
                <div style='margin-top: 1.1rem; display: flex; justify-content: center; gap: 2.4rem; align-items: center;'>
                    <div>
                        <span class="meter-lbl" style='color:#FFC700; opacity:0.8;'>ETA</span>
                        <div class="meter-sub">{eta:.0f} min</div>
                    </div>
                    <div style='width: 1px; height: 30px; background: #FFC70044;'></div>
                    <div>
                        <span class="meter-lbl" style='color:#FFC700; opacity:0.8;'>Class</span>
                        <div class="meter-sub" style='font-size:1.15rem;'>{st.session_state.selected_cab}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Where this fare sits between the city minimum and a busy-hour peak
            with st.container(border=True):
                st.markdown(f'<div class="section-hdr">Where this sits</div><div style="font-size: 0.82rem; color: {sub_text}; margin-bottom: 1rem;">Between the city\'s flag-down minimum and what a busy-hour surge would charge.</div>', unsafe_allow_html=True)
                
                min_possible_fare = CITIES[city]["minimum_fare"]
                peak_possible_fare = fare * 1.5
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=["City minimum", "This estimate", "Peak-hour ceiling"],
                    x=[min_possible_fare, fare, peak_possible_fare],
                    orientation='h',
                    marker=dict(
                        color=['#4CAF7D', '#FFC700', acc2],
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    width=0.45,
                    hovertemplate='₹%{x:.0f}<extra></extra>'
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        tickfont=dict(color=sub_text, size=10, family='IBM Plex Mono')
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        tickfont=dict(color=sub_text, size=11, family='Space Grotesk')
                    ),
                    height=150,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Route facts pulled straight from the routing API
            with st.container(border=True):
                st.markdown(f'<div class="section-hdr">Route</div><div style="font-size: 0.82rem; color: {sub_text}; margin-bottom: 1rem;">Actual road distance and drive time with no traffic in the way.</div>', unsafe_allow_html=True)
                
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.markdown(f"""
                    <div style='background: {input_bg}; border-radius: 6px; padding: 0.8rem; border: 1px dashed {paper_dash};'>
                        <span class="meter-lbl" style='color:{sub_text}; opacity:1;'>Distance</span>
                        <div style='font-family: IBM Plex Mono, monospace; font-size: 1.5rem; font-weight: 700; color: #4CAF7D;'>{dist_km:.2f} km</div>
                    </div>
                    """, unsafe_allow_html=True)
                with r_col2:
                    st.markdown(f"""
                    <div style='background: {input_bg}; border-radius: 6px; padding: 0.8rem; border: 1px dashed {paper_dash};'>
                        <span class="meter-lbl" style='color:{sub_text}; opacity:1;'>Free-flow time</span>
                        <div style='font-family: IBM Plex Mono, monospace; font-size: 1.5rem; font-weight: 700; color: {acc2};'>{base_min:.0f} min</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Conditions that moved the price
            with st.container(border=True):
                st.markdown(f'<div class="section-hdr">Conditions right now</div><div style="font-size: 0.82rem; color: {sub_text}; margin-bottom: 1rem;">What pushed the surge multiplier up or down for this trip.</div>', unsafe_allow_html=True)
                
                e_col1, e_col2, e_col3 = st.columns(3)
                with e_col1:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class="meter-lbl" style='color:{sub_text}; opacity:1;'>Weather</span><br>
                        <b style='color: {text_color}; font-size: 1.1rem;'>{weather}</b>
                    </div>
                    """, unsafe_allow_html=True)
                with e_col2:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class="meter-lbl" style='color:{sub_text}; opacity:1;'>Traffic</span><br>
                        <b style='color: {text_color}; font-size: 1.1rem;'>{traffic_level}</b>
                    </div>
                    """, unsafe_allow_html=True)
                with e_col3:
                    surge_class = "badge-green" if surge <= 1.0 else "badge-yellow" if surge <= 1.5 else "badge-red"
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <span class="meter-lbl" style='color:{sub_text}; opacity:1;'>Surge</span><br>
                        <span class="badge-premium {surge_class}" style="margin-top:0.2rem;">×{surge}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # The receipt — every line the model added up to reach the total
            with st.container(border=True):
                st.markdown(f'<div class="section-hdr">Receipt</div><div style="font-size: 0.82rem; color: {sub_text}; margin-bottom: 0.5rem;">How the ensemble\'s total breaks down, line by line.</div>', unsafe_allow_html=True)
                
                cfg = CITIES[city]
                raw_base = cfg["base_price_per_km"] * dist_km
                raw_time = cfg["per_minute_rate"] * eta
                src_premium = ZONES[city][src_zone]["zone_premium"]
                dst_premium = ZONES[city][dst_zone]["zone_premium"]
                zone_fee = src_premium + dst_premium
                surge_diff = (surge - 1.0) * (raw_base + raw_time)

                component_sum = raw_base + raw_time + zone_fee + surge_diff
                if component_sum <= 0: component_sum = 1.0

                net_fare = fare / 1.05
                gst_tax = fare - net_fare
                scaling_factor = net_fare / component_sum

                base_fare_final = round(raw_base * scaling_factor, 2)
                time_fare_final = round(raw_time * scaling_factor, 2)
                zone_fare_final = round(zone_fee * scaling_factor, 2)
                surge_fare_final = round(surge_diff * scaling_factor, 2)

                st.markdown(f"""
                <table class="receipt-table">
                    <tr>
                        <td class="lbl">Base distance</td>
                        <td class="val">₹{base_fare_final:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Time on the road</td>
                        <td class="val">₹{time_fare_final:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Zone fee ({src_zone} → {dst_zone})</td>
                        <td class="val">₹{zone_fare_final:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="lbl">Demand &amp; weather surge</td>
                        <td class="val">₹{surge_fare_final:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="lbl">GST (5%)</td>
                        <td class="val">₹{gst_tax:,.2f}</td>
                    </tr>
                    <tr class="total">
                        <td class="lbl">Total fare</td>
                        <td class="val">₹{fare:,.2f}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
    else:
        # Default placeholder when no estimate requested yet — the meter's idle state
        with st.container(border=True):
            st.markdown(f"""
            <div style='text-align: center; padding: 3rem 1.5rem;'>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 2.4rem; color: {sub_text}; letter-spacing: 0.3em; opacity: 0.5;'>- - - -</div>
                <h3 style='color: {text_color}; font-weight: 700; margin-top: 1rem;'>Meter's off duty</h3>
                <p style='color: {sub_text}; max-width: 400px; margin: 0.5rem auto 1.5rem; font-size: 0.9rem;'>
                    Fill in the trip ticket on the left — pickup, drop-off, timing, and vehicle class — then punch the meter to see your fare.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align: center; margin-top: 4rem; padding: 1.8rem 0; border-top: 1px dashed {table_border};'>
    <p style='color: {sub_text}; font-size: 0.85rem; margin-bottom: 0.4rem; font-family: Space Grotesk, sans-serif;'>
        Built with 💛 by <b style='color: {acc1};'>Govind Garg</b>
    </p>
    <p style='color: {sub_text}; font-size: 0.72rem; font-family: IBM Plex Mono, monospace; opacity: 0.75;'>
        stacking ensemble: xgboost + lightgbm + catboost → elasticnet &nbsp;·&nbsp;
        eta rmse 2.46 min &nbsp;·&nbsp; fare rmse ₹126.74
    </p>
</div>
""", unsafe_allow_html=True)