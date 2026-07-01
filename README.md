# 🚖 KaaliPeeli — Cab Price Predictor

An end-to-end machine learning system that predicts **cab fare** and **ETA** for rides across 5 Indian cities, built on a fully custom data pipeline (real routing, real historical weather, real traffic baselines) and a **stacking ensemble** of gradient-boosted models. Served through an interactive Streamlit dashboard styled after Mumbai's iconic Kaali-Peeli taxi meters.

| Metric | Value |
|---|---|
| ETA model RMSE | 2.46 min |
| Fare model RMSE | ₹126.74 |
| Cities covered | Mumbai, Delhi, Bengaluru, Hyderabad, Chandigarh |
| Dataset size | 1,887 real routed coordinate pairs × 5 variations ≈ 9,435 rows |
| Base learners | XGBoost, LightGBM, CatBoost |
| Meta-learner | Ridge / Lasso / Elastic Net(best selected via CV) |

---

## Why this project exists

Most "cab fare predictor" tutorials train on a single scraped CSV with no real-world grounding. This project instead builds the **entire data layer from scratch** using live APIs — real road distances, real historical weather, real traffic baselines — and formula-drives the fare/time targets the way an actual dispatch platform would, before training a genuine stacking ensemble on top of it. The goal was to practice the full ML lifecycle end-to-end, not just model fitting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  route_fetcher.py     → OpenRouteService: real distance/time     │
│                          for 1,887 coordinate pairs (resumable)  │
│  weather_fetcher.py   → Open-Meteo Archive API: 4 weeks of real  │
│                          hourly weather per city (WMO code map)  │
│  traffic_fetcher.py   → TomTom Flow API: freeFlowSpeed baseline  │
│                          per sub-region (~76 points, city-       │
│                          normalized congestion ratio)            │
│  holiday_fetcher.py   → 2026 holiday calendar per city           │
│  fare_calculator.py   → Formula-driven ETA/fare/surge generation │
│                          (traffic × weather × time-of-day)       │
│  pipeline.py          → Orchestrates all of the above into       │
│                          cab_dataset.csv                          │
├─────────────────────────────────────────────────────────────────┤
│                          PREPROCESSING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  preprocessing.py      → Data-driven feature selection (RF       │
│                          importance check before dropping any    │
│                          column) + three parallel feature sets:  │
│                                                                    │
│    XGBoost   → ColumnTransformer: StandardScaler (numeric) +     │
│                OrdinalEncoder (traffic/time/cab_type) +          │
│                OneHotEncoder (city/zone/weather/day) +           │
│                custom FrequencyEncoder (sub-region, if kept)     │
│    LightGBM  → passthrough numerics + integer-coded categoricals │
│                (native categorical_feature handling)             │
│    CatBoost  → raw categorical strings, untouched (its strength) │
├─────────────────────────────────────────────────────────────────┤
│                            MODELING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  train_models.py       → Optuna-tuned XGBoost + LightGBM +       │
│                          CatBoost base learners → OOF predictions │
│                          → Ridge/Lasso meta-learner (stacking)    │
│                                                                    │
│    Model 1 (ETA)   trains first                                  │
│         │  hands off OOF + test predictions                      │
│         ▼                                                         │
│    Model 2 (Fare)  trains on Model 1's *predicted* ETA,          │
│                     not ground truth — matches real inference    │
├─────────────────────────────────────────────────────────────────┤
│                          APPLICATION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  app/app.py             → Streamlit dashboard: route map, live   │
│                           fare meter, receipt breakdown, surge   │
│                           conditions, theme toggle                │
└─────────────────────────────────────────────────────────────────┘
```

---

## The data pipeline, in detail

### 1. Routes — real road distances (`route_fetcher.py`, `topup_routes.py`)
1,887–2,000 coordinate pairs across 5 cities, each routed through **OpenRouteService** for genuine driving distance and time (not straight-line estimates). The fetcher is fully resumable — it checkpoints progress and tops up shortfalls without re-spending quota on already-fetched routes, since ORS's free tier caps at 2,000 calls/day.

### 2. Weather — real historical data, not a thin forecast (`weather_fetcher.py`)
Started with OpenWeatherMap's free forecast endpoint, which only returns a forward-looking 5-day window — meaning some days of the week would have had **zero real weather coverage**, breaking day-of-week signal for the model. Switched to **Open-Meteo's free historical Archive API**, pulling the last 4 weeks of real hourly weather per city (5 calls total, no key required). WMO weather codes are mapped to 5 categories (Sunny / Cloudy / Rainy / Foggy / Stormy), with a probabilistic city-bias fallback for any gaps. The 4-week window was deliberately kept inside the 2026 holiday-calendar range so `is_holiday` stays correctly calibrated against it.

### 3. Traffic — a real structural baseline, not a live snapshot (`traffic_fetcher.py`)
Queries **TomTom's Flow API** across ~76 named sub-regions (not just one point per city) and uses `freeFlowSpeed` — the road's ideal-conditions speed — as the structural baseline, rather than `currentSpeed`, which would just be a live snapshot baked in as permanent data. Each sub-region's speed is normalized against its city average into a `relative_speed_factor`, so genuinely different areas (e.g. a dense railway station vs. a suburb) produce meaningfully different congestion ratios under identical time/weather conditions — a divergence bug caught and fixed during development.

### 4. Fare & ETA formula (`fare_calculator.py`)
`estimated_time_min` and `fare_amount` are generated from a transparent formula layering traffic multiplier, surge (bounded to a real MoRTH-style 1.0–2.0x cap), zone premiums, and noise — not a black box. A double-counting bug (where time-of-day and weather were being applied both directly *and* indirectly through `traffic_level`, compounding to unrealistic 4x time inflation) was caught and fixed, bringing worst-case estimates down to a realistic ~1.5–1.7x base time.

### 5. Orchestration (`pipeline.py`)
Ties every module together: for each route, generates 5 randomized (timestamp × cab type) variations, resolves weather/traffic/holiday context for that timestamp, and computes the final row — producing **9,435 rows** from 1,887 base routes.

---

## Exploratory Data Analysis (`eda/eda.ipynb`)

Built around two goals: proving the synthetic data is realistic, and surfacing a genuine business insight.

**Sanity checks:** distribution shape for fare/time/distance, surge correctly bounded to [1.0, 2.0], fare never below city minimums, distance-vs-fare and traffic-vs-time relationships hold as expected.

**Business breakdowns:** average fare by city, time-of-day, weather, cab type, and holiday status — confirming the formula-driven pricing tiers and surge premiums show up cleanly in the generated data.

**Correlation heatmap** (used instead of a confusion matrix, which only applies to classification — this is a regression problem) to confirm `distance_km` and `estimated_time_min` are the dominant fare drivers and check for redundant features.

**Supply-optimization insight:** grouping by city + zone to compare average `cab_availability` against average `surge_multiplier` surfaces zones where riders are consistently paying a scarcity premium rather than a distance-driven one. The comparison is done **within each city** (not against a global average) to avoid a cross-city Simpson's-paradox trap, since base pricing differs structurally by city tier. Business takeaway: deploying additional cabs in specific zone/time windows (e.g. Airport zones during evening hours) could let the platform undercut competitor surge pricing while preserving margin — a targeted supply recommendation rather than a blanket fleet increase.

---

## Preprocessing (`preprocessing/preprocessing.py`, `custom_transformers.py`)

- **Data-driven feature selection** — a baseline Random Forest importance check runs against both targets *before* any column is dropped, rather than dropping lat/lng or sub-region on assumption. On the real dataset, longitude was kept (carries meaningful signal across the 5 cities) while latitude and raw sub-region were dropped as low-importance.
- **Mixed encoding, matched to actual structure** — `traffic_level`, `time_of_day`, and `cab_type` are ordinal-encoded (they have genuine inherent order); `city`, `zone`, `weather`, `day_of_week` are one-hot encoded (no meaningful magnitude between categories).
- **Three parallel feature sets** — because CatBoost wants raw categoricals, LightGBM wants integer-coded categoricals with `categorical_feature` indices, and XGBoost needs fully encoded/scaled numeric input, `preprocessing.py` builds all three from the same source rows via a single `ColumnTransformer` + `Pipeline` per model type — no manual multi-step encoding, no column-order drift between train/test.
- **Custom `FrequencyEncoder`** — a proper `BaseEstimator`/`TransformerMixin` implementation for sub-region frequency encoding, built specifically to fix a leakage bug: the original version computed frequencies on the full dataset before the train/test split. As a standalone transformer inside the pipeline, it now only ever fits on the training fold — plus it's pulled into a stable, never-run-as-`__main__` module so `joblib`/`pickle` can always resolve the class on load.
- **Full dataset shuffle** before splitting, since rows were generated route-by-route sequentially and could otherwise carry subtle ordering artifacts.

---

## Modeling (`models/train_models.py`)

**Stacking ensemble:** XGBoost + LightGBM + CatBoost as base learners, each hyperparameter-tuned with **Optuna**, scored via **cross-validated RMSE** (`cross_val_score` for XGBoost/LightGBM; a manual K-Fold loop for CatBoost, which doesn't clone cleanly through sklearn's CV utilities when `cat_features` is set in the constructor). Out-of-fold predictions from all three feed a **Ridge/Lasso meta-learner**, with the better-performing one selected automatically. Final base-learner and stacked-ensemble RMSEs are compared directly to confirm stacking actually helps.

**Two chained models, handled correctly:** the ETA model (Model 1) and fare model (Model 2) aren't independent — the real fare formula depends on ride duration. Training Model 2 on *ground-truth* ETA would look great on paper but silently break at inference, since deployment only ever has Model 1's *predicted* ETA available. The pipeline trains Model 1 first, then explicitly overwrites the `estimated_time_min` feature in Model 2's training and test sets with Model 1's out-of-fold and test predictions respectively — so Model 2 learns to work with the same noisy, imperfect ETA signal it will actually receive in production.

---

## Project structure

```
CAB_PRICE_PREDICTOR/
├── app/
│   └── app.py                   # Streamlit dashboard (UI + inference)
├── config/
│   └── config.py                # Cities, zones, cab types, API keys, traffic/weather constants
├── data/
│   ├── raw/                     # Route/weather/traffic API outputs
│   └── processed/
│       ├── cab_dataset.csv      # Final generated dataset (tracked in git)
│       ├── model_ready/         # Preprocessed train/test artifacts (gitignored)
│       └── trained_models/      # Saved model artifacts (gitignored)
├── data_collection/
│   ├── fare_calculator.py
│   ├── holiday_fetcher.py
│   ├── pipeline.py
│   ├── route_fetcher.py
│   ├── topup_routes.py
│   ├── traffic_fetcher.py
│   └── weather_fetcher.py
├── eda/
│   └── eda.ipynb
├── models/
│   └── train_models.py
├── preprocessing/
│   └── preprocessing.py
├── custom_transformers.py       # FrequencyEncoder — stable import path for joblib
├── requirements.txt
├── .env                         # API keys (gitignored)
└── .gitignore
```

---

## Getting started

### 1. Clone and set up the environment
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd CAB_PRICE_PREDICTOR
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys
Create a `.env` file in the project root:
```
ORS_API_KEY=your_openrouteservice_key
TOMTOM_API_KEY=your_tomtom_key
```
(Open-Meteo needs no key.)

### 3. Rebuild the dataset (optional — `cab_dataset.csv` is already tracked)
```bash
python3 data_collection/route_fetcher.py
python3 data_collection/weather_fetcher.py
python3 data_collection/traffic_fetcher.py
python3 data_collection/pipeline.py
```

### 4. Preprocess and train
```bash
python3 preprocessing/preprocessing.py
python3 models/train_models.py
```

### 5. Run the app
```bash
cd app
streamlit run app.py
```

---

## Tech stack

- **Data collection:** OpenRouteService, Open-Meteo Archive API, TomTom Flow API
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **App:** Streamlit, Plotly, PyDeck

## Known limitations

- `cab_availability` is a static zone-level config value, not a live signal — the supply-optimization insight is structural, not real-time dispatch.
- Sub-region traffic baselines use a single representative coordinate's `freeFlowSpeed` as a proxy for an entire named area, which can occasionally produce counter-intuitive results (e.g. a well-connected arterial road near a busy station reading as lower-congestion than expected).
- Model 2 is trained on Model 1's OOF/test predictions specifically to match production behavior — but this also means Model 2's accuracy is inherently capped by Model 1's, by design.

---

Built with 💛 by Govind Garg
