# =============================================================================
# preprocessing.py — Build Train/Test Splits for XGBoost + LightGBM + CatBoost
#
# Pipeline:
#   1. Load + shuffle full dataset (eliminates route-by-route generation order)
#   2. Quick Random Forest importance check on ALL candidate columns
#      (including subregion/lat-lng) BEFORE deciding what to drop
#   3. Build feature sets for Model 1 (time) and Model 2 (fare)
#   4. Encode + scale:
#        - XGBoost: ordinal encoding for columns with real order
#                    (traffic_level, time_of_day, cab_type),
#                    one-hot for the rest (city, zones, weather, day_of_week),
#                    StandardScaler on numeric features
#        - LightGBM: passthrough for numerics (no scaling), OrdinalEncoder
#                    for all nominal/ordinal categoricals, and FrequencyEncoder
#                    for any kept subregion columns
#        - CatBoost: RAW categorical strings, untouched — manually encoding
#                    anything here throws away CatBoost's native (better)
#                    categorical handling, so this path ignores the
#                    ordinal/one-hot choice entirely
#   5. Save train/test artifacts per model
#
# Usage:
#   python preprocessing/preprocessing.py
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cab_dataset.csv')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_ready')

TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_TIME = "estimated_time_min"
TARGET_FARE = "fare_amount"

# Always-drop: pure identifiers / fully redundant with engineered columns.
# (timestamp is fully decomposed into hour/day_of_week/time_of_day/is_holiday already)
ALWAYS_DROP = ["route_id", "timestamp"]

# Candidate columns we're UNSURE about — decided via importance check, not assumption
CANDIDATE_DROP = ["source_lat", "source_lng", "dest_lat", "dest_lng",
                   "source_subregion", "dest_subregion"]

NUMERIC_FEATURES = [
    "distance_km", "base_time_min", "hour", "is_holiday",
    "cab_availability", "surge_multiplier",
    "base_price_per_km", "per_minute_rate", "city_tier",
]

# Columns with a genuine inherent order -> ordinal encoding (XGBoost path only)
ORDINAL_FEATURES = {
    "traffic_level": ["Low", "Medium", "High"],
    "time_of_day": ["Morning", "Afternoon", "Evening", "Night"],
    "cab_type": ["Mini", "Sedan", "SUV"],
}

# Columns with no meaningful magnitude -> one-hot encoding (XGBoost path only)
ONEHOT_FEATURES = ["city", "source_zone", "dest_zone", "weather_condition", "day_of_week"]

CATEGORICAL_FEATURES = list(ORDINAL_FEATURES.keys()) + ONEHOT_FEATURES  # used as-is for CatBoost


# =============================================================================
# CUSTOM TRANSFORMER — Frequency Encoder for high-cardinality subregion columns
# =============================================================================

from custom_transformers import FrequencyEncoder  # noqa: F401 (re-exported for convenience)


# =============================================================================
# LOAD + SHUFFLE
# =============================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ALWAYS_DROP if c in df.columns])

    # Explicit full-dataset shuffle — rows were generated route-by-route
    # sequentially, so this removes any residual ordering artifacts before
    # train_test_split's own (separate) shuffling happens
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"📂 Loaded + shuffled dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# FEATURE IMPORTANCE CHECK
# =============================================================================

def check_candidate_feature_importance(df: pd.DataFrame, target_col: str) -> pd.Series:
    temp = df.copy()

    for col in ["source_subregion", "dest_subregion"]:
        freq_map = temp[col].value_counts(normalize=True)
        temp[col + "_freq"] = temp[col].map(freq_map)

    check_cols = NUMERIC_FEATURES + ["source_lat", "source_lng", "dest_lat", "dest_lng",
                                      "source_subregion_freq", "dest_subregion_freq"]
    check_cols = [c for c in check_cols if c != target_col]

    X_check = temp[check_cols]
    y_check = temp[target_col]

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_check, y_check)

    importances = pd.Series(rf.feature_importances_, index=check_cols).sort_values(ascending=False)
    return importances


def decide_candidate_drops(df: pd.DataFrame, importance_threshold: float = 0.01) -> list:
    print("\n🔍 Checking importance of candidate columns (subregion, lat/lng) before dropping...")

    imp_time = check_candidate_feature_importance(df, TARGET_TIME)
    imp_fare = check_candidate_feature_importance(df, TARGET_FARE)

    print("\n  Importance for estimated_time_min:")
    print(imp_time.to_string())
    print("\n  Importance for fare_amount:")
    print(imp_fare.to_string())

    candidates = {
        "source_lat": ["source_lat"], "source_lng": ["source_lng"],
        "dest_lat": ["dest_lat"], "dest_lng": ["dest_lng"],
        "source_subregion": ["source_subregion_freq"], "dest_subregion": ["dest_subregion_freq"],
    }

    to_drop = []
    print("\n  Decision:")
    for original_col, check_names in candidates.items():
        score_time = imp_time.get(check_names[0], 0)
        score_fare = imp_fare.get(check_names[0], 0)
        max_score = max(score_time, score_fare)

        if max_score < importance_threshold:
            to_drop.append(original_col)
            print(f"    ❌ Drop {original_col} (importance: time={score_time:.4f}, fare={score_fare:.4f})")
        else:
            print(f"    ✅ Keep {original_col} (importance: time={score_time:.4f}, fare={score_fare:.4f})")

    return to_drop


# =============================================================================
# SHARED TRAIN/TEST SPLIT
# =============================================================================

def get_shared_split_indices(df: pd.DataFrame):
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"🔀 Split: {len(train_idx)} train rows, {len(test_idx)} test rows")
    return train_idx, test_idx


# =============================================================================
# BUILD FEATURE SETS
# =============================================================================

def build_time_feature_set(df: pd.DataFrame, keep_cols: list) -> tuple:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + keep_cols
    X = df[feature_cols].copy()
    y = df[TARGET_TIME].copy()
    return X, y


def build_fare_feature_set(df: pd.DataFrame, keep_cols: list) -> tuple:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + keep_cols + [TARGET_TIME]
    X = df[feature_cols].copy()
    y = df[TARGET_FARE].copy()
    return X, y


# =============================================================================
# ENCODING — XGBOOST (Ordinal + One-Hot + Scaler)
# =============================================================================

def build_xgb_preprocessor(numeric_cols, freq_cols=None):
    freq_cols = freq_cols or []

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    ordinal_cols = list(ORDINAL_FEATURES.keys())
    ordinal_categories = [ORDINAL_FEATURES[c] for c in ordinal_cols]
    ordinal_pipeline = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(categories=ordinal_categories,
                                    handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    onehot_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("ord", ordinal_pipeline, ordinal_cols),
        ("onehot", onehot_pipeline, ONEHOT_FEATURES),
    ]

    if freq_cols:
        freq_pipeline = Pipeline(steps=[
            ("freq", FrequencyEncoder())
        ])
        transformers.append(("freq", freq_pipeline, freq_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def prepare_xgboost_features(X_train, X_test, numeric_cols, freq_cols=None):
    preprocessor = build_xgb_preprocessor(numeric_cols, freq_cols)

    train_arr = preprocessor.fit_transform(X_train)
    test_arr = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_xgb = pd.DataFrame(train_arr, columns=feature_names)
    X_test_xgb = pd.DataFrame(test_arr, columns=feature_names)

    return X_train_xgb, X_test_xgb, preprocessor


# =============================================================================
# ENCODING — LIGHTGBM (Passthrough Numerics + Ordinal Encoding for ALL Cats)
# =============================================================================

def build_lgbm_preprocessor(numeric_cols, freq_cols=None):
    freq_cols = freq_cols or []
    categorical_cols = CATEGORICAL_FEATURES

    # Passthrough for numerics (no scaling), OrdinalEncoder for categoricals
    transformers = [
        ("num", "passthrough", numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ]

    if freq_cols:
        freq_pipeline = Pipeline(steps=[
            ("freq", FrequencyEncoder())
        ])
        transformers.append(("freq", freq_pipeline, freq_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def prepare_lightgbm_features(X_train, X_test, numeric_cols, freq_cols=None):
    preprocessor = build_lgbm_preprocessor(numeric_cols, freq_cols)

    train_arr = preprocessor.fit_transform(X_train)
    test_arr = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_lgbm = pd.DataFrame(train_arr, columns=feature_names)
    X_test_lgbm = pd.DataFrame(test_arr, columns=feature_names)

    # Track indices starting with 'cat__' for native LightGBM processing
    lgbm_cat_feature_indices = [
        i for i, col in enumerate(feature_names) if col.startswith("cat__")
    ]

    return X_train_lgbm, X_test_lgbm, lgbm_cat_feature_indices, preprocessor


# =============================================================================
# ENCODING — CATBOOST (native categorical, untouched)
# =============================================================================

def prepare_catboost_features(X_train, X_test, categorical_cols, extra_raw_categorical_cols=None):
    extra_raw_categorical_cols = extra_raw_categorical_cols or []
    all_categorical_cols = categorical_cols + extra_raw_categorical_cols

    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()

    for col in all_categorical_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cat_feature_indices = [X_train_cb.columns.get_loc(c) for c in all_categorical_cols]

    return X_train_cb, X_test_cb, cat_feature_indices


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_model_artifacts(model_name: str, artifacts: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_artifacts.pkl")
    joblib.dump(artifacts, path)
    print(f"  💾 Saved {model_name} artifacts → {path}")


# =============================================================================
# PROCESS ONE TARGET
# =============================================================================

def process_target(model_name: str, X: pd.DataFrame, y: pd.Series, train_idx, test_idx,
                    numeric_cols, freq_cols=None):
    print(f"\n{'='*60}")
    print(f"  Processing: {model_name}")
    print(f"{'='*60}")

    freq_cols = freq_cols or []

    X_train, X_test = X.loc[train_idx].reset_index(drop=True), X.loc[test_idx].reset_index(drop=True)
    y_train, y_test = y.loc[train_idx].reset_index(drop=True), y.loc[test_idx].reset_index(drop=True)

    X_train_xgb, X_test_xgb, xgb_preprocessor = prepare_xgboost_features(
        X_train, X_test, numeric_cols, freq_cols
    )
    print(f"  XGBoost feature matrix: {X_train_xgb.shape[1]} columns "
          f"(ColumnTransformer: ordinal + one-hot + scaled{' + frequency' if freq_cols else ''})")

    X_train_lgbm, X_test_lgbm, lgbm_cat_indices, lgbm_preprocessor = prepare_lightgbm_features(
        X_train, X_test, numeric_cols, freq_cols
    )
    print(f"  LightGBM feature matrix: {X_train_lgbm.shape[1]} columns, {len(lgbm_cat_indices)} categorical (integer-encoded)")

    X_train_cb, X_test_cb, cat_indices = prepare_catboost_features(
        X_train, X_test, CATEGORICAL_FEATURES, extra_raw_categorical_cols=freq_cols
    )
    print(f"  CatBoost feature matrix: {X_train_cb.shape[1]} columns, {len(cat_indices)} categorical (raw)")

    artifacts = {
        "X_train_xgb": X_train_xgb, "X_test_xgb": X_test_xgb,
        "X_train_lgbm": X_train_lgbm, "X_test_lgbm": X_test_lgbm,
        "X_train_cb": X_train_cb, "X_test_cb": X_test_cb,
        "y_train": y_train, "y_test": y_test,
        "xgb_preprocessor": xgb_preprocessor,
        "lgbm_preprocessor": lgbm_preprocessor,
        "cat_feature_indices": cat_indices,
        "lgbm_cat_feature_indices": lgbm_cat_indices,
        "categorical_columns": CATEGORICAL_FEATURES + freq_cols,
        "numeric_columns": numeric_cols,
        "feature_columns": list(X.columns),
    }

    save_model_artifacts(model_name, artifacts)
    return artifacts


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  PREPROCESSING — Building model-ready train/test sets")
    print("=" * 60)

    df = load_data()

    # Evidence-based decision on subregion/lat-lng, not assumption
    dropped_candidates = decide_candidate_drops(df)
    keep_cols = [c for c in CANDIDATE_DROP if c not in dropped_candidates]

    if dropped_candidates:
        df = df.drop(columns=dropped_candidates)
        print(f"\n🗑️  Dropping (low importance): {dropped_candidates}")

    if keep_cols:
        print(f"📌 Keeping (real predictive signal): {keep_cols}")

    latlng_keep_cols = [c for c in keep_cols if c in
                         ["source_lat", "source_lng", "dest_lat", "dest_lng"]]
    subregion_keep_cols = [c for c in keep_cols if c in
                            ["source_subregion", "dest_subregion"]]

    train_idx, test_idx = get_shared_split_indices(df)

    numeric_cols_final = NUMERIC_FEATURES + latlng_keep_cols

    # Model 1: Time
    X_time, y_time = build_time_feature_set(df, keep_cols)
    process_target("model1_time", X_time, y_time, train_idx, test_idx,
                    numeric_cols_final, freq_cols=subregion_keep_cols)

    # Model 2: Fare — numeric_cols MUST include estimated_time_min here
    numeric_cols_fare = numeric_cols_final + [TARGET_TIME]
    X_fare, y_fare = build_fare_feature_set(df, keep_cols)
    process_target("model2_fare", X_fare, y_fare, train_idx, test_idx,
                    numeric_cols_fare, freq_cols=subregion_keep_cols)

    print(f"\n✅ Preprocessing complete! Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()