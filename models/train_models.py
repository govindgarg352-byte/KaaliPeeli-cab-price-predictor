# =============================================================================
# train_models.py — Stacking Ensemble: XGBoost + LightGBM + CatBoost -> Ridge/Lasso
#
# TWO CASCADED MODELS:
#   Model 1 (ETA):  predicts estimated_time_min from route + context features
#   Model 2 (Fare): predicts fare_amount — depends on Model 1's output
# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from custom_transformers import FrequencyEncoder  # noqa: F401 — needed for joblib unpickling

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================

ARTIFACTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_ready')
MODELS_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'trained_models')

RANDOM_STATE       = 42
N_FOLDS            = 7
N_TRIALS_BASE      = 30   # Model 1
N_TRIALS_BASE_M2   = 60   # Model 2 — more search for higher-variance target
N_TRIALS_META      = 20
N_TRIALS_META_M2   = 30   # deeper meta search for Model 2

ETA_COL = "estimated_time_min"   # the column Model 2 receives from Model 1


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

def load_artifacts(model_name: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, f"{model_name}_artifacts.pkl")
    arts = joblib.load(path)
    print(f"  📂 {model_name}: "
          f"{arts['X_train_xgb'].shape[0]} train | "
          f"XGB={arts['X_train_xgb'].shape[1]} cols | "
          f"LGBM={arts['X_train_lgbm'].shape[1]} cols | "
          f"CB={arts['X_train_cb'].shape[1]} cols")
    return arts


# =============================================================================
# XGBOOST — OPTUNA TUNING
# =============================================================================

def tune_xgboost(X_train, y_train, n_trials=N_TRIALS_BASE) -> dict:
    print("  🔧 Tuning XGBoost...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist",
        }
        scores = cross_val_score(XGBRegressor(**params), X_train, y_train,
                                  cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     ✅ Best CV RMSE: {study.best_value:.4f}")
    return study.best_params


# =============================================================================
# LIGHTGBM — OPTUNA TUNING (manual KFold — handles categorical features safely)
# =============================================================================

def tune_lightgbm(X_train, y_train, lgbm_cat_idx, n_trials=N_TRIALS_BASE) -> dict:
    print("  🔧 Tuning LightGBM...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X  = X_train.reset_index(drop=True)
    y  = pd.Series(y_train).reset_index(drop=True)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
        }
        fold_rmses = []
        for tr, val in kf.split(X):
            m = LGBMRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=lgbm_cat_idx)
            fold_rmses.append(rmse(y.iloc[val], m.predict(X.iloc[val])))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     ✅ Best CV RMSE: {study.best_value:.4f}")
    return study.best_params


# =============================================================================
# CATBOOST — OPTUNA TUNING
# =============================================================================

def tune_catboost(X_train, y_train, cat_idx, n_trials=N_TRIALS_BASE) -> dict:
    print("  🔧 Tuning CatBoost...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X  = X_train.reset_index(drop=True)
    y  = pd.Series(y_train).reset_index(drop=True)

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 200, 1000),
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":     trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        }
        fold_rmses = []
        for tr, val in kf.split(X):
            m = CatBoostRegressor(**params, cat_features=cat_idx,
                                   random_state=RANDOM_STATE, verbose=False,
                                   allow_writing_files=False)
            m.fit(X.iloc[tr], y.iloc[tr])
            fold_rmses.append(rmse(y.iloc[val], m.predict(X.iloc[val])))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     ✅ Best CV RMSE: {study.best_value:.4f}")
    best = study.best_params
    best["cat_features"] = cat_idx
    return best


# =============================================================================
# OUT-OF-FOLD PREDICTIONS
# =============================================================================

def oof_xgboost(params, X_train, y_train) -> np.ndarray:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    m  = XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    return cross_val_predict(m, X_train, y_train, cv=kf, n_jobs=-1)


def oof_lightgbm(params, X_train, y_train, lgbm_cat_idx) -> np.ndarray:
    kf    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X     = X_train.reset_index(drop=True)
    y     = pd.Series(y_train).reset_index(drop=True)
    preds = np.zeros(len(y))
    for tr, val in kf.split(X):
        m = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr], categorical_feature=lgbm_cat_idx)
        preds[val] = m.predict(X.iloc[val])
    return preds


def oof_catboost(params, X_train, y_train, cat_idx) -> np.ndarray:
    kf    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X     = X_train.reset_index(drop=True)
    y     = pd.Series(y_train).reset_index(drop=True)
    preds = np.zeros(len(y))
    for tr, val in kf.split(X):
        m = CatBoostRegressor(
            **{k: v for k, v in params.items() if k != "cat_features"},
            cat_features=cat_idx, random_state=RANDOM_STATE,
            verbose=False, allow_writing_files=False,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[val] = m.predict(X.iloc[val])
    return preds


# =============================================================================
# BUILD STACKED PREDICTION
# =============================================================================

def build_stack(xgb_m, lgbm_m, cb_m, meta_m,
                X_xgb, X_lgbm, X_cb) -> np.ndarray:
    return meta_m.predict(pd.DataFrame({
        "xgb_pred":  xgb_m.predict(X_xgb),
        "lgbm_pred": lgbm_m.predict(X_lgbm),
        "cb_pred":   cb_m.predict(X_cb),
    }))


# =============================================================================
# META-LEARNER — OPTUNA (Ridge vs Lasso on 3-col OOF matrix)
# =============================================================================

def tune_meta_learner(meta_X_train, y_train, n_trials=N_TRIALS_META,
                      use_gbm=False) -> tuple:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def make_linear_obj(cls):
        def objective(trial):
            kw = {"alpha": trial.suggest_float("alpha", 1e-4, 200.0, log=True)}
            if cls is Lasso:
                kw["max_iter"] = 5000
            if cls is ElasticNet:
                kw["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
                kw["max_iter"] = 5000
            scores = cross_val_score(cls(**kw), meta_X_train, y_train,
                                      cv=kf, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        return objective

    results = {}
    for name, cls in [("Ridge", Ridge), ("Lasso", Lasso), ("ElasticNet", ElasticNet)]:
        study = optuna.create_study(direction="minimize")
        study.optimize(make_linear_obj(cls), n_trials=n_trials, show_progress_bar=False)
        results[name] = {"cls": cls, "params": study.best_params, "rmse": study.best_value}
        print(f"     {name}: CV RMSE={study.best_value:.4f}")

    if use_gbm:
        def gbm_obj(trial):
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
                "max_depth":      trial.suggest_int("max_depth", 2, 5),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
                "random_state":   RANDOM_STATE,
            }
            scores = cross_val_score(GradientBoostingRegressor(**params), meta_X_train, y_train,
                                      cv=kf, scoring="neg_root_mean_squared_error")
            return -scores.mean()
        gbm_study = optuna.create_study(direction="minimize")
        gbm_study.optimize(gbm_obj, n_trials=n_trials, show_progress_bar=False)
        results["GBM"] = {"cls": GradientBoostingRegressor, "params": gbm_study.best_params,
                          "rmse": gbm_study.best_value}
        print(f"     GBM: CV RMSE={gbm_study.best_value:.4f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"     🏆 Meta-learner chosen: {best_name}")
    return best_name, results[best_name]["cls"], results[best_name]["params"]


# =============================================================================
# TRAIN MODEL 1 — ETA (returns trained models + OOF/test predictions)
# =============================================================================

def train_eta_model() -> dict:
    print(f"\n{'='*60}")
    print("  MODEL 1 — ETA (estimated_time_min)")
    print(f"{'='*60}\n")

    arts        = load_artifacts("model1_time")
    X_tr_xgb    = arts["X_train_xgb"]
    X_te_xgb    = arts["X_test_xgb"]
    X_tr_lgbm   = arts["X_train_lgbm"]
    X_te_lgbm   = arts["X_test_lgbm"]
    X_tr_cb     = arts["X_train_cb"]
    X_te_cb     = arts["X_test_cb"]
    y_train     = arts["y_train"]
    y_test      = arts["y_test"]
    cat_idx     = arts["cat_feature_indices"]
    lgbm_cat_idx = arts["lgbm_cat_feature_indices"]

    xgb_params  = tune_xgboost(X_tr_xgb, y_train)
    lgbm_params = tune_lightgbm(X_tr_lgbm, y_train, lgbm_cat_idx)
    cb_params   = tune_catboost(X_tr_cb, y_train, cat_idx)

    print("\n  📊 Generating OOF predictions for Model 1...")
    xgb_oof  = oof_xgboost(xgb_params,   X_tr_xgb,  y_train)
    lgbm_oof = oof_lightgbm(lgbm_params, X_tr_lgbm, y_train, lgbm_cat_idx)
    cb_oof   = oof_catboost(cb_params,   X_tr_cb,   y_train, cat_idx)

    meta_X_tr = pd.DataFrame({"xgb_pred": xgb_oof, "lgbm_pred": lgbm_oof, "cb_pred": cb_oof})
    print(f"     OOF RMSE  XGB={rmse(y_train, xgb_oof):.4f}  "
          f"LGBM={rmse(y_train, lgbm_oof):.4f}  CB={rmse(y_train, cb_oof):.4f}")

    print("\n  🔧 Tuning meta-learner (Ridge vs Lasso vs ElasticNet)...")
    meta_name, meta_cls, meta_params = tune_meta_learner(meta_X_tr, y_train,
                                                          n_trials=N_TRIALS_META)
    if meta_cls is GradientBoostingRegressor:
        meta_m = meta_cls(**meta_params)
    else:
        kw = {k: v for k, v in meta_params.items()}
        if "max_iter" not in kw and meta_cls in (Lasso, ElasticNet):
            kw["max_iter"] = 5000
        meta_m = meta_cls(**kw)
    meta_m.fit(meta_X_tr, y_train)

    print("\n  🏋️  Refitting base learners on full training set...")
    xgb_m  = XGBRegressor(**xgb_params,  random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    lgbm_m = LGBMRegressor(**lgbm_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    cb_m   = CatBoostRegressor(**cb_params, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)
    xgb_m.fit(X_tr_xgb, y_train)
    lgbm_m.fit(X_tr_lgbm, y_train, categorical_feature=lgbm_cat_idx)
    cb_m.fit(X_tr_cb, y_train)

    # Test-set stacked prediction for Model 1
    eta_test_pred = build_stack(xgb_m, lgbm_m, cb_m, meta_m,
                                 X_te_xgb, X_te_lgbm, X_te_cb)

    # Training-set stacked OOF prediction — passed to Model 2 to replace ground truth ETA
    eta_train_oof = meta_m.predict(meta_X_tr)

    rmse_xgb  = rmse(y_test, xgb_m.predict(X_te_xgb))
    rmse_lgbm = rmse(y_test, lgbm_m.predict(X_te_lgbm))
    rmse_cb   = rmse(y_test, cb_m.predict(X_te_cb))
    rmse_stk  = rmse(y_test, eta_test_pred)

    print(f"\n  📊 Model 1 Test RMSE (minutes):")
    print(f"     XGBoost:   {rmse_xgb:.4f}")
    print(f"     LightGBM:  {rmse_lgbm:.4f}")
    print(f"     CatBoost:  {rmse_cb:.4f}")
    print(f"     Stacked:   {rmse_stk:.4f}  ← used as Model 2 input")

    return {
        "xgb_model": xgb_m, "lgbm_model": lgbm_m, "cb_model": cb_m, "meta_model": meta_m,
        "meta_learner_name": meta_name,
        "xgb_params": xgb_params, "lgbm_params": lgbm_params, "cb_params": cb_params,
        "meta_params": meta_params,
        "rmse_xgb": rmse_xgb, "rmse_lgbm": rmse_lgbm, "rmse_cb": rmse_cb, "rmse_stacked": rmse_stk,
        "eta_train_oof": eta_train_oof,
        "eta_test_pred": eta_test_pred,
        "xgb_preprocessor": arts["xgb_preprocessor"],
        "lgbm_preprocessor": arts.get("lgbm_preprocessor"),
        "cat_feature_indices": cat_idx,
        "lgbm_cat_feature_indices": lgbm_cat_idx,
    }


# =============================================================================
# TRAIN MODEL 2 — FARE (injects Model 1's OOF predictions, not ground truth)
# =============================================================================

def train_fare_model(eta_bundle: dict) -> dict:
    print(f"\n{'='*60}")
    print("  MODEL 2 — FARE (fare_amount)")
    print("  ETA feature = Model 1 OOF predictions (not ground truth)")
    print(f"{'='*60}\n")

    arts      = load_artifacts("model2_fare")
    X_tr_xgb  = arts["X_train_xgb"].copy()
    X_te_xgb  = arts["X_test_xgb"].copy()
    X_tr_lgbm = arts["X_train_lgbm"].copy()
    X_te_lgbm = arts["X_test_lgbm"].copy()
    X_tr_cb   = arts["X_train_cb"].copy()
    X_te_cb   = arts["X_test_cb"].copy()
    y_train   = arts["y_train"]
    y_test    = arts["y_test"]
    cat_idx   = arts["cat_feature_indices"]
    lgbm_cat_idx = arts["lgbm_cat_feature_indices"]

    # ---- THE KEY CASCADING FIX ----
    # Replace ground-truth ETA with Model 1's stacked predictions in XGB (scaled),
    # LightGBM (passthrough), and CatBoost (raw string) feature matrices.
    xgb_eta_col  = "num__estimated_time_min"
    lgbm_eta_col = "num__estimated_time_min"
    cb_eta_col   = ETA_COL

    if xgb_eta_col in X_tr_xgb.columns:
        X_tr_xgb[xgb_eta_col] = eta_bundle["eta_train_oof"]
        X_te_xgb[xgb_eta_col] = eta_bundle["eta_test_pred"]
        print(f"  ✅ Replaced '{xgb_eta_col}' in XGB matrix with Model 1 predictions")
    else:
        print(f"  ⚠️  '{xgb_eta_col}' not found in XGB matrix — check ColumnTransformer prefix")

    if lgbm_eta_col in X_tr_lgbm.columns:
        X_tr_lgbm[lgbm_eta_col] = eta_bundle["eta_train_oof"]
        X_te_lgbm[lgbm_eta_col] = eta_bundle["eta_test_pred"]
        print(f"  ✅ Replaced '{lgbm_eta_col}' in LGBM matrix with Model 1 predictions")
    else:
        print(f"  ⚠️  '{lgbm_eta_col}' not found in LGBM matrix — check ColumnTransformer prefix")

    if cb_eta_col in X_tr_cb.columns:
        # Keep as float — CatBoost handles numeric columns natively; casting to str destroys signal
        X_tr_cb[cb_eta_col] = eta_bundle["eta_train_oof"].astype(float)
        X_te_cb[cb_eta_col] = eta_bundle["eta_test_pred"].astype(float)
        # Remove from cat_feature_indices since it is now purely numeric
        eta_cb_pos = X_tr_cb.columns.get_loc(cb_eta_col)
        cat_idx = [i for i in cat_idx if i != eta_cb_pos]
        print(f"  ✅ Replaced '{cb_eta_col}' in CatBoost matrix with Model 1 predictions (numeric)")
    else:
        print(f"  ⚠️  '{cb_eta_col}' not found in CatBoost matrix — check preprocessing output")

    xgb_params  = tune_xgboost(X_tr_xgb,   y_train, n_trials=N_TRIALS_BASE_M2)
    lgbm_params = tune_lightgbm(X_tr_lgbm, y_train, lgbm_cat_idx, n_trials=N_TRIALS_BASE_M2)
    cb_params   = tune_catboost(X_tr_cb,   y_train, cat_idx, n_trials=N_TRIALS_BASE_M2)

    print("\n  📊 Generating OOF predictions for Model 2...")
    xgb_oof  = oof_xgboost(xgb_params,   X_tr_xgb,  y_train)
    lgbm_oof = oof_lightgbm(lgbm_params, X_tr_lgbm, y_train, lgbm_cat_idx)
    cb_oof   = oof_catboost(cb_params,   X_tr_cb,   y_train, cat_idx)

    meta_X_tr = pd.DataFrame({"xgb_pred": xgb_oof, "lgbm_pred": lgbm_oof, "cb_pred": cb_oof})
    print(f"     OOF RMSE  XGB={rmse(y_train, xgb_oof):.4f}  "
          f"LGBM={rmse(y_train, lgbm_oof):.4f}  CB={rmse(y_train, cb_oof):.4f}")

    print("\n  🔧 Tuning meta-learner (Ridge vs Lasso vs ElasticNet vs GBM)...")
    meta_name, meta_cls, meta_params = tune_meta_learner(meta_X_tr, y_train,
                                                          n_trials=N_TRIALS_META_M2,
                                                          use_gbm=True)
    if meta_cls is GradientBoostingRegressor:
        meta_m = meta_cls(**meta_params)
    else:
        kw = {k: v for k, v in meta_params.items()}
        if "max_iter" not in kw and meta_cls in (Lasso, ElasticNet):
            kw["max_iter"] = 5000
        meta_m = meta_cls(**kw)
    meta_m.fit(meta_X_tr, y_train)

    print("\n  🏋️  Refitting base learners on full training set...")
    xgb_m  = XGBRegressor(**xgb_params,   random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    lgbm_m = LGBMRegressor(**lgbm_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    cb_m   = CatBoostRegressor(**cb_params, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)
    xgb_m.fit(X_tr_xgb, y_train)
    lgbm_m.fit(X_tr_lgbm, y_train, categorical_feature=lgbm_cat_idx)
    cb_m.fit(X_tr_cb, y_train)

    fare_test_pred = build_stack(xgb_m, lgbm_m, cb_m, meta_m,
                                  X_te_xgb, X_te_lgbm, X_te_cb)

    rmse_xgb  = rmse(y_test, xgb_m.predict(X_te_xgb))
    rmse_lgbm = rmse(y_test, lgbm_m.predict(X_te_lgbm))
    rmse_cb   = rmse(y_test, cb_m.predict(X_te_cb))
    rmse_stk  = rmse(y_test, fare_test_pred)

    print(f"\n  📊 Model 2 Test RMSE (₹) — using Model 1 predictions as ETA feature:")
    print(f"     XGBoost:   {rmse_xgb:.4f}")
    print(f"     LightGBM:  {rmse_lgbm:.4f}")
    print(f"     CatBoost:  {rmse_cb:.4f}")
    print(f"     Stacked:   {rmse_stk:.4f}")

    return {
        "xgb_model": xgb_m, "lgbm_model": lgbm_m, "cb_model": cb_m, "meta_model": meta_m,
        "meta_learner_name": meta_name,
        "xgb_params": xgb_params, "lgbm_params": lgbm_params, "cb_params": cb_params,
        "meta_params": meta_params,
        "rmse_xgb": rmse_xgb, "rmse_lgbm": rmse_lgbm, "rmse_cb": rmse_cb, "rmse_stacked": rmse_stk,
        "cat_feature_indices": cat_idx,
        "lgbm_cat_feature_indices": lgbm_cat_idx,
        "xgb_preprocessor": arts["xgb_preprocessor"],
        "lgbm_preprocessor": arts.get("lgbm_preprocessor"),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  TRAIN MODELS")
    print("  Stack: XGBoost + LightGBM + CatBoost → Ridge/Lasso")
    print("  Tuning: Optuna  |  Eval: 5-fold CV RMSE")
    print("  Cascaded: Model 1 ETA → feeds Model 2 Fare")
    print("=" * 60)

    # ---- Train Model 1 first ----
    eta_bundle  = train_eta_model()

    # ---- Train Model 2 using Model 1's predictions, not ground truth ----
    fare_bundle = train_fare_model(eta_bundle)

    # ---- Save both bundles ----
    os.makedirs(MODELS_OUT_DIR, exist_ok=True)

    joblib.dump(eta_bundle,  os.path.join(MODELS_OUT_DIR, "model1_time_bundle.pkl"))
    joblib.dump(fare_bundle, os.path.join(MODELS_OUT_DIR, "model2_fare_bundle.pkl"))
    print(f"\n  💾 Bundles saved → {MODELS_OUT_DIR}")

    # ---- Summary ----
    summary = {
        "model1_time": {
            "rmse_xgb":     round(eta_bundle["rmse_xgb"], 4),
            "rmse_lgbm":    round(eta_bundle["rmse_lgbm"], 4),
            "rmse_cb":      round(eta_bundle["rmse_cb"], 4),
            "rmse_stacked": round(eta_bundle["rmse_stacked"], 4),
            "meta_learner": eta_bundle["meta_learner_name"],
        },
        "model2_fare": {
            "rmse_xgb":     round(fare_bundle["rmse_xgb"], 4),
            "rmse_lgbm":    round(fare_bundle["rmse_lgbm"], 4),
            "rmse_cb":      round(fare_bundle["rmse_cb"], 4),
            "rmse_stacked": round(fare_bundle["rmse_stacked"], 4),
            "meta_learner": fare_bundle["meta_learner_name"],
            "note": "ETA feature = Model 1 stacked OOF predictions (not ground truth)",
        },
    }

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    for target, res in summary.items():
        print(f"\n  {target}:")
        for k, v in res.items():
            print(f"    {k}: {v}")

    summary_path = os.path.join(MODELS_OUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  💾 Summary → {summary_path}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()