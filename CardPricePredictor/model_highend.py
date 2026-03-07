"""
model_highend.py — Specialised regressor for expensive cards (>€10).

Global models are dominated by bulk cards (80% below €2).  This module
trains a dedicated LightGBM on the high-value tail where getting the
price right matters most.

Architecture:
    - Filtered training set: only cards with price_eur > THRESHOLD (default €10)
    - Huber loss (robust to fat tails — dual/alpha lands, RL, etc.)
    - Heavier sample weights on >€50 and >€100 cards
    - Same feature set, but TF-IDF SVD & scarcity features get more signal
      since bulk noise is removed

At inference:
    The caller decides when to use this model.  Typical pattern:
        1. Run the global model → get a base prediction.
        2. If base_pred > €8 (or whatever gate), also run the high-end model.
        3. Blend or replace.

    Or: the ensemble layer selects this model for the upper tail.
"""

import os
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import config
from feature_engineer import build_feature_dataframe, get_feature_columns


# ─── Config ──────────────────────────────────────────────────────────────────

HIGHEND_THRESHOLD = 10.0       # minimum price for inclusion in training
GATE_THRESHOLD    =  8.0       # at inference: use high-end model above this

HIGHEND_MODEL_PATH      = os.path.join(config.MODEL_DIR, "highend_model.joblib")
HIGHEND_SCALER_PATH     = os.path.join(config.MODEL_DIR, "highend_scaler.joblib")
HIGHEND_FEATURE_COLS_PATH = os.path.join(config.MODEL_DIR, "highend_feature_columns.joblib")
HIGHEND_RL_MODEL_PATH   = os.path.join(config.MODEL_DIR, "highend_rl_model.joblib")
HIGHEND_RL_SCALER_PATH  = os.path.join(config.MODEL_DIR, "highend_rl_scaler.joblib")
HIGHEND_RL_FEATURE_COLS_PATH = os.path.join(config.MODEL_DIR, "highend_rl_feature_columns.joblib")

LGBM_HIGHEND_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 7,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_child_samples": 10,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "huber",          # robust to outliers in the tail
    "huber_delta": 1.0,            # Huber transition threshold (log-space)
}

LGBM_HIGHEND_RL_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "min_child_samples": 5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "huber",
    "huber_delta": 1.5,
}


# ─── Public API ──────────────────────────────────────────────────────────────

def train_highend(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train a high-end specialist on cards > €HIGHEND_THRESHOLD."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List (trained separately)
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards.")

    # Filter to expensive cards only
    df = df[df["price_eur"] >= HIGHEND_THRESHOLD].reset_index(drop=True)
    print(f"High-end training set (≥€{HIGHEND_THRESHOLD}): {len(df):,} cards")

    if len(df) < 100:
        print("ERROR: Not enough expensive cards to train. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    # Heavy weighting for truly expensive cards
    weights = _compute_highend_weights(y)
    print(f"  Weight stats: min={weights.min():.2f} max={weights.max():.2f} "
          f"mean={weights.mean():.2f}")

    # Temporal split
    if "_days_since_release" in df.columns:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train
    model = lgb.LGBMRegressor(**LGBM_HIGHEND_PARAMS)
    model.fit(
        X_train_sc, y_train,
        sample_weight=w_train.values,
        eval_set=[(X_test_sc, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(80, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    best_iter = getattr(model, "best_iteration_", LGBM_HIGHEND_PARAMS["n_estimators"])
    print(f"  Best iteration: {best_iter}")

    # Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.maximum(np.expm1(y_pred_log), HIGHEND_THRESHOLD * 0.5)
    y_actual = np.expm1(y_test.values)

    metrics = _evaluate(y_actual, y_pred)
    _print_metrics(metrics, "High-End Specialist (Main)")
    _price_bracket_breakdown(y_actual, y_pred)

    # Top-20 feature importances (scarcity/demand related)
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(20)
    print("\n  Top-20 features (high-end model):")
    for fname, fval in imp.items():
        print(f"    {fname:40s}  {fval:>6d}")

    # Save
    joblib.dump(model, HIGHEND_MODEL_PATH)
    joblib.dump(scaler, HIGHEND_SCALER_PATH)
    joblib.dump(feature_cols, HIGHEND_FEATURE_COLS_PATH)
    print(f"\n  High-end model saved to {config.MODEL_DIR}")

    return metrics


def train_highend_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train high-end specialist on Reserved-List expensive cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    df = df[df["price_eur"] >= HIGHEND_THRESHOLD].reset_index(drop=True)
    print(f"\nHigh-end RL training set (≥€{HIGHEND_THRESHOLD}): {len(df):,} cards")

    if len(df) < 30:
        print("  Not enough expensive RL cards. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    weights = _compute_highend_weights(y, rl=True)

    if "_days_since_release" in df.columns and len(df) >= 60:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]
    else:
        from sklearn.model_selection import train_test_split as _tts
        idx = np.arange(len(df))
        train_idx, test_idx = _tts(
            idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = lgb.LGBMRegressor(**LGBM_HIGHEND_RL_PARAMS)
    if len(X_test) >= 5:
        model.fit(
            X_train_sc, y_train, sample_weight=w_train.values,
            eval_set=[(X_test_sc, y_test)],
            callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(100)],
        )
    else:
        model.fit(X_train_sc, y_train, sample_weight=w_train.values)

    y_pred_log = model.predict(X_test_sc)
    y_pred = np.maximum(np.expm1(y_pred_log), HIGHEND_THRESHOLD * 0.5)
    y_actual = np.expm1(y_test.values)

    metrics = _evaluate(y_actual, y_pred)
    _print_metrics(metrics, "High-End RL Specialist")

    joblib.dump(model, HIGHEND_RL_MODEL_PATH)
    joblib.dump(scaler, HIGHEND_RL_SCALER_PATH)
    joblib.dump(feature_cols, HIGHEND_RL_FEATURE_COLS_PATH)
    print(f"  High-end RL model saved to {config.MODEL_DIR}")

    return metrics


def load_highend_model():
    model = joblib.load(HIGHEND_MODEL_PATH)
    scaler = joblib.load(HIGHEND_SCALER_PATH)
    feature_cols = joblib.load(HIGHEND_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def load_highend_reserved_list_model():
    model = joblib.load(HIGHEND_RL_MODEL_PATH)
    scaler = joblib.load(HIGHEND_RL_SCALER_PATH)
    feature_cols = joblib.load(HIGHEND_RL_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_highend_weights(y, rl=False):
    """Heavy weighting curve for expensive cards."""
    weights = pd.Series(1.0, index=y.index)
    # Progressive: costlier cards get more weight since they're rarer
    # and errors matter more in absolute €
    weights.loc[y >= 20]  = 2.0
    weights.loc[y >= 50]  = 4.0
    weights.loc[y >= 100] = 6.0
    weights.loc[y >= 200] = 8.0
    if rl:
        weights.loc[y >= 500]  = 10.0
        weights.loc[y >= 1000] = 12.0
    return weights


def _evaluate(y_true, y_pred):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    mape_mask = y_t >= 1.0
    mape = (np.mean(np.abs(y_t[mape_mask] - y_p[mape_mask]) / y_t[mape_mask]) * 100
            if mape_mask.sum() else float("nan"))
    denom = np.abs(y_t) + np.abs(y_p)
    smape = np.mean(2.0 * np.abs(y_t - y_p) / np.where(denom == 0, 1, denom)) * 100
    s = (y_t > 0) & (y_p > 0)
    log_mae = float(np.mean(np.abs(np.log(y_p[s]) - np.log(y_t[s])))) if s.sum() else float("nan")
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "MedAE": median_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "R2": r2_score(y_t, y_p),
        "MAPE_above_1": mape,
        "SMAPE": smape,
        "Log MAE": log_mae,
    }


def _print_metrics(m, title):
    print(f"\n╔═══════════════════════════════════════════════════╗")
    print(f"║  {title:^47s}  ║")
    print(f"╠═══════════════════════════════════════════════════╣")
    print(f"║  MAE          : €{m['MAE']:.4f}                       ║")
    print(f"║  MedAE        : €{m['MedAE']:.4f}                       ║")
    print(f"║  RMSE         : €{m['RMSE']:.4f}                       ║")
    print(f"║  R²           :  {m['R2']:.4f}                        ║")
    print(f"║  MAPE (≥€1)   :  {m['MAPE_above_1']:.2f}%                       ║")
    print(f"║  SMAPE        :  {m['SMAPE']:.2f}%                       ║")
    print(f"║  Log MAE      :  {m['Log MAE']:.4f}                       ║")
    print(f"╚═══════════════════════════════════════════════════╝\n")


def _price_bracket_breakdown(y_actual, y_pred):
    y_a, y_p = np.array(y_actual), np.array(y_pred)
    brackets = [
        ("€10–€20", 10, 20), ("€20–€50", 20, 50),
        ("€50–€100", 50, 100), ("€100–€200", 100, 200),
        ("€200–€500", 200, 500), ("€500+", 500, 99999),
    ]
    print("  Per-bracket breakdown (high-end):")
    print(f"    {'Bracket':>12s}  {'MAE':>10s}  {'MAPE':>8s}  {'n':>7s}")
    print(f"    {'─'*12}  {'─'*10}  {'─'*8}  {'─'*7}")
    for label, lo, hi in brackets:
        mask = (y_a >= lo) & (y_a < hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = mean_absolute_error(y_a[mask], y_p[mask])
        valid = y_a[mask] >= 1.0
        mape = (np.mean(np.abs(y_a[mask][valid] - y_p[mask][valid]) / y_a[mask][valid]) * 100
                if valid.sum() else float("nan"))
        print(f"    {label:>12s}  €{mae:>8.2f}  {mape:>7.1f}%  {n:>7,}")
    print()


# ─── CLI convenience ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Loading cached feature matrix …")
    df = pd.read_csv(config.FEATURES_PATH)
    print(f"Total rows: {len(df):,}")

    print("\n" + "=" * 60)
    print("  HIGH-END SPECIALIST — Main")
    print("=" * 60)
    train_highend(df=df)

    print("\n" + "=" * 60)
    print("  HIGH-END SPECIALIST — Reserved List")
    print("=" * 60)
    train_highend_reserved_list(df=pd.read_csv(config.FEATURES_PATH))
