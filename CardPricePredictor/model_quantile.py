"""
model_quantile.py — LightGBM Quantile Regression (P10 / P50 / P90).

Instead of a point estimate, produce a *range*:

    "Sheoldred, the Apocalypse  →  €38 (range €22–€61)"

This uses three separate LightGBM models, each trained with the
quantile loss at α=0.10, 0.50, 0.90.  The P50 model serves as the
central prediction; P10 and P90 form the 80% prediction interval.
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


QUANTILES = [0.10, 0.50, 0.90]
QUANTILE_NAMES = ["P10", "P50", "P90"]

_MAIN_PATHS = {
    "P10": config.QUANTILE_MODEL_P10_PATH,
    "P50": config.QUANTILE_MODEL_P50_PATH,
    "P90": config.QUANTILE_MODEL_P90_PATH,
}
_RL_PATHS = {
    "P10": config.QUANTILE_RL_MODEL_P10_PATH,
    "P50": config.QUANTILE_RL_MODEL_P50_PATH,
    "P90": config.QUANTILE_RL_MODEL_P90_PATH,
}


# ─── Public API ──────────────────────────────────────────────────────────────

def train_quantile(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train three quantile models (P10/P50/P90). Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude RL
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards.")
    print(f"Quantile training set: {len(df):,} cards\n")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    weights = _compute_sample_weights(df, y)

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
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}\n")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {}
    for alpha, name in zip(QUANTILES, QUANTILE_NAMES):
        print(f"── Training {name} (α={alpha}) ──")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=3000,
            max_depth=8,
            learning_rate=0.03,
            num_leaves=127,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=3.0,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train_sc, y_train,
            sample_weight=w_train.values,
            eval_set=[(X_test_sc, y_test)],
            eval_metric="quantile",
            callbacks=[
                lgb.early_stopping(80, verbose=True),
                lgb.log_evaluation(200),
            ],
        )
        best = model.best_iteration_ if hasattr(model, "best_iteration_") else 3000
        print(f"  {name} best iteration: {best}")
        models[name] = model
        joblib.dump(model, _MAIN_PATHS[name])
        print(f"  Saved → {_MAIN_PATHS[name]}\n")

    joblib.dump(scaler, config.QUANTILE_SCALER_PATH)
    joblib.dump(feature_cols, config.QUANTILE_FEATURE_COLS_PATH)

    # Evaluate
    preds = {}
    for name, model in models.items():
        raw = np.clip(np.expm1(model.predict(X_test_sc)), 0.01, None)
        preds[name] = raw

    y_actual = np.expm1(y_test).values

    metrics = _evaluate_quantile(y_actual, preds)
    _print_metrics(metrics)
    _calibration_check(y_actual, preds)
    _price_bracket_breakdown(y_actual, preds)

    return metrics


def train_quantile_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train quantile models on Reserved List cards only."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nQuantile RL model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough RL cards. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    weights = pd.Series(1.0, index=df.index)
    weights.loc[y > 10.0] *= 3.0
    weights.loc[y > 100.0] *= 2.0

    if "_days_since_release" in df.columns and len(df) >= 100:
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

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {}
    for alpha, name in zip(QUANTILES, QUANTILE_NAMES):
        print(f"  Training {name} (α={alpha}) for RL ...")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.60,
            reg_alpha=3.0,
            reg_lambda=8.0,
            min_child_samples=5,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train_sc, y_train,
            sample_weight=w_train.values,
            eval_set=[(X_test_sc, y_test)],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        models[name] = model
        joblib.dump(model, _RL_PATHS[name])

    joblib.dump(scaler, config.QUANTILE_RL_SCALER_PATH)
    joblib.dump(feature_cols, config.QUANTILE_RL_FEATURE_COLS_PATH)

    preds = {}
    for name, model in models.items():
        preds[name] = np.clip(np.expm1(model.predict(X_test_sc)), 0.01, None)
    y_actual = np.expm1(y_test).values

    metrics = _evaluate_quantile(y_actual, preds)
    print("\n  ── Quantile RL Evaluation ──")
    _print_metrics(metrics)
    _calibration_check(y_actual, preds)

    return metrics


def load_quantile_models():
    """Returns (models_dict, scaler, feature_cols).
    models_dict has keys 'P10', 'P50', 'P90'.
    """
    models = {}
    for name, path in _MAIN_PATHS.items():
        models[name] = joblib.load(path)
    scaler = joblib.load(config.QUANTILE_SCALER_PATH)
    feature_cols = joblib.load(config.QUANTILE_FEATURE_COLS_PATH)
    return models, scaler, feature_cols


def load_quantile_reserved_list_models():
    models = {}
    for name, path in _RL_PATHS.items():
        models[name] = joblib.load(path)
    scaler = joblib.load(config.QUANTILE_RL_SCALER_PATH)
    feature_cols = joblib.load(config.QUANTILE_RL_FEATURE_COLS_PATH)
    return models, scaler, feature_cols


# ─── Evaluation helpers ─────────────────────────────────────────────────────

def _compute_sample_weights(df, y):
    weights = pd.Series(1.0, index=df.index)
    for r, w in config.RARITY_WEIGHTS.items():
        col = f"rarity_{r}"
        if col in df.columns:
            weights.loc[df[col] == 1] = w
    weights.loc[y > config.PRICE_WEIGHT_THRESHOLD] *= config.PRICE_WEIGHT_FACTOR
    weights.loc[y > 20.0] *= 2.0
    return weights


def _evaluate_quantile(y_true, preds):
    y_t = np.array(y_true)
    p50 = preds["P50"]
    p10 = preds["P10"]
    p90 = preds["P90"]

    # Point metrics (P50 = median prediction)
    mae = mean_absolute_error(y_t, p50)
    medae = median_absolute_error(y_t, p50)
    rmse = np.sqrt(mean_squared_error(y_t, p50))
    r2 = r2_score(y_t, p50)

    mape_mask = y_t >= 0.05
    mape = np.mean(np.abs(y_t[mape_mask] - p50[mape_mask]) / y_t[mape_mask]) * 100 if mape_mask.sum() else float("nan")

    m1 = y_t >= 1.0
    mape1 = np.mean(np.abs(y_t[m1] - p50[m1]) / y_t[m1]) * 100 if m1.sum() else float("nan")

    # Coverage: fraction of actuals within [P10, P90]
    coverage = np.mean((y_t >= p10) & (y_t <= p90))

    # Mean interval width
    mean_width = np.mean(p90 - p10)

    return {
        "MAE": mae, "MedAE": medae, "RMSE": rmse, "R2": r2,
        "MAPE": mape, "MAPE_above_1": mape1,
        "Coverage_80": coverage, "Mean_Interval_Width": mean_width,
    }


def _print_metrics(m):
    print(f"\n╔═══════════════════════════════════════════╗")
    print(f"║        Quantile Model (P10/P50/P90)       ║")
    print(f"╠═══════════════════════════════════════════╣")
    print(f"║  MAE  (P50)   : €{m['MAE']:.4f}                 ║")
    print(f"║  MedAE (P50)  : €{m['MedAE']:.4f}                 ║")
    print(f"║  RMSE  (P50)  : €{m['RMSE']:.4f}                 ║")
    print(f"║  R²    (P50)  :  {m['R2']:.4f}                  ║")
    print(f"║  MAPE  (all)  :  {m['MAPE']:.2f}%                 ║")
    print(f"║  MAPE  (≥€1)  :  {m['MAPE_above_1']:.2f}%                 ║")
    print(f"║  Coverage 80% :  {m['Coverage_80']:.2%}                ║")
    print(f"║  Mean Interval: €{m['Mean_Interval_Width']:.2f}                  ║")
    print(f"╚═══════════════════════════════════════════╝\n")


def _calibration_check(y_actual, preds):
    """Check how well the quantiles are calibrated."""
    y = np.array(y_actual)
    p10, p50, p90 = preds["P10"], preds["P50"], preds["P90"]

    below_p10 = np.mean(y < p10)
    below_p50 = np.mean(y < p50)
    below_p90 = np.mean(y < p90)

    print("Quantile calibration (ideal: 10% / 50% / 90%):")
    print(f"  Below P10: {below_p10:.1%}")
    print(f"  Below P50: {below_p50:.1%}")
    print(f"  Below P90: {below_p90:.1%}")

    # By price bracket
    brackets = [("Bulk ≤€0.50", y <= 0.50), ("€0.50–€5", (y > 0.50) & (y <= 5)),
                ("€5–€20", (y > 5) & (y <= 20)), ("€20+", y > 20)]
    print("\n  Per-bracket coverage (P10–P90 interval):")
    for label, mask in brackets:
        n = mask.sum()
        if n < 10:
            continue
        cov = np.mean((y[mask] >= p10[mask]) & (y[mask] <= p90[mask]))
        width = np.mean(p90[mask] - p10[mask])
        print(f"    {label:>15s}: coverage={cov:.1%}  avg_width=€{width:.2f}  (n={n:,})")
    print()


def _price_bracket_breakdown(y_actual, preds):
    y_a = np.array(y_actual)
    p50 = preds["P50"]
    brackets = [
        ("€0–€0.50", 0, 0.50), ("€0.50–€1", 0.50, 1),
        ("€1–€2", 1, 2), ("€2–€5", 2, 5),
        ("€5–€10", 5, 10), ("€10–€20", 10, 20),
        ("€20–€50", 20, 50), ("€50+", 50, 9999),
    ]
    print("P50 MAE by price bracket:")
    for label, lo, hi in brackets:
        mask = (y_a >= lo) & (y_a < hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = mean_absolute_error(y_a[mask], p50[mask])
        print(f"  {label:>12s}: €{mae:.4f}  (n={n:,})")
    print()
