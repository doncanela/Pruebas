"""
model_catboost.py — CatBoost model with native categorical feature handling.

CatBoost excels when the feature matrix includes true categoricals like
set_code, rarity, color_identity, type_bucket — it uses ordered target
statistics and symmetrical trees that handle these natively without one-hot
encoding overhead.

The feature matrix includes both:
 • Standard numeric features (same as other models)
 • Categorical metadata columns (_cat_rarity, _cat_color_identity,
   _cat_type_bucket) that are normally excluded by other models.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import config
from feature_engineer import build_feature_dataframe, get_feature_columns


# ─── Public API ──────────────────────────────────────────────────────────────

def _get_catboost_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """
    Return (feature_cols, cat_indices) where cat_indices are positional
    indices of the categorical columns within feature_cols.

    CatBoost uses the numeric features + the _cat_* categorical columns.
    """
    # Standard numeric features
    numeric_cols = get_feature_columns(df)

    # Categorical columns (underscore-prefixed, but we explicitly include them)
    cat_cols = [c for c in config.CATBOOST_CAT_FEATURES if c in df.columns]

    feature_cols = numeric_cols + cat_cols
    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))

    return feature_cols, cat_indices


def train_catboost(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full CatBoost training pipeline. Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List cards
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
    print(f"CatBoost training set: {len(df):,} cards")

    feature_cols, cat_indices = _get_catboost_feature_cols(df)
    cat_col_names = [feature_cols[i] for i in cat_indices]
    print(f"  Features: {len(feature_cols)} ({len(cat_indices)} categorical: {cat_col_names})")

    X = df[feature_cols].copy()
    # Ensure categoricals are strings
    for ci in cat_indices:
        col = feature_cols[ci]
        X[col] = X[col].astype(str)

    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    # Sample weights
    weights = _compute_sample_weights(df, y)
    print(f"\nSample weight stats:")
    print(f"  min={weights.min():.2f}  max={weights.max():.2f}  "
          f"mean={weights.mean():.2f}  median={np.median(weights):.2f}")

    # Temporal split
    if "_days_since_release" in df.columns:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]

        train_days = df["_days_since_release"].iloc[train_idx]
        test_days = df["_days_since_release"].iloc[test_idx]
        print(f"Temporal split: train cards released ≥ {train_days.min()} days ago")
        print(f"  Test set covers the most recent ~{test_days.max()} → {test_days.min()} days")
    else:
        from sklearn.model_selection import train_test_split
        rarity_strat = _get_rarity_labels(df)
        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X, y_log, weights, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=rarity_strat,
        )
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in X_train.columns:
            print(f"  {r:>10s}: train={int((X_train[col] == 1).sum()):,}  "
                  f"test={int((X_test[col] == 1).sum()):,}")

    # Build CatBoost Pools (native data structure for categoricals)
    train_pool = Pool(
        X_train, label=y_train,
        weight=w_train.values,
        cat_features=cat_indices,
    )
    test_pool = Pool(
        X_test, label=y_test,
        cat_features=cat_indices,
    )

    # Train
    params = config.CATBOOST_PARAMS.copy()
    print(f"\nTraining CatBoost ({params['iterations']} iterations) …")
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    best = model.get_best_iteration()
    print(f"  ✓ Training complete. Best iteration: {best}")

    # Evaluate
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual.values, y_pred)
    _print_metrics(metrics, "CatBoost")
    _rarity_breakdown(df, X_test.index, y_actual, y_pred)
    _price_bracket_breakdown(y_actual.values, y_pred)

    # Feature importance
    importances = model.get_feature_importance()
    _print_feature_importance(importances, feature_cols)

    # Save — CatBoost has its own save format
    model.save_model(config.CATBOOST_MODEL_PATH)
    joblib.dump(feature_cols, config.CATBOOST_FEATURE_COLS_PATH)
    print(f"CatBoost model saved to {config.CATBOOST_MODEL_PATH}")

    return metrics


def train_catboost_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train CatBoost dedicated to Reserved List cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nCatBoost Reserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough Reserved List cards. Skipping.")
        return {}

    feature_cols, cat_indices = _get_catboost_feature_cols(df)
    X = df[feature_cols].copy()
    for ci in cat_indices:
        X[feature_cols[ci]] = X[feature_cols[ci]].astype(str)

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
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]
        print(f"  Temporal split: train {len(X_train):,}  test {len(X_test):,}")
    else:
        from sklearn.model_selection import train_test_split as _tts
        X_train, X_test, y_train, y_test, w_train, _ = _tts(
            X, y_log, weights, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )
        print(f"  Random split: train {len(X_train):,}  test {len(X_test):,}")

    train_pool = Pool(X_train, label=y_train, weight=w_train.values, cat_features=cat_indices)
    test_pool = Pool(X_test, label=y_test, cat_features=cat_indices)

    params = config.CATBOOST_RL_PARAMS.copy()
    print(f"\n  Training CatBoost RL ({params['iterations']} iterations) …")
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    best = model.get_best_iteration()
    print(f"  Best iteration: {best}")

    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual.values if hasattr(y_actual, 'values') else y_actual, y_pred)
    print("\n  ── CatBoost RL Evaluation ──")
    _print_metrics(metrics, "CatBoost RL")
    _price_bracket_breakdown(y_actual.values if hasattr(y_actual, 'values') else y_actual, y_pred)

    importances = model.get_feature_importance()
    _print_feature_importance(importances, feature_cols, top_n=20)

    model.save_model(config.CATBOOST_RL_MODEL_PATH)
    joblib.dump(feature_cols, config.CATBOOST_RL_FEATURE_COLS_PATH)
    print(f"  CatBoost RL saved to {config.CATBOOST_RL_MODEL_PATH}")

    return metrics


def load_catboost_model():
    model = CatBoostRegressor()
    model.load_model(config.CATBOOST_MODEL_PATH)
    feature_cols = joblib.load(config.CATBOOST_FEATURE_COLS_PATH)
    cat_indices = [i for i, c in enumerate(feature_cols) if c.startswith("_cat_")]
    return model, feature_cols, cat_indices


def load_catboost_reserved_list_model():
    model = CatBoostRegressor()
    model.load_model(config.CATBOOST_RL_MODEL_PATH)
    feature_cols = joblib.load(config.CATBOOST_RL_FEATURE_COLS_PATH)
    cat_indices = [i for i, c in enumerate(feature_cols) if c.startswith("_cat_")]
    return model, feature_cols, cat_indices


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_sample_weights(df, y):
    weights = pd.Series(1.0, index=df.index)
    for r, w in config.RARITY_WEIGHTS.items():
        col = f"rarity_{r}"
        if col in df.columns:
            weights.loc[df[col] == 1] = w
    weights.loc[y > config.PRICE_WEIGHT_THRESHOLD] *= config.PRICE_WEIGHT_FACTOR
    weights.loc[y > 20.0] *= 2.0
    return weights


def _get_rarity_labels(df):
    labels = pd.Series("common", index=df.index)
    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in df.columns:
            labels.loc[df[col] == 1] = r
    return labels


def _evaluate(y_true, y_pred):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    mape_mask = y_t >= 0.05
    mape = np.mean(np.abs(y_t[mape_mask] - y_p[mape_mask]) / y_t[mape_mask]) * 100 if mape_mask.sum() else float("nan")
    m1 = y_t >= 1.0
    mape1 = np.mean(np.abs(y_t[m1] - y_p[m1]) / y_t[m1]) * 100 if m1.sum() else float("nan")
    denom = np.abs(y_t) + np.abs(y_p)
    smape = np.mean(2.0 * np.abs(y_t - y_p) / np.where(denom == 0, 1, denom)) * 100
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "MedAE": median_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "R2": r2_score(y_t, y_p),
        "MAPE": mape, "MAPE_above_1": mape1, "SMAPE": smape,
    }


def _print_metrics(m, title="CatBoost"):
    print(f"\n╔═══════════════════════════════════════════╗")
    print(f"║  {title:^39s}  ║")
    print(f"╠═══════════════════════════════════════════╣")
    print(f"║  MAE          : €{m['MAE']:.4f}                 ║")
    print(f"║  MedAE        : €{m['MedAE']:.4f}                 ║")
    print(f"║  RMSE         : €{m['RMSE']:.4f}                 ║")
    print(f"║  R²           :  {m['R2']:.4f}                  ║")
    print(f"║  MAPE (all)   :  {m['MAPE']:.2f}%                 ║")
    print(f"║  MAPE (≥€1)   :  {m['MAPE_above_1']:.2f}%                 ║")
    print(f"║  SMAPE        :  {m['SMAPE']:.2f}%                 ║")
    print(f"╚═══════════════════════════════════════════╝\n")


def _rarity_breakdown(df, test_idx, y_actual, y_pred):
    print("Per-rarity MAE:")
    test_df = df.loc[test_idx].copy()
    test_df["y_actual"] = y_actual.values if hasattr(y_actual, 'values') else y_actual
    test_df["y_pred"] = y_pred
    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col not in test_df.columns:
            continue
        mask = test_df[col] == 1
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(test_df.loc[mask, "y_actual"], test_df.loc[mask, "y_pred"])
        print(f"  {r:>10s}: €{mae:.4f}  (n={mask.sum():,})")
    print()


def _print_feature_importance(importances, feature_cols, top_n=25):
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"Top {top_n} most important features:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {feature_cols[idx]:<35s}  {importances[idx]:.2f}")
    print()


def _price_bracket_breakdown(y_actual, y_pred):
    y_a, y_p = np.array(y_actual), np.array(y_pred)
    brackets = [
        ("€0–€0.50", 0, 0.50, 0.25), ("€0.50–€1", 0.50, 1, 0.75),
        ("€1–€2", 1, 2, 1.5), ("€2–€5", 2, 5, 3.5),
        ("€5–€10", 5, 10, 7.5), ("€10–€20", 10, 20, 15.0),
        ("€20–€50", 20, 50, 35.0), ("€50+", 50, 9999, 100.0),
    ]
    print("Per-price-bracket breakdown:")
    print(f"  {'Bracket':>12s}  {'MAE':>10s}  {'nMAE':>6s}  {'MAPE':>8s}  {'SMAPE':>8s}  {'n':>7s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}")
    for label, lo, hi, mid in brackets:
        mask = (y_a >= lo) & (y_a < hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = mean_absolute_error(y_a[mask], y_p[mask])
        nmae = (mae / mid) * 100
        valid = y_a[mask] >= 0.05
        mape = np.mean(np.abs(y_a[mask][valid] - y_p[mask][valid]) / y_a[mask][valid]) * 100 if valid.sum() else float("nan")
        denom = np.abs(y_a[mask]) + np.abs(y_p[mask])
        smape = np.mean(2.0 * np.abs(y_a[mask] - y_p[mask]) / np.where(denom == 0, 1, denom)) * 100
        print(f"  {label:>12s}  €{mae:>8.4f}  {nmae:>5.0f}%  {mape:>7.1f}%  {smape:>7.1f}%  {n:>7,}")
    print()
    _bracket_accuracy(y_a, y_p, brackets)
    _within_threshold_accuracy(y_a, y_p)


def _bracket_accuracy(y_a, y_p, brackets):
    def _gb(price):
        for i, (_, lo, hi, _) in enumerate(brackets):
            if lo <= price < hi:
                return i
        return len(brackets) - 1
    n = len(y_a)
    exact = sum(1 for a, p in zip(y_a, y_p) if _gb(a) == _gb(p))
    w1 = sum(1 for a, p in zip(y_a, y_p) if abs(_gb(a) - _gb(p)) <= 1)
    print(f"Bracket classification accuracy:")
    print(f"  Exact bracket match: {exact:,}/{n:,} = {exact/n*100:.1f}%")
    print(f"  Within ±1 bracket:   {w1:,}/{n:,} = {w1/n*100:.1f}%")
    print()


def _within_threshold_accuracy(y_a, y_p):
    mask = y_a >= 1.0
    if mask.sum() < 10:
        return
    ya, yp = y_a[mask], y_p[mask]
    n = len(ya)
    pct = np.abs(ya - yp) / ya
    print(f"Within-X% accuracy (cards ≥€1, n={n:,}):")
    for t in [0.25, 0.50, 0.75, 1.0]:
        hit = (pct <= t).sum()
        print(f"  Within ±{t*100:.0f}%: {hit:,}/{n:,} = {hit/n*100:.1f}%")
    print()
