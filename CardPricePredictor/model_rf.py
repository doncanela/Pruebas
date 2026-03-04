"""
model_rf.py — Train, evaluate, and persist a Random Forest price-prediction model.

Random Forest is an ensemble of decision trees that averages their predictions.
It requires minimal tuning, handles non-linear relationships well, and provides
built-in feature importance via impurity decrease.

Pipeline mirrors model.py / model_lasso.py:
    1. Load feature CSV
    2. Compute sample weights
    3. Temporal train / test split
    4. Scale features (StandardScaler)
    5. Train a RandomForestRegressor on log-transformed prices
    6. Evaluate
    7. Save model + scaler + column list
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import config
from feature_engineer import build_feature_dataframe, get_feature_columns


# ─── Public API ──────────────────────────────────────────────────────────────

def train_rf(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full Random Forest training pipeline.  Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List cards
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
        print(f"Random Forest training set: {len(df):,} cards")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

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
            X, y_log, weights,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=rarity_strat,
        )
        print("  ⚠ No _days_since_release — using random stratified split")
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in X_train.columns:
            print(f"  {r:>10s}: train={(X_train[col] == 1).sum():,}  "
                  f"test={(X_test[col] == 1).sum():,}")

    # Scale (RF doesn't strictly need it, but keeps consistency with other models)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train
    print(f"\nTraining Random Forest ({config.RF_PARAMS['n_estimators']} trees, "
          f"max_depth={config.RF_PARAMS['max_depth']}) …")
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X_train_sc, y_train, sample_weight=w_train.values)
    print("  ✓ Training complete.")

    # Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = _evaluate(y_actual.values, y_pred)
    _print_metrics(metrics, "Random Forest")

    _rarity_breakdown(df, X_test.index, y_actual, y_pred)
    _price_bracket_breakdown(y_actual.values, y_pred)
    _print_feature_importance(model, feature_cols, top_n=25)

    # Persist
    _save(model, scaler, feature_cols,
          config.RF_MODEL_PATH, config.RF_SCALER_PATH, config.RF_FEATURE_COLS_PATH,
          "Random Forest")

    return metrics


def train_rf_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train a Random Forest dedicated to Reserved List cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nRandom Forest Reserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough cards. Skipping.")
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

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\n  Training RF Reserved List model …")
    model = RandomForestRegressor(**config.RF_RL_PARAMS)
    model.fit(X_train_sc, y_train, sample_weight=w_train.values)
    print("  ✓ Training complete.")

    y_pred_log = model.predict(X_test_sc)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual.values, y_pred)
    print("\n  ── RF Reserved List Model Evaluation ──")
    _print_metrics(metrics, "RF Reserved List")
    _price_bracket_breakdown(y_actual.values, y_pred)
    _print_feature_importance(model, feature_cols, top_n=20)

    _save(model, scaler, feature_cols,
          config.RF_RL_MODEL_PATH, config.RF_RL_SCALER_PATH, config.RF_RL_FEATURE_COLS_PATH,
          "RF Reserved List")

    return metrics


def load_rf_model():
    """Load the trained RF model, scaler, and feature column list."""
    model = joblib.load(config.RF_MODEL_PATH)
    scaler = joblib.load(config.RF_SCALER_PATH)
    feature_cols = joblib.load(config.RF_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def load_rf_reserved_list_model():
    """Load the RF Reserved List model."""
    model = joblib.load(config.RF_RL_MODEL_PATH)
    scaler = joblib.load(config.RF_RL_SCALER_PATH)
    feature_cols = joblib.load(config.RF_RL_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


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


def _print_metrics(m, label="Random Forest"):
    print(f"\n╔═══════════════════════════════════════════╗")
    print(f"║    {label:^35s}    ║")
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


def _print_feature_importance(model, feature_cols, top_n=25):
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"Top {top_n} most important features (impurity decrease):")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {feature_cols[idx]:<35s}  {importances[idx]:.4f}")
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
        sub = y_a[mask]
        valid = sub >= 0.05
        mape = np.mean(np.abs(y_a[mask][valid] - y_p[mask][valid]) / sub[valid]) * 100 if valid.sum() else float("nan")
        denom = np.abs(y_a[mask]) + np.abs(y_p[mask])
        smape = np.mean(2.0 * np.abs(y_a[mask] - y_p[mask]) / np.where(denom == 0, 1, denom)) * 100
        print(f"  {label:>12s}  €{mae:>8.4f}  {nmae:>5.0f}%  {mape:>7.1f}%  {smape:>7.1f}%  {n:>7,}")
    print()
    # Bracket accuracy
    def _get_bracket(price):
        for i, (_, lo, hi, _) in enumerate(brackets):
            if lo <= price < hi:
                return i
        return len(brackets) - 1
    n = len(y_a)
    exact = sum(1 for a, p in zip(y_a, y_p) if _get_bracket(a) == _get_bracket(p))
    within_1 = sum(1 for a, p in zip(y_a, y_p) if abs(_get_bracket(a) - _get_bracket(p)) <= 1)
    print("Bracket classification accuracy:")
    print(f"  Exact bracket match: {exact:,}/{n:,} = {exact/n*100:.1f}%")
    print(f"  Within ±1 bracket:   {within_1:,}/{n:,} = {within_1/n*100:.1f}%")
    print()
    # Within-X%
    m1 = y_a >= 1.0
    if m1.sum() >= 10:
        y_af, y_pf = y_a[m1], y_p[m1]
        pct_err = np.abs(y_af - y_pf) / y_af
        nn = len(y_af)
        print(f"Within-X% accuracy (cards ≥€1, n={nn:,}):")
        for t in [0.25, 0.50, 0.75, 1.0]:
            hit = (pct_err <= t).sum()
            print(f"  Within ±{t*100:.0f}%: {hit:,}/{nn:,} = {hit/nn*100:.1f}%")
        print()


def _save(model, scaler, feature_cols, model_path, scaler_path, cols_path, label):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, cols_path)
    print(f"{label} model saved to {model_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(config.FEATURES_PATH):
        print("Loading feature matrix from disk …")
        df = pd.read_csv(config.FEATURES_PATH)
        train_rf(df=df)
    else:
        print("No feature matrix found. Run data_collector.py first.")
