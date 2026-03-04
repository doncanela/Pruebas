"""
model_tabnet.py — Train, evaluate, and persist a TabNet price-prediction model.

TabNet (Arik & Pfister 2019) is a deep-learning architecture designed for tabular
data.  It uses sequential attention to select features at each decision step,
providing built-in interpretability (feature importance masks) while achieving
competitive accuracy with gradient-boosted trees.

Pipeline mirrors model.py / model_lasso.py / model_rf.py:
    1. Load feature CSV
    2. Compute sample weights
    3. Temporal train / test split
    4. Scale features (StandardScaler)
    5. Train a TabNetRegressor on log-transformed prices
    6. Evaluate with MAE, MedAE, RMSE, R², MAPE, SMAPE and breakdowns
    7. Save model + scaler + column list
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from pytorch_tabnet.tab_model import TabNetRegressor

import config
from feature_engineer import build_feature_dataframe, get_feature_columns


# ─── Public API ──────────────────────────────────────────────────────────────

def train_tabnet(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full TabNet training pipeline.  Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List cards
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
        print(f"TabNet training set: {len(df):,} cards")

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

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    # TabNet expects numpy arrays; y must be 2D
    y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
    y_test_np = y_test.values.reshape(-1, 1).astype(np.float32)
    w_train_np = w_train.values.astype(np.float32)

    # Train
    print(f"\nTraining TabNet (n_d={config.TABNET_PARAMS['n_d']}, "
          f"n_steps={config.TABNET_PARAMS['n_steps']}, "
          f"max_epochs={config.TABNET_FIT_PARAMS['max_epochs']}) …")

    model = TabNetRegressor(**config.TABNET_PARAMS, seed=config.RANDOM_STATE,
                            verbose=10, optimizer_params={"lr": 0.02})
    model.fit(
        X_train_sc, y_train_np,
        eval_set=[(X_test_sc, y_test_np)],
        eval_metric=["mae"],
        weights=w_train_np,
        **config.TABNET_FIT_PARAMS,
    )
    print(f"  ✓ Training complete. Best epoch: {model.best_epoch}")

    # Evaluate
    y_pred_log = model.predict(X_test_sc).flatten()
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test.values)
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = _evaluate(y_actual, y_pred)
    _print_metrics(metrics, "TabNet")

    _rarity_breakdown(df, X_test.index, y_actual, y_pred)
    _price_bracket_breakdown(y_actual, y_pred)
    _print_feature_importance(model, feature_cols, top_n=25)

    # Save
    os.makedirs(config.TABNET_MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(config.TABNET_MODEL_DIR, "tabnet"))
    joblib.dump(scaler, config.TABNET_SCALER_PATH)
    joblib.dump(feature_cols, config.TABNET_FEATURE_COLS_PATH)
    print(f"TabNet model saved to {config.TABNET_MODEL_DIR}")

    return metrics


def train_tabnet_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train a TabNet model dedicated to Reserved List cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nTabNet Reserved List model — {len(df):,} cards")

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
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)
    y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
    y_test_np = y_test.values.reshape(-1, 1).astype(np.float32)
    w_train_np = w_train.values.astype(np.float32)

    print("\n  Training TabNet Reserved List model …")
    model = TabNetRegressor(**config.TABNET_RL_PARAMS, seed=config.RANDOM_STATE,
                            verbose=10, optimizer_params={"lr": 0.01})
    model.fit(
        X_train_sc, y_train_np,
        eval_set=[(X_test_sc, y_test_np)],
        eval_metric=["mae"],
        weights=w_train_np,
        **config.TABNET_RL_FIT_PARAMS,
    )
    print(f"  ✓ Training complete. Best epoch: {model.best_epoch}")

    y_pred_log = model.predict(X_test_sc).flatten()
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test.values)

    metrics = _evaluate(y_actual, y_pred)
    print("\n  ── TabNet Reserved List Model Evaluation ──")
    _print_metrics(metrics, "TabNet RL")
    _price_bracket_breakdown(y_actual, y_pred)
    _print_feature_importance(model, feature_cols, top_n=20)

    os.makedirs(config.TABNET_RL_MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(config.TABNET_RL_MODEL_DIR, "tabnet_rl"))
    joblib.dump(scaler, config.TABNET_RL_SCALER_PATH)
    joblib.dump(feature_cols, config.TABNET_RL_FEATURE_COLS_PATH)
    print(f"  TabNet RL model saved to {config.TABNET_RL_MODEL_DIR}")

    return metrics


def load_tabnet_model():
    """Load the trained TabNet model, scaler, and feature columns."""
    model = TabNetRegressor()
    model.load_model(os.path.join(config.TABNET_MODEL_DIR, "tabnet.zip"))
    scaler = joblib.load(config.TABNET_SCALER_PATH)
    feature_cols = joblib.load(config.TABNET_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def load_tabnet_reserved_list_model():
    """Load the TabNet Reserved List model."""
    model = TabNetRegressor()
    model.load_model(os.path.join(config.TABNET_RL_MODEL_DIR, "tabnet_rl.zip"))
    scaler = joblib.load(config.TABNET_RL_SCALER_PATH)
    feature_cols = joblib.load(config.TABNET_RL_FEATURE_COLS_PATH)
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


def _print_metrics(m, label="TabNet"):
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
    test_df["y_actual"] = y_actual
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
    """Print top features by TabNet aggregated attention importance."""
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"Top {top_n} features (TabNet attention importance):")
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


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(config.FEATURES_PATH):
        print("Loading feature matrix from disk …")
        df = pd.read_csv(config.FEATURES_PATH)
        train_tabnet(df=df)
    else:
        print("No feature matrix found. Run data_collector.py first.")
