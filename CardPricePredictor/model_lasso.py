"""
model_lasso.py — Train, evaluate, and persist a Lasso regression price-prediction model.

This is an alternative to the XGBoost model (model.py).  Lasso (L1-regularised
linear regression) produces a simpler, more interpretable model that
automatically performs feature selection by driving irrelevant coefficients to
zero.  LassoCV is used so the regularisation strength (alpha) is automatically
chosen via cross-validation.

Pipeline:
    1. Load feature CSV produced by feature_engineer.py
    2. Compute sample weights (rarity + price-based)
    3. Temporal train / test split
    4. Scale features (StandardScaler)
    5. Train a LassoCV regressor on log-transformed prices
    6. Evaluate with MAE, MedAE, RMSE, R², MAPE, SMAPE and breakdowns
    7. Save model + scaler + column list for inference
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import config
from feature_engineer import build_feature_dataframe, get_feature_columns
from sample_weights import compute_sample_weights


# ─── Public API ──────────────────────────────────────────────────────────────

def train_lasso(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full Lasso training pipeline.  Returns a dict of evaluation metrics.

    You can pass either:
      - ``cards`` : raw Scryfall dicts (feature engineering happens here)
      - ``df``    : an already-built feature DataFrame
    """
    # 1. Build features if needed
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # ── Exclude Reserved List cards (trained separately) ──
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
        print(f"Lasso main training set: {len(df):,} cards")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()

    # Log-transform the target
    y_log = np.log1p(y)

    # 2. Sample weights (V4 — centralised, capped)
    weights = compute_sample_weights(df, y)
    print(f"\nSample weight stats:")
    print(f"  min={weights.min():.2f}  max={weights.max():.2f}  "
          f"mean={weights.mean():.2f}  median={np.median(weights):.2f}")

    # 3. Temporal train / test split
    if "_days_since_release" in df.columns:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]  # oldest first
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_log.iloc[train_idx]
        y_test = y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]

        train_days = df["_days_since_release"].iloc[train_idx]
        test_days = df["_days_since_release"].iloc[test_idx]
        cutoff_days = train_days.min()
        print(f"Temporal split: train cards released ≥ {cutoff_days} days ago")
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
        print("  ⚠ No _days_since_release column — using random stratified split")
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    # Print rarity distribution
    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in X_train.columns:
            n_tr = (X_train[col] == 1).sum()
            n_te = (X_test[col] == 1).sum()
            print(f"  {r:>10s}: train={n_tr:,}  test={n_te:,}")

    # 4. Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train LassoCV (auto‑selects best alpha)
    print("\nTraining LassoCV (with sample weights, cross-validated alpha) …")
    lasso_params = config.LASSO_PARAMS.copy()
    model = LassoCV(**lasso_params)
    model.fit(X_train_sc, y_train, sample_weight=w_train.values)

    best_alpha = model.alpha_
    n_nonzero = np.sum(model.coef_ != 0)
    n_features = len(feature_cols)
    print(f"  ✓ Best alpha: {best_alpha:.6f}")
    print(f"  ✓ Non-zero coefficients: {n_nonzero}/{n_features} "
          f"({n_nonzero / n_features * 100:.1f}% features retained)")

    # 6. Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    # Clip negative predictions to 0
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = _evaluate(y_actual.values if hasattr(y_actual, 'values') else y_actual, y_pred)
    _print_metrics(metrics)

    # Per-rarity breakdown
    _rarity_breakdown(df, X_test.index, y_actual, y_pred)

    # Price bracket breakdown
    _price_bracket_breakdown(
        y_actual.values if hasattr(y_actual, 'values') else y_actual,
        y_pred,
    )

    # Feature importance (top 25 by absolute coefficient)
    _print_feature_importance(model, feature_cols, top_n=25)

    # 7. Persist
    _save_lasso(model, scaler, feature_cols)

    return metrics


def train_lasso_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Train a Lasso model dedicated to Reserved List cards.
    """
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found in features.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nLasso Reserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough Reserved List cards to train. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    # Sample weights — boost expensive cards
    weights = pd.Series(1.0, index=df.index)
    weights.loc[y > 10.0] *= 3.0
    weights.loc[y > 100.0] *= 2.0

    # Temporal split
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

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train
    print("\n  Training Lasso Reserved List model …")
    # For smaller dataset, use more alphas and more CV folds
    model = LassoCV(
        alphas=200,
        cv=min(5, len(X_train)),
        max_iter=30_000,
        tol=1e-4,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train, sample_weight=w_train.values)

    best_alpha = model.alpha_
    n_nonzero = np.sum(model.coef_ != 0)
    print(f"  Best alpha: {best_alpha:.6f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")

    # Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(
        y_actual.values if hasattr(y_actual, 'values') else y_actual,
        y_pred,
    )
    print("\n  ── Lasso Reserved List Model Evaluation ──")
    _print_metrics(metrics)
    _price_bracket_breakdown(
        y_actual.values if hasattr(y_actual, 'values') else y_actual,
        y_pred,
    )
    _print_feature_importance(model, feature_cols, top_n=20)

    # Save
    joblib.dump(model, config.LASSO_RL_MODEL_PATH)
    joblib.dump(scaler, config.LASSO_RL_SCALER_PATH)
    joblib.dump(feature_cols, config.LASSO_RL_FEATURE_COLS_PATH)
    print(f"  Lasso RL model saved to {config.LASSO_RL_MODEL_PATH}")

    return metrics


def load_lasso_model():
    """Load the trained Lasso model, scaler, and feature column list."""
    model = joblib.load(config.LASSO_MODEL_PATH)
    scaler = joblib.load(config.LASSO_SCALER_PATH)
    feature_cols = joblib.load(config.LASSO_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def load_lasso_reserved_list_model():
    """Load the Lasso Reserved List model, scaler, and feature columns."""
    model = joblib.load(config.LASSO_RL_MODEL_PATH)
    scaler = joblib.load(config.LASSO_RL_SCALER_PATH)
    feature_cols = joblib.load(config.LASSO_RL_FEATURE_COLS_PATH)
    return model, scaler, feature_cols



def _get_rarity_labels(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("common", index=df.index)
    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in df.columns:
            labels.loc[df[col] == 1] = r
    if "rarity_special" in df.columns:
        labels.loc[df["rarity_special"] == 1] = "special"
    return labels


# ─── Evaluation helpers (reused from model.py patterns) ─────────────────────

def _evaluate(y_true, y_pred) -> dict:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    mape_mask = y_t >= 0.05
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs(y_t[mape_mask] - y_p[mape_mask]) / y_t[mape_mask]) * 100
    else:
        mape = float("nan")
    mape1_mask = y_t >= 1.0
    if mape1_mask.sum() > 0:
        mape_above_1 = np.mean(np.abs(y_t[mape1_mask] - y_p[mape1_mask]) / y_t[mape1_mask]) * 100
    else:
        mape_above_1 = float("nan")
    denom = np.abs(y_t) + np.abs(y_p)
    smape = np.mean(2.0 * np.abs(y_t - y_p) / np.where(denom == 0, 1, denom)) * 100
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "MedAE": median_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "R2": r2_score(y_t, y_p),
        "MAPE": mape,
        "MAPE_above_1": mape_above_1,
        "SMAPE": smape,
    }


def _print_metrics(m: dict) -> None:
    print("\n╔═══════════════════════════════════════════╗")
    print("║       Lasso Model Evaluation (EUR)        ║")
    print("╠═══════════════════════════════════════════╣")
    print(f"║  MAE          : €{m['MAE']:.4f}                 ║")
    print(f"║  MedAE        : €{m['MedAE']:.4f}                 ║")
    print(f"║  RMSE         : €{m['RMSE']:.4f}                 ║")
    print(f"║  R²           :  {m['R2']:.4f}                  ║")
    print(f"║  MAPE (all)   :  {m['MAPE']:.2f}%                 ║")
    print(f"║  MAPE (≥€1)   :  {m['MAPE_above_1']:.2f}%                 ║")
    print(f"║  SMAPE        :  {m['SMAPE']:.2f}%                 ║")
    print("╚═══════════════════════════════════════════╝\n")


def _rarity_breakdown(
    df: pd.DataFrame,
    test_idx: pd.Index,
    y_actual,
    y_pred,
) -> None:
    print("Per-rarity MAE:")
    test_df = df.loc[test_idx].copy()
    y_actual_arr = y_actual.values if hasattr(y_actual, 'values') else np.array(y_actual)
    test_df["y_actual"] = y_actual_arr
    test_df["y_pred"] = y_pred

    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col not in test_df.columns:
            continue
        mask = test_df[col] == 1
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(
            test_df.loc[mask, "y_actual"],
            test_df.loc[mask, "y_pred"],
        )
        print(f"  {r:>10s}: €{mae:.4f}  (n={mask.sum():,})")
    print()


def _print_feature_importance(model, feature_cols: list[str], top_n: int = 25) -> None:
    """Print top features by absolute Lasso coefficient + summary."""
    coefs = model.coef_
    abs_coefs = np.abs(coefs)
    n_nonzero = np.sum(coefs != 0)
    n_positive = np.sum(coefs > 0)
    n_negative = np.sum(coefs < 0)

    print(f"Feature selection summary:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Non-zero (selected): {n_nonzero}")
    print(f"  Positive coefficients: {n_positive}")
    print(f"  Negative coefficients: {n_negative}")
    print(f"  Zeroed out (dropped): {len(feature_cols) - n_nonzero}")

    top_idx = np.argsort(abs_coefs)[-top_n:][::-1]
    print(f"\nTop {top_n} features by |coefficient|:")
    for i, idx in enumerate(top_idx, 1):
        sign = "+" if coefs[idx] >= 0 else "−"
        print(f"  {i:2d}. {feature_cols[idx]:<35s}  {sign}{abs_coefs[idx]:.4f}")
    print()


def _price_bracket_breakdown(y_actual, y_pred) -> None:
    y_a = np.array(y_actual)
    y_p = np.array(y_pred)
    brackets = [
        ("€0–€0.50", 0, 0.50, 0.25),
        ("€0.50–€1", 0.50, 1, 0.75),
        ("€1–€2", 1, 2, 1.5),
        ("€2–€5", 2, 5, 3.5),
        ("€5–€10", 5, 10, 7.5),
        ("€10–€20", 10, 20, 15.0),
        ("€20–€50", 20, 50, 35.0),
        ("€50+", 50, 9999, 100.0),
    ]
    print("Per-price-bracket breakdown:")
    print(f"  {'Bracket':>12s}  {'MAE':>10s}  {'nMAE':>6s}  {'MAPE':>8s}  {'SMAPE':>8s}  {'n':>7s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}")
    for label, lo, hi, midpoint in brackets:
        mask = (y_a >= lo) & (y_a < hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = mean_absolute_error(y_a[mask], y_p[mask])
        nmae = (mae / midpoint) * 100
        mape_sub = y_a[mask]
        if (mape_sub >= 0.05).sum() > 0:
            valid = mape_sub >= 0.05
            mape = np.mean(np.abs(y_a[mask][valid] - y_p[mask][valid]) / mape_sub[valid]) * 100
        else:
            mape = float("nan")
        denom = np.abs(y_a[mask]) + np.abs(y_p[mask])
        smape = np.mean(2.0 * np.abs(y_a[mask] - y_p[mask]) / np.where(denom == 0, 1, denom)) * 100
        print(f"  {label:>12s}  €{mae:>8.4f}  {nmae:>5.0f}%  {mape:>7.1f}%  {smape:>7.1f}%  {n:>7,}")
    print()

    _bracket_accuracy(y_a, y_p, brackets)
    _within_threshold_accuracy(y_a, y_p)


def _bracket_accuracy(y_actual, y_pred, brackets) -> None:
    def _get_bracket(price, brackets):
        for i, (_, lo, hi, _) in enumerate(brackets):
            if lo <= price < hi:
                return i
        return len(brackets) - 1

    y_a = np.array(y_actual)
    y_p = np.array(y_pred)
    n = len(y_a)
    exact = sum(1 for a, p in zip(y_a, y_p) if _get_bracket(a, brackets) == _get_bracket(p, brackets))
    within_1 = sum(1 for a, p in zip(y_a, y_p) if abs(_get_bracket(a, brackets) - _get_bracket(p, brackets)) <= 1)
    print("Bracket classification accuracy:")
    print(f"  Exact bracket match: {exact:,}/{n:,} = {exact/n*100:.1f}%")
    print(f"  Within ±1 bracket:   {within_1:,}/{n:,} = {within_1/n*100:.1f}%")
    print()


def _within_threshold_accuracy(y_actual, y_pred) -> None:
    y_a = np.array(y_actual)
    y_p = np.array(y_pred)
    mask = y_a >= 1.0
    if mask.sum() < 10:
        return
    y_a_f = y_a[mask]
    y_p_f = y_p[mask]
    n = len(y_a_f)
    pct_errors = np.abs(y_a_f - y_p_f) / y_a_f

    thresholds = [0.25, 0.50, 0.75, 1.0]
    print(f"Within-X% accuracy (cards ≥€1, n={n:,}):")
    for t in thresholds:
        hit = (pct_errors <= t).sum()
        print(f"  Within ±{t*100:.0f}%: {hit:,}/{n:,} = {hit/n*100:.1f}%")
    print()


# ─── Persistence ─────────────────────────────────────────────────────────────

def _save_lasso(model, scaler, feature_cols) -> None:
    joblib.dump(model, config.LASSO_MODEL_PATH)
    joblib.dump(scaler, config.LASSO_SCALER_PATH)
    joblib.dump(feature_cols, config.LASSO_FEATURE_COLS_PATH)
    print(f"Lasso model saved to {config.LASSO_MODEL_PATH}")
    print(f"Scaler saved to {config.LASSO_SCALER_PATH}")
    print(f"Feature columns saved to {config.LASSO_FEATURE_COLS_PATH}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(config.FEATURES_PATH):
        print("Loading feature matrix from disk …")
        df = pd.read_csv(config.FEATURES_PATH)
        train_lasso(df=df)
    else:
        print("No feature matrix found. Run data_collector.py first, then feature_engineer.py.")
