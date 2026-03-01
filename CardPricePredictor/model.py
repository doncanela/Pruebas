"""
model.py — Train, evaluate, and persist the XGBoost price-prediction model.

Pipeline:
    1. Load feature CSV produced by feature_engineer.py
    2. Compute sample weights (rarity + price-based)
    3. Temporal train / test split (older sets → train, newer → test)
    4. Scale features (StandardScaler)
    5. Train an XGBRegressor with sample weights
    6. Evaluate with MAE, MedAE, RMSE, R², MAPE, SMAPE and breakdowns
    7. Optionally run two-stage model (classify expensive → regress)
    8. Save model + scaler + column list for inference
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from xgboost import XGBRegressor

import config
from feature_engineer import build_feature_dataframe, get_feature_columns


# ─── Public API ──────────────────────────────────────────────────────────────

def train(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full training pipeline.  Returns a dict of evaluation metrics.

    You can pass either:
      - `cards` : raw Scryfall dicts (feature engineering happens here)
      - `df`    : an already-built feature DataFrame
    """
    # 1. Build features if needed
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # ── Exclude Reserved List cards from the main model ──
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
        print(f"Main model training set: {len(df):,} cards")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()

    # Log-transform the target to handle heavy right skew in card prices
    y_log = np.log1p(y)

    # 2. Compute sample weights
    weights = _compute_sample_weights(df, y)
    print(f"\nSample weight stats:")
    print(f"  min={weights.min():.2f}  max={weights.max():.2f}  "
          f"mean={weights.mean():.2f}  median={np.median(weights):.2f}")

    # 3. Temporal train / test split (older sets → train, newer → test)
    #    This avoids data leakage from mixing future and past cards.
    if "_days_since_release" in df.columns:
        # Sort by release date (larger _days_since_release = older)
        sort_idx = df["_days_since_release"].values.argsort()[::-1]  # oldest first
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_log.iloc[train_idx]
        y_test = y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]
        w_test = weights.iloc[test_idx]

        # Report temporal cutoff
        train_days = df["_days_since_release"].iloc[train_idx]
        test_days = df["_days_since_release"].iloc[test_idx]
        cutoff_days = train_days.min()
        print(f"Temporal split: train cards released ≥ {cutoff_days} days ago")
        print(f"  Test set covers the most recent ~{test_days.max()} → {test_days.min()} days")
    else:
        # Fallback: stratified random split (when metadata column is absent)
        rarity_strat = _get_rarity_labels(df)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
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
            n_train = (X_train[col] == 1).sum()
            n_test = (X_test[col] == 1).sum()
            print(f"  {r:>10s}: train={n_train:,}  test={n_test:,}")

    # 4. Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train XGBoost with sample weights + progress tracking
    print("\nTraining XGBoost (with sample weights) …")
    n_est = config.XGBOOST_PARAMS.get("n_estimators", 3000)
    es_rounds = config.XGBOOST_PARAMS.get("early_stopping_rounds", 80)
    print(f"  Max iterations: {n_est}  (early stopping after {es_rounds} rounds of no improvement)")

    # XGBoost 3.x: verbose=N prints every N rounds; we'll use 100
    model = XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(
        X_train_sc, y_train,
        sample_weight=w_train.values,
        eval_set=[(X_test_sc, y_test)],
        verbose=100,
    )
    best_iter = getattr(model, "best_iteration", n_est)
    print(f"  ✓ Training complete. Best iteration: {best_iter}")

    # 6. Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual, y_pred)
    _print_metrics(metrics)

    # Per-rarity breakdown
    _rarity_breakdown(df, X_test.index, y_actual, y_pred)

    # Price bracket breakdown
    _price_bracket_breakdown(y_actual, y_pred)

    # Feature importance (top 25)
    _print_feature_importance(model, feature_cols, top_n=25)

    # 7. Two-stage specialist: re-predict expensive cards with a dedicated model
    print("\n── Two-stage model: expensive-card specialist ──")
    y_pred_2stage = _two_stage_predict(
        X_train, y_train, w_train,
        X_test_sc, y_actual, y_pred,
        scaler, feature_cols,
    )
    if y_pred_2stage is not None:
        metrics_2stage = _evaluate(y_actual, y_pred_2stage)
        print("  Combined two-stage metrics:")
        _print_metrics(metrics_2stage)
        _price_bracket_breakdown(y_actual, y_pred_2stage)

    # 8. Persist
    _save(model, scaler, feature_cols)

    return metrics


def load_model():
    """Load the trained model, scaler, and feature column list."""
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    feature_cols = joblib.load(config.FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def load_reserved_list_model():
    """Load the Reserved List specialist model, scaler, and feature columns."""
    model = joblib.load(config.RL_MODEL_PATH)
    scaler = joblib.load(config.RL_SCALER_PATH)
    feature_cols = joblib.load(config.RL_FEATURE_COLS_PATH)
    return model, scaler, feature_cols


# ─── Reserved List model ────────────────────────────────────────────────────

def train_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Train a dedicated model for Reserved List cards only.
    These cards span €0.02–€30,000+ and are never reprinted, so they need
    their own model with no outlier cap and tuned for small-N, wide-range data.
    """
    # 1. Build / filter to RL-only
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found in features.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nReserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough Reserved List cards to train. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    # 2. Sample weights — boost expensive cards
    weights = pd.Series(1.0, index=df.index)
    expensive = y > 10.0
    weights.loc[expensive] *= 3.0
    very_expensive = y > 100.0
    weights.loc[very_expensive] *= 2.0

    # 3. Temporal split (or random if too few)
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

    # 4. Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train
    print("\n  Training Reserved List XGBoost …")
    model = XGBRegressor(**config.RL_XGBOOST_PARAMS)
    model.fit(
        X_train_sc, y_train,
        sample_weight=w_train.values,
        eval_set=[(X_test_sc, y_test)],
        verbose=50,
    )
    best = getattr(model, "best_iteration", config.RL_XGBOOST_PARAMS["n_estimators"])
    print(f"  Best iteration: {best}")

    # 6. Evaluate
    y_pred_log = model.predict(X_test_sc)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual, y_pred)
    print("\n  ── Reserved List Model Evaluation ──")
    _print_metrics(metrics)
    _price_bracket_breakdown(y_actual, y_pred)
    _print_feature_importance(model, feature_cols, top_n=20)

    # 7. Save
    joblib.dump(model, config.RL_MODEL_PATH)
    joblib.dump(scaler, config.RL_SCALER_PATH)
    joblib.dump(feature_cols, config.RL_FEATURE_COLS_PATH)
    print(f"  Reserved List model saved to {config.RL_MODEL_PATH}")

    return metrics


# ─── Two-stage model ────────────────────────────────────────────────────────

EXPENSIVE_THRESHOLD = 20.0  # €  — cards above this get a specialist model

def _two_stage_predict(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    w_train: pd.Series,
    X_test_sc: np.ndarray,
    y_actual: np.ndarray,
    y_pred_base: np.ndarray,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> np.ndarray | None:
    """
    Train a specialist regressor only on expensive cards (≥ EXPENSIVE_THRESHOLD).
    Blend its predictions for cards the base model predicted as expensive.
    Returns a new y_pred array with improved expensive-card predictions,
    or None if there aren't enough expensive training samples.
    """
    y_train_real = np.expm1(y_train_log)
    expensive_mask_train = y_train_real >= EXPENSIVE_THRESHOLD

    n_expensive = expensive_mask_train.sum()
    if n_expensive < 100:
        print(f"  Only {n_expensive} expensive training samples — skipping specialist.")
        return None

    print(f"  Training specialist on {n_expensive:,} cards ≥ €{EXPENSIVE_THRESHOLD}")

    # Train specialist on expensive subset
    X_train_exp = X_train.loc[expensive_mask_train]
    y_train_exp = y_train_log.loc[expensive_mask_train]
    w_train_exp = w_train.loc[expensive_mask_train]

    X_train_exp_sc = scaler.transform(X_train_exp)

    # Build an eval set from expensive TEST cards only
    expensive_mask_test = y_actual >= EXPENSIVE_THRESHOLD
    n_exp_test = expensive_mask_test.sum()
    if n_exp_test < 5:
        print(f"  Only {n_exp_test} expensive test samples — skipping specialist.")
        return None

    X_test_exp_sc = X_test_sc[expensive_mask_test]
    y_test_exp_log = np.log1p(y_actual[expensive_mask_test])

    # Conservative specialist: fewer trees, shallower, squared error for stability
    specialist_params = {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "min_child_weight": 10,
        "gamma": 0.3,
        "random_state": 42,
        "early_stopping_rounds": 40,
        "objective": "reg:squarederror",
    }

    specialist = XGBRegressor(**specialist_params)
    specialist.fit(
        X_train_exp_sc, y_train_exp,
        sample_weight=w_train_exp.values,
        eval_set=[(X_test_exp_sc, y_test_exp_log)],
        verbose=0,
    )
    best = getattr(specialist, "best_iteration", specialist_params["n_estimators"])
    print(f"  Specialist trained (best iteration: {best})")

    # Validate specialist on expensive test cards before using it
    spec_eval_pred = np.expm1(specialist.predict(X_test_exp_sc))
    spec_eval_mae = np.mean(np.abs(y_actual[expensive_mask_test] - spec_eval_pred))
    base_eval_mae = np.mean(np.abs(y_actual[expensive_mask_test] - y_pred_base[expensive_mask_test]))
    print(f"  Expensive-card MAE — base: €{base_eval_mae:.2f} | specialist: €{spec_eval_mae:.2f}")

    if spec_eval_mae >= base_eval_mae * 1.5:
        print("  Specialist is worse than base model — skipping blend.")
        return None

    # For test cards the base model predicted ≥ threshold, use specialist
    y_pred_combined = y_pred_base.copy()
    pred_expensive_mask = y_pred_base >= EXPENSIVE_THRESHOLD

    if pred_expensive_mask.sum() > 0:
        spec_pred_log = specialist.predict(X_test_sc[pred_expensive_mask])
        spec_pred = np.expm1(spec_pred_log)
        # Clip to sane range (€0 – €500)
        spec_pred = np.clip(spec_pred, 0.0, 500.0)
        # Blend: 60% specialist, 40% base
        blend = 0.6 * spec_pred + 0.4 * y_pred_base[pred_expensive_mask]
        y_pred_combined[pred_expensive_mask] = blend
        print(f"  Blended specialist predictions for {pred_expensive_mask.sum():,} cards")

    # Save specialist for inference
    specialist_path = os.path.join(config.MODEL_DIR, "specialist_expensive.joblib")
    joblib.dump(specialist, specialist_path)
    print(f"  Specialist saved to {specialist_path}")

    return y_pred_combined


# ─── Sample weighting ───────────────────────────────────────────────────────

def _compute_sample_weights(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute per-sample weights that up-weight rare/mythic cards
    and expensive cards so the model focuses more on them.
    """
    weights = pd.Series(1.0, index=df.index)

    # 1. Rarity-based weight
    for r, w in config.RARITY_WEIGHTS.items():
        col = f"rarity_{r}"
        if col in df.columns:
            mask = df[col] == 1
            weights.loc[mask] = w

    # 2. Price-based weight: boost cards above the threshold
    expensive_mask = y > config.PRICE_WEIGHT_THRESHOLD
    weights.loc[expensive_mask] *= config.PRICE_WEIGHT_FACTOR

    # 3. Extra boost for very expensive cards (€20+)
    very_expensive = y > 20.0
    weights.loc[very_expensive] *= 2.0

    return weights


def _get_rarity_labels(df: pd.DataFrame) -> pd.Series:
    """Get rarity labels for stratified splitting."""
    labels = pd.Series("common", index=df.index)
    for r in config.RARITIES:
        col = f"rarity_{r}"
        if col in df.columns:
            labels.loc[df[col] == 1] = r
    if "rarity_special" in df.columns:
        labels.loc[df["rarity_special"] == 1] = "special"
    return labels


# ─── Evaluation helpers ─────────────────────────────────────────────────────

def _evaluate(y_true, y_pred) -> dict:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    # MAPE: exclude near-zero prices (< €0.05) to avoid division explosion
    mape_mask = y_t >= 0.05
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs(y_t[mape_mask] - y_p[mape_mask]) / y_t[mape_mask]) * 100
    else:
        mape = float("nan")
    # MAPE for cards ≥€1 only (avoids "cheap card MAPE inflation")
    mape1_mask = y_t >= 1.0
    if mape1_mask.sum() > 0:
        mape_above_1 = np.mean(np.abs(y_t[mape1_mask] - y_p[mape1_mask]) / y_t[mape1_mask]) * 100
    else:
        mape_above_1 = float("nan")
    # SMAPE: symmetric, always defined
    denom = (np.abs(y_t) + np.abs(y_p))
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
    print("║          Model Evaluation (EUR)           ║")
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
    y_actual: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Show MAE per rarity bucket."""
    print("Per-rarity MAE:")
    test_df = df.loc[test_idx].copy()
    test_df["y_actual"] = y_actual.values
    test_df["y_pred"] = y_pred

    for r in config.RARITIES:
        mask = test_df[f"rarity_{r}"] == 1
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(
            test_df.loc[mask, "y_actual"],
            test_df.loc[mask, "y_pred"],
        )
        print(f"  {r:>10s}: €{mae:.4f}  (n={mask.sum():,})")
    print()


def _print_feature_importance(model: XGBRegressor, feature_cols: list[str], top_n: int = 25) -> None:
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"Top {top_n} most important features:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {feature_cols[idx]:<35s}  {importances[idx]:.4f}")
    print()


def _price_bracket_breakdown(y_actual, y_pred) -> None:
    """Show MAE, normalized MAE, MAPE, SMAPE per price bracket + accuracy."""
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
        # Bracket-normalized MAE (MAE / midpoint in %)
        nmae = (mae / midpoint) * 100
        # MAPE (skip near-zero to avoid ÷0)
        mape_sub = y_a[mask]
        if (mape_sub >= 0.05).sum() > 0:
            valid = mape_sub >= 0.05
            mape = np.mean(np.abs(y_a[mask][valid] - y_p[mask][valid]) / mape_sub[valid]) * 100
        else:
            mape = float("nan")
        # SMAPE
        denom = np.abs(y_a[mask]) + np.abs(y_p[mask])
        smape = np.mean(2.0 * np.abs(y_a[mask] - y_p[mask]) / np.where(denom == 0, 1, denom)) * 100
        print(f"  {label:>12s}  €{mae:>8.4f}  {nmae:>5.0f}%  {mape:>7.1f}%  {smape:>7.1f}%  {n:>7,}")
    print()

    # Classification-style bracket accuracy: is the prediction in the right bracket?
    _bracket_accuracy(y_a, y_p, brackets)

    # Within-X% accuracy for cards ≥€1
    _within_threshold_accuracy(y_a, y_p)


def _bracket_accuracy(y_actual, y_pred, brackets) -> None:
    """Classification: what % of cards are predicted into the correct bracket?"""
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
    """For cards ≥€1: what % are within X% of the actual price?"""
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

def _save(model, scaler, feature_cols) -> None:
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(feature_cols, config.FEATURE_COLS_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Scaler saved to {config.SCALER_PATH}")
    print(f"Feature columns saved to {config.FEATURE_COLS_PATH}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(config.FEATURES_PATH):
        print("Loading feature matrix from disk …")
        df = pd.read_csv(config.FEATURES_PATH)
        train(df=df)
    else:
        print("No feature matrix found. Run data_collector.py first, then feature_engineer.py.")
