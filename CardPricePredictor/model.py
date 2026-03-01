"""
model.py — Train, evaluate, and persist the XGBoost price-prediction model.

Pipeline:
    1. Load feature CSV produced by feature_engineer.py
    2. Compute sample weights (rarity + price-based)
    3. Split into train / test (stratified by rarity)
    4. Scale features (StandardScaler)
    5. Train an XGBRegressor with sample weights
    6. Evaluate with MAE, RMSE, R², and per-rarity breakdown
    7. Save model + scaler + column list for inference
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

    # 3. Stratified train / test split (stratify by rarity)
    rarity_strat = _get_rarity_labels(df)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_log, weights,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=rarity_strat,
    )
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

    # 7. Persist
    _save(model, scaler, feature_cols)

    return metrics


def load_model():
    """Load the trained model, scaler, and feature column list."""
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    feature_cols = joblib.load(config.FEATURE_COLS_PATH)
    return model, scaler, feature_cols


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
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def _print_metrics(m: dict) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║       Model Evaluation (EUR)         ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  MAE    : €{m['MAE']:.4f}               ║")
    print(f"║  MedAE  : €{m['MedAE']:.4f}               ║")
    print(f"║  RMSE   : €{m['RMSE']:.4f}               ║")
    print(f"║  R²     :  {m['R2']:.4f}                ║")
    print("╚══════════════════════════════════════╝\n")


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
    """Show MAE per price bracket."""
    y_a = np.array(y_actual)
    y_p = np.array(y_pred)
    brackets = [
        ("€0–€0.50", 0, 0.50),
        ("€0.50–€1", 0.50, 1),
        ("€1–€2", 1, 2),
        ("€2–€5", 2, 5),
        ("€5–€10", 5, 10),
        ("€10–€20", 10, 20),
        ("€20–€50", 20, 50),
        ("€50+", 50, 9999),
    ]
    print("Per-price-bracket MAE:")
    for label, lo, hi in brackets:
        mask = (y_a >= lo) & (y_a < hi)
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_a[mask], y_p[mask])
        print(f"  {label:>12s}: €{mae:.4f}  (n={mask.sum():,})")
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
