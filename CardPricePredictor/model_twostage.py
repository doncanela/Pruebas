"""
model_twostage.py — Two-stage bulk / non-bulk price model.

Prices are bimodal: bulk (≤ €0.50, ~60% of cards) vs non-bulk.
Bulk dominates error metrics, making it hard for a single regressor to
optimise both ranges simultaneously.

Architecture:
    Stage A: LightGBM binary classifier → "is this card bulk?"
    Stage B: LightGBM regressor on the non-bulk subset (log-price)

At inference:
    1. Classify the card.
    2. If bulk → predict the training-set bulk median (~€0.10).
    3. If non-bulk → run the regressor.
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
    accuracy_score,
    f1_score,
)

import config
from feature_engineer import build_feature_dataframe, get_feature_columns
from sample_weights import compute_sample_weights


BULK_THRESHOLD = config.TWOSTAGE_BULK_THRESHOLD  # €0.50


# ─── Public API ──────────────────────────────────────────────────────────────

def train_twostage(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full two-stage training pipeline. Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List cards
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
    print(f"Two-stage training set: {len(df):,} cards")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)
    is_bulk = (y <= BULK_THRESHOLD).astype(int)
    print(f"  Bulk (≤€{BULK_THRESHOLD}): {is_bulk.sum():,}  "
          f"Non-bulk: {(~is_bulk.astype(bool)).sum():,}")

    # Sample weights (V4 — centralised, capped)
    weights = compute_sample_weights(df, y)

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
            idx, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
    y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
    bulk_train, bulk_test = is_bulk.iloc[train_idx], is_bulk.iloc[test_idx]
    w_train = weights.iloc[train_idx]
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Stage A: Bulk classifier ────────────────────────────────
    print("\n── Stage A: Bulk vs non-bulk classifier ──")
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train_sc, bulk_train,
        sample_weight=w_train.values,
        eval_set=[(X_test_sc, bulk_test)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(40, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    bulk_pred = clf.predict(X_test_sc)
    bulk_proba = clf.predict_proba(X_test_sc)[:, 1]  # P(bulk)

    acc = accuracy_score(bulk_test, bulk_pred)
    f1 = f1_score(bulk_test, bulk_pred)
    print(f"  Classifier accuracy: {acc:.3f}")
    print(f"  F1 (bulk class): {f1:.3f}")
    print(f"  Predicted bulk: {bulk_pred.sum():,} / {len(bulk_pred):,}")

    # ── Stage B: Non-bulk regressor ─────────────────────────────
    print("\n── Stage B: Non-bulk regressor ──")
    nonbulk_mask_train = bulk_train == 0
    n_nonbulk_train = nonbulk_mask_train.sum()
    print(f"  Training regressor on {n_nonbulk_train:,} non-bulk cards")

    X_train_nb = X_train_sc[nonbulk_mask_train.values]
    y_train_nb = y_train.iloc[nonbulk_mask_train.values.nonzero()[0]]
    w_train_nb = w_train.iloc[nonbulk_mask_train.values.nonzero()[0]]

    # Eval set: non-bulk test cards
    nonbulk_mask_test = bulk_test == 0
    X_test_nb = X_test_sc[nonbulk_mask_test.values]
    y_test_nb = y_test.iloc[nonbulk_mask_test.values.nonzero()[0]]

    reg = lgb.LGBMRegressor(
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
        objective="huber",
    )
    reg.fit(
        X_train_nb, y_train_nb,
        sample_weight=w_train_nb.values,
        eval_set=[(X_test_nb, y_test_nb)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(80, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    best = reg.best_iteration_ if hasattr(reg, "best_iteration_") else 3000
    print(f"  Regressor trained. Best iteration: {best}")

    # ── Combined prediction ─────────────────────────────────────
    print("\n── Combined two-stage prediction ──")
    bulk_median = y_train_raw[bulk_train == 1].median()
    print(f"  Bulk median price: €{bulk_median:.2f}")

    y_pred_combined = np.zeros(len(X_test_sc))
    # All cards classified as bulk get the bulk median
    pred_bulk_mask = bulk_pred == 1
    y_pred_combined[pred_bulk_mask] = bulk_median

    # Non-bulk cards get the regressor prediction
    pred_nonbulk_mask = ~pred_bulk_mask
    if pred_nonbulk_mask.sum() > 0:
        nb_pred_log = reg.predict(X_test_sc[pred_nonbulk_mask])
        y_pred_combined[pred_nonbulk_mask] = np.clip(np.expm1(nb_pred_log), 0.01, None)

    y_actual = np.expm1(y_test).values

    metrics = _evaluate(y_actual, y_pred_combined)
    _print_metrics(metrics, "Two-Stage (Bulk/Non-Bulk)")
    _rarity_breakdown(df, X_test.index, pd.Series(y_actual, index=X_test.index), y_pred_combined)
    _price_bracket_breakdown(y_actual, y_pred_combined)

    # Save everything
    joblib.dump(clf, config.TWOSTAGE_CLASSIFIER_PATH)
    joblib.dump(reg, config.TWOSTAGE_REGRESSOR_PATH)
    joblib.dump(scaler, config.TWOSTAGE_SCALER_PATH)
    joblib.dump(feature_cols, config.TWOSTAGE_FEATURE_COLS_PATH)
    joblib.dump(float(bulk_median), config.TWOSTAGE_BULK_MEDIAN_PATH)
    print(f"Two-stage model saved to {config.MODEL_DIR}")
    print(f"  Bulk median: €{bulk_median:.4f} → {config.TWOSTAGE_BULK_MEDIAN_PATH}")

    return metrics


def train_twostage_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train two-stage model on Reserved List cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nTwo-Stage Reserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough Reserved List cards. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)
    is_bulk = (y <= BULK_THRESHOLD).astype(int)

    weights = pd.Series(1.0, index=df.index)
    weights.loc[y > 10.0] *= 3.0
    weights.loc[y > 100.0] *= 2.0

    if "_days_since_release" in df.columns and len(df) >= 100:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]
        print(f"  Temporal split: train {len(train_idx):,}  test {len(test_idx):,}")
    else:
        from sklearn.model_selection import train_test_split as _tts
        idx = np.arange(len(df))
        train_idx, test_idx = _tts(
            idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        )
        print(f"  Random split: train {len(train_idx):,}  test {len(test_idx):,}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
    y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]
    bulk_train, bulk_test = is_bulk.iloc[train_idx], is_bulk.iloc[test_idx]
    w_train = weights.iloc[train_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Stage A
    clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    clf.fit(
        X_train_sc, bulk_train, sample_weight=w_train.values,
        eval_set=[(X_test_sc, bulk_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    bulk_pred = clf.predict(X_test_sc)

    # Stage B
    nonbulk_train = bulk_train == 0
    X_train_nb = X_train_sc[nonbulk_train.values]
    y_train_nb = y_train.iloc[nonbulk_train.values.nonzero()[0]]
    w_train_nb = w_train.iloc[nonbulk_train.values.nonzero()[0]]

    nonbulk_test = bulk_test == 0
    X_test_nb = X_test_sc[nonbulk_test.values]
    y_test_nb = y_test.iloc[nonbulk_test.values.nonzero()[0]]

    reg = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=3.0, reg_lambda=8.0,
        min_child_samples=5, n_jobs=-1, random_state=42, verbose=-1,
    )
    if len(y_train_nb) >= 20 and len(y_test_nb) >= 5:
        reg.fit(
            X_train_nb, y_train_nb, sample_weight=w_train_nb.values,
            eval_set=[(X_test_nb, y_test_nb)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        reg.fit(X_train_nb, y_train_nb, sample_weight=w_train_nb.values)

    # Combined
    bulk_median = y_train_raw[bulk_train == 1].median() if (bulk_train == 1).sum() > 0 else 0.10
    y_pred = np.zeros(len(X_test_sc))
    pred_bulk_mask = bulk_pred == 1
    y_pred[pred_bulk_mask] = bulk_median
    if (~pred_bulk_mask).sum() > 0:
        y_pred[~pred_bulk_mask] = np.clip(np.expm1(reg.predict(X_test_sc[~pred_bulk_mask])), 0.01, None)

    y_actual = np.expm1(y_test).values if hasattr(y_test, 'values') else np.expm1(y_test)

    metrics = _evaluate(y_actual, y_pred)
    print("\n  ── Two-Stage RL Evaluation ──")
    _print_metrics(metrics, "Two-Stage RL")
    _price_bracket_breakdown(y_actual, y_pred)

    joblib.dump(clf, config.TWOSTAGE_RL_CLASSIFIER_PATH)
    joblib.dump(reg, config.TWOSTAGE_RL_REGRESSOR_PATH)
    joblib.dump(scaler, config.TWOSTAGE_RL_SCALER_PATH)
    joblib.dump(feature_cols, config.TWOSTAGE_RL_FEATURE_COLS_PATH)
    joblib.dump(float(bulk_median), config.TWOSTAGE_RL_BULK_MEDIAN_PATH)
    print(f"  Two-Stage RL saved to {config.MODEL_DIR}")
    print(f"  RL Bulk median: €{bulk_median:.4f} → {config.TWOSTAGE_RL_BULK_MEDIAN_PATH}")

    return metrics


def load_twostage_model():
    clf = joblib.load(config.TWOSTAGE_CLASSIFIER_PATH)
    reg = joblib.load(config.TWOSTAGE_REGRESSOR_PATH)
    scaler = joblib.load(config.TWOSTAGE_SCALER_PATH)
    feature_cols = joblib.load(config.TWOSTAGE_FEATURE_COLS_PATH)
    return clf, reg, scaler, feature_cols


def load_twostage_reserved_list_model():
    clf = joblib.load(config.TWOSTAGE_RL_CLASSIFIER_PATH)
    reg = joblib.load(config.TWOSTAGE_RL_REGRESSOR_PATH)
    scaler = joblib.load(config.TWOSTAGE_RL_SCALER_PATH)
    feature_cols = joblib.load(config.TWOSTAGE_RL_FEATURE_COLS_PATH)
    return clf, reg, scaler, feature_cols


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def _print_metrics(m, title="Two-Stage"):
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
