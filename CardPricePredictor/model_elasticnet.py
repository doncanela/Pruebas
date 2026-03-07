"""
model_elasticnet.py — Elastic Net on TF-IDF (text baseline that punches above its weight).

Uses sparse TF-IDF features (word n-grams) combined with dense numeric features.
Elastic Net is chosen over pure Lasso because MTG oracle-text tokens are
correlated (templating, synonyms), and the L2 component handles collinearity.

Pipeline:
    1. Load feature CSV (with _oracle_text metadata column)
    2. Fit a TfidfVectorizer on the training text corpus
    3. Horizontally stack sparse TF-IDF + dense scaled features
    4. Train ElasticNetCV (auto-selects alpha + l1_ratio via CV)
    5. Evaluate and save model + vectorizer + scaler
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_extraction.text import TfidfVectorizer
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

def train_elasticnet(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Full Elastic Net training pipeline. Returns evaluation metrics."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    # Exclude Reserved List cards
    if "is_reserved_list" in df.columns:
        n_rl = (df["is_reserved_list"] == 1).sum()
        df = df[df["is_reserved_list"] != 1].reset_index(drop=True)
        print(f"Excluded {n_rl:,} Reserved-List cards (trained separately).")
    print(f"Elastic Net training set: {len(df):,} cards")

    feature_cols = get_feature_columns(df)
    X_dense = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)

    # Oracle text for TF-IDF
    corpus = df["_oracle_text"].fillna("").astype(str) if "_oracle_text" in df.columns else pd.Series([""] * len(df))

    # Sample weights (V4 — centralised, capped)
    weights = compute_sample_weights(df, y)
    print(f"\nSample weight stats:")
    print(f"  min={weights.min():.2f}  max={weights.max():.2f}  "
          f"mean={weights.mean():.2f}  median={np.median(weights):.2f}")

    # Temporal split
    if "_days_since_release" in df.columns:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]

        X_train_dense = X_dense.iloc[train_idx]
        X_test_dense = X_dense.iloc[test_idx]
        y_train = y_log.iloc[train_idx]
        y_test = y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]
        corpus_train = corpus.iloc[train_idx]
        corpus_test = corpus.iloc[test_idx]

        train_days = df["_days_since_release"].iloc[train_idx]
        test_days = df["_days_since_release"].iloc[test_idx]
        print(f"Temporal split: train cards released ≥ {train_days.min()} days ago")
        print(f"  Test set covers the most recent ~{test_days.max()} → {test_days.min()} days")
    else:
        from sklearn.model_selection import train_test_split
        rarity_strat = _get_rarity_labels(df)
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(
            idx, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=rarity_strat,
        )
        X_train_dense = X_dense.iloc[tr_idx]
        X_test_dense = X_dense.iloc[te_idx]
        y_train = y_log.iloc[tr_idx]
        y_test = y_log.iloc[te_idx]
        w_train = weights.iloc[tr_idx]
        corpus_train = corpus.iloc[tr_idx]
        corpus_test = corpus.iloc[te_idx]
    print(f"Train: {len(y_train):,}   Test: {len(y_test):,}")

    # TF-IDF: fit on training text only (avoid data leakage)
    print("\nFitting TF-IDF vectorizer on training corpus …")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=5,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_train_tfidf = tfidf.fit_transform(corpus_train)
    X_test_tfidf = tfidf.transform(corpus_test)
    print(f"  TF-IDF vocabulary: {len(tfidf.vocabulary_):,} terms")

    # Scale dense features
    scaler = StandardScaler()
    X_train_dense_sc = csr_matrix(scaler.fit_transform(X_train_dense))
    X_test_dense_sc = csr_matrix(scaler.transform(X_test_dense))

    # Stack: [TF-IDF sparse | dense scaled]
    X_train_combined = hstack([X_train_tfidf, X_train_dense_sc])
    X_test_combined = hstack([X_test_tfidf, X_test_dense_sc])
    print(f"  Combined feature matrix: {X_train_combined.shape[1]:,} columns "
          f"({X_train_tfidf.shape[1]} TF-IDF + {X_train_dense_sc.shape[1]} dense)")

    # Train ElasticNetCV
    print("\nTraining ElasticNetCV (cross-validated alpha + l1_ratio) …")
    model = ElasticNetCV(**config.ELASTICNET_PARAMS)
    model.fit(X_train_combined, y_train, sample_weight=w_train.values)

    n_nonzero = np.sum(model.coef_ != 0)
    total_feats = X_train_combined.shape[1]
    print(f"  ✓ Best alpha: {model.alpha_:.6f}")
    print(f"  ✓ Best l1_ratio: {model.l1_ratio_:.3f}")
    print(f"  ✓ Non-zero coefficients: {n_nonzero}/{total_feats} "
          f"({n_nonzero / total_feats * 100:.1f}%)")

    # Evaluate
    y_pred_log = model.predict(X_test_combined)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual.values, y_pred)
    _print_metrics(metrics, "Elastic Net")
    _rarity_breakdown(df, X_test_dense.index, y_actual, y_pred)
    _price_bracket_breakdown(y_actual.values, y_pred)

    # Top TF-IDF features
    _print_top_features(model, tfidf, feature_cols)

    # Save
    joblib.dump(model, config.ELASTICNET_MODEL_PATH)
    joblib.dump(scaler, config.ELASTICNET_SCALER_PATH)
    joblib.dump(feature_cols, config.ELASTICNET_FEATURE_COLS_PATH)
    joblib.dump(tfidf, config.ELASTICNET_TFIDF_PATH)
    print(f"Elastic Net model saved to {config.ELASTICNET_MODEL_PATH}")

    return metrics


def train_elasticnet_reserved_list(
    cards: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """Train an Elastic Net model dedicated to Reserved List cards."""
    if df is None:
        if cards is None:
            raise ValueError("Provide either `cards` or `df`.")
        df = build_feature_dataframe(cards)

    if "is_reserved_list" not in df.columns:
        print("ERROR: 'is_reserved_list' column not found.")
        return {}

    df = df[df["is_reserved_list"] == 1].reset_index(drop=True)
    print(f"\nElastic Net Reserved List model — {len(df):,} cards")

    if len(df) < 50:
        print("Not enough Reserved List cards. Skipping.")
        return {}

    feature_cols = get_feature_columns(df)
    X_dense = df[feature_cols].copy()
    y = df["price_eur"].copy()
    y_log = np.log1p(y)
    corpus = df["_oracle_text"].fillna("").astype(str) if "_oracle_text" in df.columns else pd.Series([""] * len(df))

    weights = pd.Series(1.0, index=df.index)
    weights.loc[y > 10.0] *= 3.0
    weights.loc[y > 100.0] *= 2.0

    # Temporal split
    if "_days_since_release" in df.columns and len(df) >= 100:
        sort_idx = df["_days_since_release"].values.argsort()[::-1]
        n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
        train_idx = sort_idx[:n_train]
        test_idx = sort_idx[n_train:]
        X_train_dense, X_test_dense = X_dense.iloc[train_idx], X_dense.iloc[test_idx]
        y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]
        w_train = weights.iloc[train_idx]
        corpus_train, corpus_test = corpus.iloc[train_idx], corpus.iloc[test_idx]
        print(f"  Temporal split: train {len(y_train):,}  test {len(y_test):,}")
    else:
        from sklearn.model_selection import train_test_split as _tts
        idx = np.arange(len(df))
        tr_idx, te_idx = _tts(idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        X_train_dense, X_test_dense = X_dense.iloc[tr_idx], X_dense.iloc[te_idx]
        y_train, y_test = y_log.iloc[tr_idx], y_log.iloc[te_idx]
        w_train = weights.iloc[tr_idx]
        corpus_train, corpus_test = corpus.iloc[tr_idx], corpus.iloc[te_idx]
        print(f"  Random split: train {len(y_train):,}  test {len(y_test):,}")

    # TF-IDF
    tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=3000,
        sublinear_tf=True, min_df=2, max_df=0.95, strip_accents="unicode",
    )
    X_train_tfidf = tfidf.fit_transform(corpus_train)
    X_test_tfidf = tfidf.transform(corpus_test)

    scaler = StandardScaler()
    X_train_dense_sc = csr_matrix(scaler.fit_transform(X_train_dense))
    X_test_dense_sc = csr_matrix(scaler.transform(X_test_dense))

    X_train_combined = hstack([X_train_tfidf, X_train_dense_sc])
    X_test_combined = hstack([X_test_tfidf, X_test_dense_sc])

    print("\n  Training Elastic Net RL model …")
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        cv=min(5, len(X_train_dense)),
        max_iter=30_000, tol=1e-4,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train_combined, y_train, sample_weight=w_train.values)

    n_nonzero = np.sum(model.coef_ != 0)
    print(f"  Best alpha: {model.alpha_:.6f}, l1_ratio: {model.l1_ratio_:.3f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{X_train_combined.shape[1]}")

    y_pred_log = model.predict(X_test_combined)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    y_actual = np.expm1(y_test)

    metrics = _evaluate(y_actual.values if hasattr(y_actual, 'values') else y_actual, y_pred)
    print("\n  ── Elastic Net RL Evaluation ──")
    _print_metrics(metrics, "Elastic Net RL")
    _price_bracket_breakdown(y_actual.values if hasattr(y_actual, 'values') else y_actual, y_pred)

    joblib.dump(model, config.ELASTICNET_RL_MODEL_PATH)
    joblib.dump(scaler, config.ELASTICNET_RL_SCALER_PATH)
    joblib.dump(feature_cols, config.ELASTICNET_RL_FEATURE_COLS_PATH)
    joblib.dump(tfidf, config.ELASTICNET_RL_TFIDF_PATH)
    print(f"  Elastic Net RL model saved to {config.ELASTICNET_RL_MODEL_PATH}")

    return metrics


def load_elasticnet_model():
    model = joblib.load(config.ELASTICNET_MODEL_PATH)
    scaler = joblib.load(config.ELASTICNET_SCALER_PATH)
    feature_cols = joblib.load(config.ELASTICNET_FEATURE_COLS_PATH)
    tfidf = joblib.load(config.ELASTICNET_TFIDF_PATH)
    return model, scaler, feature_cols, tfidf


def load_elasticnet_reserved_list_model():
    model = joblib.load(config.ELASTICNET_RL_MODEL_PATH)
    scaler = joblib.load(config.ELASTICNET_RL_SCALER_PATH)
    feature_cols = joblib.load(config.ELASTICNET_RL_FEATURE_COLS_PATH)
    tfidf = joblib.load(config.ELASTICNET_RL_TFIDF_PATH)
    return model, scaler, feature_cols, tfidf


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def _print_metrics(m, title="Elastic Net"):
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


def _print_top_features(model, tfidf, dense_cols, top_n=25):
    """Print top features by absolute coefficient (TF-IDF names + dense names)."""
    tfidf_names = tfidf.get_feature_names_out().tolist()
    all_names = tfidf_names + list(dense_cols)
    coefs = model.coef_
    abs_coefs = np.abs(coefs)
    top_idx = np.argsort(abs_coefs)[-top_n:][::-1]

    n_nonzero = np.sum(coefs != 0)
    n_tfidf_nz = np.sum(coefs[:len(tfidf_names)] != 0)
    n_dense_nz = np.sum(coefs[len(tfidf_names):] != 0)
    print(f"Feature selection: {n_nonzero} non-zero "
          f"({n_tfidf_nz} TF-IDF + {n_dense_nz} dense)")

    print(f"\nTop {top_n} features by |coefficient|:")
    for i, idx in enumerate(top_idx, 1):
        sign = "+" if coefs[idx] >= 0 else "−"
        name = all_names[idx] if idx < len(all_names) else f"feature_{idx}"
        src = "tfidf" if idx < len(tfidf_names) else "dense"
        print(f"  {i:2d}. [{src:>5s}] {name:<35s}  {sign}{abs_coefs[idx]:.4f}")
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
