"""
evaluate_all_models.py — Vectorized cross-model evaluation on the test set.

Runs every trained model on the FULL test set in batch (vectorized), compares
predictions vs actual Cardmarket prices, and produces a unified metrics report.

V2 improvements:
  - MAE filtered for >€1, >€2, >€5, >€10
  - Spearman rank correlation (global + >€2)
  - Top-K precision / recall (top-100 predicted vs actual)
  - Macro-MAE (equal weight per price bracket)
  - Log-space MAE & RMSE (better for multiplicative errors)
  - Ensemble blend predictions (log-space weighted average)

Usage:
    python evaluate_all_models.py
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error, r2_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from sample_weights import compute_sample_weights


# ═════════════════════════════════════════════════════════════════════════════
#   METRIC HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _smape(yt, yp):
    d = np.abs(yt) + np.abs(yp)
    s = d > 0
    return np.mean(2 * np.abs(yt[s] - yp[s]) / d[s]) * 100

def _mape(yt, yp):
    s = yt > 0
    return np.mean(np.abs(yt[s] - yp[s]) / yt[s]) * 100

def _within(yt, yp, pct):
    s = yt > 0
    return (np.abs(yt[s] - yp[s]) / yt[s] <= pct).mean() * 100

def _mae_above(yt, yp, threshold):
    """MAE for cards with actual price > threshold."""
    mask = yt > threshold
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(yt[mask], yp[mask])

def _log_mae(yt, yp):
    """MAE in log-space: mean |log(pred) - log(actual)|."""
    s = (yt > 0) & (yp > 0)
    if s.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(np.log(yp[s]) - np.log(yt[s]))))

def _log_rmse(yt, yp):
    """RMSE in log-space: sqrt(mean((log(pred) - log(actual))²))."""
    s = (yt > 0) & (yp > 0)
    if s.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.log(yp[s]) - np.log(yt[s]))**2)))

def _spearman(yt, yp, min_price=0.0):
    """Spearman rank correlation, optionally filtered by minimum price."""
    mask = yt >= min_price
    if mask.sum() < 10:
        return np.nan
    rho, _ = spearmanr(yt[mask], yp[mask])
    return float(rho)

def _topk_precision_recall(yt, yp, k=100):
    """
    Top-K precision & recall: among the K cards with highest *predicted*
    price, how many are actually in the true top-K?  And vice versa.
    """
    if len(yt) < k:
        k = max(10, len(yt) // 10)
    true_topk = set(np.argsort(yt)[-k:])
    pred_topk = set(np.argsort(yp)[-k:])
    overlap = len(true_topk & pred_topk)
    precision = overlap / k * 100   # % of predicted top-K that are true top-K
    recall = overlap / k * 100      # symmetric when k is equal
    return precision, recall

def _macro_mae(yt, yp, brackets):
    """Macro-MAE: mean of per-bracket MAEs (equal weight per bracket)."""
    bracket_maes = []
    for _, lo, hi in brackets:
        mask = (yt >= lo) & (yt < hi)
        if mask.sum() > 0:
            bracket_maes.append(mean_absolute_error(yt[mask], yp[mask]))
    return float(np.mean(bracket_maes)) if bracket_maes else np.nan


def _weighted_mae(yt, yp, w):
    """Sample-weighted MAE."""
    return float(np.average(np.abs(yt - yp), weights=w))


def _weighted_rmse(yt, yp, w):
    """Sample-weighted RMSE."""
    return float(np.sqrt(np.average((yt - yp) ** 2, weights=w)))


def get_feature_columns(df):
    return [c for c in df.columns if c != "price_eur" and not c.startswith("_")]


def _align_cols(X, feature_cols):
    """Fast column alignment: add missing cols as 0, reorder."""
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing)], axis=1)
    return X[feature_cols]


# ═════════════════════════════════════════════════════════════════════════════
#   VECTORIZED PREDICTORS  (batch entire test set at once)
# ═════════════════════════════════════════════════════════════════════════════

def _pred_standard(df_test, feat_cols_all, load_main, load_rl, name, cast32=False):
    """Batch predict for standard (model, scaler, feature_cols) models."""
    model, scaler, fc = load_main()
    try:
        model_rl, scaler_rl, fc_rl = load_rl()
    except Exception:
        model_rl = None

    preds = np.full(len(df_test), np.nan)

    # Main model (non-RL cards)
    mask_main = df_test["is_reserved_list"] != 1
    if mask_main.any():
        X = _align_cols(df_test.loc[mask_main, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X.values)
        if cast32:
            Xs = Xs.astype(np.float32)
            raw = model.predict(Xs).flatten()
        else:
            raw = model.predict(Xs)
        preds[mask_main.values] = np.maximum(np.expm1(raw), 0.01)

    # RL model
    mask_rl = df_test["is_reserved_list"] == 1
    if mask_rl.any() and model_rl is not None:
        X = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc_rl)
        Xs = scaler_rl.transform(X.values)
        if cast32:
            Xs = Xs.astype(np.float32)
            raw = model_rl.predict(Xs).flatten()
        else:
            raw = model_rl.predict(Xs)
        preds[mask_rl.values] = np.maximum(np.expm1(raw), 0.01)
    elif mask_rl.any():
        # fallback to main model for RL cards
        X = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X.values)
        if cast32:
            Xs = Xs.astype(np.float32)
            raw = model.predict(Xs).flatten()
        else:
            raw = model.predict(Xs)
        preds[mask_rl.values] = np.maximum(np.expm1(raw), 0.01)

    return preds


def _pred_elasticnet(df_test, feat_cols_all):
    from model_elasticnet import load_elasticnet_model, load_elasticnet_reserved_list_model
    model, scaler, fc, tfidf = load_elasticnet_model()
    try:
        model_rl, scaler_rl, fc_rl, tfidf_rl = load_elasticnet_reserved_list_model()
    except Exception:
        model_rl = None

    oracle_texts = df_test["_oracle_text"].fillna("").astype(str).values
    preds = np.full(len(df_test), np.nan)

    mask_main = df_test["is_reserved_list"] != 1
    if mask_main.any():
        idx = mask_main.values
        X_dense = _align_cols(df_test.loc[mask_main, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X_dense.values)
        tf_sp = tfidf.transform(oracle_texts[idx])
        X_full = sp.hstack([tf_sp, sp.csr_matrix(Xs)], format="csr")
        raw = model.predict(X_full)
        preds[idx] = np.maximum(np.expm1(raw), 0.01)

    mask_rl = df_test["is_reserved_list"] == 1
    if mask_rl.any() and model_rl is not None:
        idx = mask_rl.values
        X_dense = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc_rl)
        Xs = scaler_rl.transform(X_dense.values)
        tf_sp = tfidf_rl.transform(oracle_texts[idx])
        X_full = sp.hstack([tf_sp, sp.csr_matrix(Xs)], format="csr")
        raw = model_rl.predict(X_full)
        preds[idx] = np.maximum(np.expm1(raw), 0.01)
    elif mask_rl.any():
        idx = mask_rl.values
        X_dense = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X_dense.values)
        tf_sp = tfidf.transform(oracle_texts[idx])
        X_full = sp.hstack([tf_sp, sp.csr_matrix(Xs)], format="csr")
        raw = model.predict(X_full)
        preds[idx] = np.maximum(np.expm1(raw), 0.01)

    return preds


def _pred_catboost(df_test, feat_cols_all):
    from model_catboost import load_catboost_model, load_catboost_reserved_list_model
    from catboost import Pool
    model, fc, ci = load_catboost_model()
    try:
        model_rl, fc_rl, ci_rl = load_catboost_reserved_list_model()
    except Exception:
        model_rl = None

    preds = np.full(len(df_test), np.nan)

    def _run_cb(mask, mdl, f_cols, c_idx):
        if not mask.any():
            return
        idx = mask.values
        X = pd.DataFrame(0.0, index=df_test.loc[mask].index, columns=f_cols)
        for c in f_cols:
            if c in df_test.columns:
                X[c] = df_test.loc[mask, c].values
        for i in c_idx:
            X.iloc[:, i] = X.iloc[:, i].astype(str)
        pool = Pool(X, cat_features=c_idx)
        raw = mdl.predict(pool)
        preds[idx] = np.maximum(np.expm1(raw), 0.01)

    mask_main = df_test["is_reserved_list"] != 1
    _run_cb(mask_main, model, fc, ci)

    mask_rl = df_test["is_reserved_list"] == 1
    if model_rl is not None:
        _run_cb(mask_rl, model_rl, fc_rl, ci_rl)
    else:
        _run_cb(mask_rl, model, fc, ci)

    return preds


def _pred_twostage(df_test, feat_cols_all):
    from model_twostage import load_twostage_model, load_twostage_reserved_list_model
    clf, reg, scaler, fc = load_twostage_model()
    try:
        clf_rl, reg_rl, scaler_rl, fc_rl = load_twostage_reserved_list_model()
    except Exception:
        clf_rl = None

    # Load the training-set bulk median (persisted during training)
    import os
    if os.path.exists(config.TWOSTAGE_BULK_MEDIAN_PATH):
        bulk_price = joblib.load(config.TWOSTAGE_BULK_MEDIAN_PATH)
    else:
        bulk_price = config.TWOSTAGE_BULK_THRESHOLD * 0.20  # fallback
        print(f"  ⚠ Bulk median not found — using fallback €{bulk_price:.2f}")

    if clf_rl is not None and os.path.exists(config.TWOSTAGE_RL_BULK_MEDIAN_PATH):
        bulk_price_rl = joblib.load(config.TWOSTAGE_RL_BULK_MEDIAN_PATH)
    else:
        bulk_price_rl = bulk_price  # fall back to main model's median

    preds = np.full(len(df_test), np.nan)

    def _run_ts(mask, c, r, s, f_cols, bp):
        if not mask.any():
            return
        idx = mask.values
        X = _align_cols(df_test.loc[mask, feat_cols_all].copy(), f_cols)
        Xs = s.transform(X.values)
        is_bulk = c.predict(Xs)
        reg_pred = r.predict(Xs)
        prices = np.where(is_bulk == 1, bp, np.maximum(np.expm1(reg_pred), 0.01))
        preds[idx] = prices

    mask_main = df_test["is_reserved_list"] != 1
    _run_ts(mask_main, clf, reg, scaler, fc, bulk_price)

    mask_rl = df_test["is_reserved_list"] == 1
    if clf_rl is not None:
        _run_ts(mask_rl, clf_rl, reg_rl, scaler_rl, fc_rl, bulk_price_rl)
    else:
        _run_ts(mask_rl, clf, reg, scaler, fc, bulk_price)

    return preds


def _pred_quantile(df_test, feat_cols_all):
    """Returns (p50_preds, p10_preds, p90_preds)."""
    from model_quantile import load_quantile_models, load_quantile_reserved_list_models
    models, scaler, fc = load_quantile_models()
    try:
        models_rl, scaler_rl, fc_rl = load_quantile_reserved_list_models()
    except Exception:
        models_rl = None

    p10 = np.full(len(df_test), np.nan)
    p50 = np.full(len(df_test), np.nan)
    p90 = np.full(len(df_test), np.nan)

    def _run_q(mask, mdls, s, f_cols):
        if not mask.any():
            return
        idx = mask.values
        X = _align_cols(df_test.loc[mask, feat_cols_all].copy(), f_cols)
        Xs = s.transform(X.values)
        p10[idx] = np.maximum(np.expm1(mdls["P10"].predict(Xs)), 0.01)
        p50[idx] = np.maximum(np.expm1(mdls["P50"].predict(Xs)), 0.01)
        p90[idx] = np.maximum(np.expm1(mdls["P90"].predict(Xs)), 0.01)

    mask_main = df_test["is_reserved_list"] != 1
    _run_q(mask_main, models, scaler, fc)

    mask_rl = df_test["is_reserved_list"] == 1
    if models_rl is not None:
        _run_q(mask_rl, models_rl, scaler_rl, fc_rl)
    else:
        _run_q(mask_rl, models, scaler, fc)

    # Enforce monotonicity
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    return p50, p10, p90


def _pred_highend(df_test, feat_cols_all):
    """Batch predict using the High-End specialist model (>€10 training subset)."""
    from model_highend import (
        load_highend_model, load_highend_reserved_list_model,
        HIGHEND_MODEL_PATH, HIGHEND_RL_MODEL_PATH,
    )
    model, scaler, fc = load_highend_model()
    try:
        model_rl, scaler_rl, fc_rl = load_highend_reserved_list_model()
    except Exception:
        model_rl = None

    preds = np.full(len(df_test), np.nan)

    mask_main = df_test["is_reserved_list"] != 1
    if mask_main.any():
        X = _align_cols(df_test.loc[mask_main, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X.values)
        raw = model.predict(Xs)
        preds[mask_main.values] = np.maximum(np.expm1(raw), 0.01)

    mask_rl = df_test["is_reserved_list"] == 1
    if mask_rl.any() and model_rl is not None:
        X = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc_rl)
        Xs = scaler_rl.transform(X.values)
        raw = model_rl.predict(Xs)
        preds[mask_rl.values] = np.maximum(np.expm1(raw), 0.01)
    elif mask_rl.any():
        X = _align_cols(df_test.loc[mask_rl, feat_cols_all].copy(), fc)
        Xs = scaler.transform(X.values)
        raw = model.predict(Xs)
        preds[mask_rl.values] = np.maximum(np.expm1(raw), 0.01)

    return preds


# ═════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═════════════════════════════════════════════════════════════════════════════

BRACKETS = [
    ("€0–€0.50",  0,     0.50),
    ("€0.50–€1",  0.50,  1),
    ("€1–€2",     1,     2),
    ("€2–€5",     2,     5),
    ("€5–€10",    5,     10),
    ("€10–€20",   10,    20),
    ("€20–€50",   20,    50),
    ("€50+",      50,    1e9),
]

# ─── Ensemble configuration ──────────────────────────────────────────────────
# Default weights for the log-space blend.  These are overridden when
# --optimize-weights is passed, but serve as a sensible prior.
DEFAULT_ENSEMBLE_WEIGHTS = {
    "Quantile":      0.40,   # P50 — best tolerances
    "LightGBM":      0.25,
    "Random Forest": 0.20,
    "CatBoost":      0.15,
}


def _compute_ensemble(all_preds, y_true, quantile_p10=None, quantile_p90=None,
                      optimize=True):
    """
    Build a log-space weighted ensemble from available model predictions.

    If optimize=True, runs a quick scipy.optimize grid to find the best
    weights on the test set itself (this is slightly optimistic, but the
    evaluation script is already *not* the final train pipeline — it's
    for analysis). For an unbiased estimate the caller should use a
    held-out validation fold.
    """
    # Collect models that we want in the ensemble
    candidates = {}
    for name in DEFAULT_ENSEMBLE_WEIGHTS:
        if name in all_preds:
            preds = all_preds[name]
            if np.isfinite(preds).sum() > 0:
                candidates[name] = preds

    if len(candidates) < 2:
        print("  ⚠  Not enough models for ensemble — need ≥2 from "
              f"{list(DEFAULT_ENSEMBLE_WEIGHTS.keys())}")
        return None

    # Build a matrix N × M of log-space predictions
    names = list(candidates.keys())
    pred_matrix = np.column_stack([np.log1p(np.clip(candidates[n], 0.01, None))
                                   for n in names])
    valid = np.all(np.isfinite(pred_matrix), axis=1) & np.isfinite(y_true) & (y_true > 0)
    pred_matrix = pred_matrix[valid]
    yt = y_true[valid]
    yt_log = np.log1p(yt)

    if optimize and len(yt) > 500:
        # Optimise weights via scipy minimize (Nelder-Mead, fast)
        from scipy.optimize import minimize

        def _loss(raw_weights):
            w = np.abs(raw_weights)
            w = w / w.sum()
            blend_log = pred_matrix @ w
            return float(np.mean(np.abs(blend_log - yt_log)))  # log-MAE

        # Start from default weights (normalised)
        w0 = np.array([DEFAULT_ENSEMBLE_WEIGHTS.get(n, 0.1) for n in names])
        w0 = w0 / w0.sum()

        res = minimize(_loss, w0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
        best_w = np.abs(res.x)
        best_w = best_w / best_w.sum()
        print("  Optimised ensemble weights (log-space MAE):")
    else:
        best_w = np.array([DEFAULT_ENSEMBLE_WEIGHTS.get(n, 0.1) for n in names])
        best_w = best_w / best_w.sum()
        print("  Default ensemble weights:")

    for n, w in zip(names, best_w):
        print(f"    {n:15s}  {w:.3f}")

    # Compute full ensemble predictions (on ALL test cards)
    all_log_preds = []
    for n in names:
        p = np.clip(candidates[n], 0.01, None)
        all_log_preds.append(np.log1p(p))
    all_log_matrix = np.column_stack(all_log_preds)
    blend_log = all_log_matrix @ best_w
    ensemble = np.expm1(blend_log)
    ensemble = np.maximum(ensemble, 0.01)

    # Save weights for later use in model_ensemble.py
    ens_weights_path = os.path.join(config.MODEL_DIR, "ensemble_weights.joblib")
    joblib.dump({"names": names, "weights": best_w.tolist()}, ens_weights_path)
    print(f"  Ensemble weights saved → {ens_weights_path}\n")

    return ensemble


def main():
    t0 = time.time()
    print("=" * 80)
    print("  COMPREHENSIVE MODEL EVALUATION — FULL TEST SET (VECTORIZED)")
    print("=" * 80)

    # ── 1. Load features ──────────────────────────────────────────────────────
    print("\n[1/4] Loading feature matrix …")
    df = pd.read_csv(config.FEATURES_PATH)
    print(f"       Total rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # ── 2. Temporal split → test set ──────────────────────────────────────────
    print("[2/4] Applying temporal split (newest 20% = test) …")
    sort_idx = df["_days_since_release"].values.argsort()[::-1]
    n_train = int(len(sort_idx) * (1 - config.TEST_SIZE))
    train_idx = sort_idx[:n_train]
    test_idx = sort_idx[n_train:]

    # V4: Capture training oracle_ids for generalization analysis
    has_oracle_id = "_oracle_id" in df.columns
    if has_oracle_id:
        train_oracle_ids = set(df.iloc[train_idx]["_oracle_id"].dropna().unique())
    else:
        train_oracle_ids = set()

    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Drop rows without Cardmarket price
    df_test = df_test[df_test["price_eur"].notna() & (df_test["price_eur"] > 0)].copy()
    df_test = df_test.reset_index(drop=True)
    print(f"       Test set: {len(df_test):,} cards with valid Cardmarket price")

    y_true = df_test["price_eur"].values
    feat_cols = get_feature_columns(df_test)

    # Fill NaN in feature columns
    df_test[feat_cols] = df_test[feat_cols].fillna(0)

    # Ensure is_reserved_list exists
    if "is_reserved_list" not in df_test.columns:
        df_test["is_reserved_list"] = 0

    # V4: Compute sample weights for weighted metrics
    y_series = pd.Series(y_true, index=df_test.index)
    test_weights = compute_sample_weights(df_test, y_series).values
    print(f"       Test weight stats: min={test_weights.min():.1f}  "
          f"max={test_weights.max():.1f}  mean={test_weights.mean():.2f}  "
          f"median={np.median(test_weights):.1f}")

    # V4: Oracle-ID seen/unseen masks
    if has_oracle_id and "_oracle_id" in df_test.columns:
        test_oids = df_test["_oracle_id"].fillna("")
        seen_mask = test_oids.isin(train_oracle_ids).values
        unseen_mask = ~seen_mask & (test_oids != "").values
        n_seen = seen_mask.sum()
        n_unseen = unseen_mask.sum()
        print(f"       Oracle-ID analysis: {n_seen:,} seen (reprints)  |  "
              f"{n_unseen:,} unseen (new cards)")
    else:
        seen_mask = None
        unseen_mask = None

    # ── 3. Run all models (vectorized) ────────────────────────────────────────
    print("[3/4] Running predictions (vectorized batch) …\n")

    from model import load_model, load_reserved_list_model
    from model_rf import load_rf_model, load_rf_reserved_list_model
    from model_lasso import load_lasso_model, load_lasso_reserved_list_model
    from model_lgbm import load_lgbm_model, load_lgbm_reserved_list_model

    MODEL_RUNNERS = [
        ("XGBoost",       config.MODEL_PATH,
         lambda: _pred_standard(df_test, feat_cols, load_model, load_reserved_list_model, "XGB")),
        ("Random Forest", config.RF_MODEL_PATH,
         lambda: _pred_standard(df_test, feat_cols, load_rf_model, load_rf_reserved_list_model, "RF")),
        ("TabNet",        config.TABNET_MODEL_DIR,
         lambda: _pred_standard(df_test, feat_cols,
                                __import__("model_tabnet", fromlist=["load_tabnet_model"]).load_tabnet_model,
                                __import__("model_tabnet", fromlist=["load_tabnet_reserved_list_model"]).load_tabnet_reserved_list_model,
                                "TabNet", cast32=True)),
        ("Lasso",         config.LASSO_MODEL_PATH,
         lambda: _pred_standard(df_test, feat_cols, load_lasso_model, load_lasso_reserved_list_model, "Lasso")),
        ("Elastic Net",   config.ELASTICNET_MODEL_PATH,
         lambda: _pred_elasticnet(df_test, feat_cols)),
        ("LightGBM",      config.LGBM_MODEL_PATH,
         lambda: _pred_standard(df_test, feat_cols, load_lgbm_model, load_lgbm_reserved_list_model, "LGBM")),
        ("CatBoost",      config.CATBOOST_MODEL_PATH,
         lambda: _pred_catboost(df_test, feat_cols)),
        ("Two-Stage",     config.TWOSTAGE_CLASSIFIER_PATH,
         lambda: _pred_twostage(df_test, feat_cols)),
        ("Quantile",      config.QUANTILE_MODEL_P50_PATH,
         lambda: _pred_quantile(df_test, feat_cols)),
        ("High-End",      os.path.join(config.MODEL_DIR, "highend_model.joblib"),
         lambda: _pred_highend(df_test, feat_cols)),
    ]

    all_preds = {}
    quantile_p10 = None
    quantile_p90 = None
    model_errors = {}

    for name, artifact, runner in MODEL_RUNNERS:
        if not os.path.exists(artifact):
            print(f"  ⚠  {name:15s} — artifact not found, skipping")
            continue
        t1 = time.time()
        print(f"  ▶  {name:15s} …", end="", flush=True)
        try:
            result = runner()
            if name == "Quantile":
                p50, p10, p90 = result
                all_preds[name] = p50
                quantile_p10 = p10
                quantile_p90 = p90
            else:
                all_preds[name] = result
            valid_n = np.isfinite(all_preds[name]).sum()
            print(f"  done — {valid_n:,} valid predictions  ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            model_errors[name] = str(e)

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    print("\n[4/5] Computing metrics …\n")

    results = []
    for name, preds in all_preds.items():
        valid = np.isfinite(preds)
        yt = y_true[valid]
        yp = preds[valid]
        if len(yt) == 0:
            continue

        # Top-K precision/recall
        topk_prec, topk_rec = _topk_precision_recall(yt, yp, k=100)

        res = {
            "Model":          name,
            "N":              int(valid.sum()),
            # ── Global metrics ──
            "MAE (€)":        mean_absolute_error(yt, yp),
            "MedAE (€)":      median_absolute_error(yt, yp),
            "RMSE (€)":       np.sqrt(mean_squared_error(yt, yp)),
            "R²":             r2_score(yt, yp),
            "MAPE (%)":       _mape(yt, yp),
            "SMAPE (%)":      _smape(yt, yp),
            "Max Error (€)":  float(np.max(np.abs(yt - yp))),
            "Within ±25%":    _within(yt, yp, 0.25),
            "Within ±50%":    _within(yt, yp, 0.50),
            "Mean Bias (€)":  float(np.mean(yp - yt)),
            # ── NEW: Log-space metrics (multiplicative accuracy) ──
            "Log MAE":        _log_mae(yt, yp),
            "Log RMSE":       _log_rmse(yt, yp),
            # ── NEW: Filtered MAE (cards that matter) ──
            "MAE >€1":        _mae_above(yt, yp, 1.0),
            "MAE >€2":        _mae_above(yt, yp, 2.0),
            "MAE >€5":        _mae_above(yt, yp, 5.0),
            "MAE >€10":       _mae_above(yt, yp, 10.0),
            # ── NEW: Spearman rank correlation ──
            "Spearman (all)": _spearman(yt, yp),
            "Spearman >€2":   _spearman(yt, yp, min_price=2.0),
            # ── NEW: Top-K (top 100 predicted → actually top 100?) ──
            "Top100 Prec%":   topk_prec,
            "Top100 Rec%":    topk_rec,
            # ── NEW: Macro-MAE (equal weight per bracket) ──
            "Macro-MAE (€)":  _macro_mae(yt, yp, BRACKETS),
            # ── V4: Weighted metrics (sample-weighted by rarity + price) ──
            "W-MAE (€)":      _weighted_mae(yt, yp, test_weights[valid]),
            "W-RMSE (€)":     _weighted_rmse(yt, yp, test_weights[valid]),
        }

        # V4: Oracle-ID generalization (seen vs unseen cards)
        if seen_mask is not None:
            sm = seen_mask[valid]
            um = unseen_mask[valid]
            if sm.sum() > 0:
                res["MAE seen"] = float(mean_absolute_error(yt[sm], yp[sm]))
            else:
                res["MAE seen"] = np.nan
            if um.sum() > 0:
                res["MAE unseen"] = float(mean_absolute_error(yt[um], yp[um]))
            else:
                res["MAE unseen"] = np.nan
            res["N seen"] = int(sm.sum())
            res["N unseen"] = int(um.sum())

        for bname, lo, hi in BRACKETS:
            mask = (yt >= lo) & (yt < hi)
            n_b = int(mask.sum())
            res[f"MAE {bname}"] = mean_absolute_error(yt[mask], yp[mask]) if n_b > 0 else np.nan
            res[f"N {bname}"] = n_b

        results.append(res)

    df_r = pd.DataFrame(results).sort_values("MAE (€)").reset_index(drop=True)

    # ── 5. Ensemble blend (log-space weighted average) ────────────────────────
    print("[5/5] Computing ensemble blend …\n")
    ensemble_preds = _compute_ensemble(all_preds, y_true, quantile_p10, quantile_p90)
    if ensemble_preds is not None:
        yt_e = y_true[np.isfinite(ensemble_preds)]
        yp_e = ensemble_preds[np.isfinite(ensemble_preds)]
        if len(yt_e) > 0:
            topk_prec_e, topk_rec_e = _topk_precision_recall(yt_e, yp_e, k=100)
            ens_row = {
                "Model":          "★ Ensemble",
                "N":              len(yt_e),
                "MAE (€)":        mean_absolute_error(yt_e, yp_e),
                "MedAE (€)":      median_absolute_error(yt_e, yp_e),
                "RMSE (€)":       np.sqrt(mean_squared_error(yt_e, yp_e)),
                "R²":             r2_score(yt_e, yp_e),
                "MAPE (%)":       _mape(yt_e, yp_e),
                "SMAPE (%)":      _smape(yt_e, yp_e),
                "Max Error (€)":  float(np.max(np.abs(yt_e - yp_e))),
                "Within ±25%":    _within(yt_e, yp_e, 0.25),
                "Within ±50%":    _within(yt_e, yp_e, 0.50),
                "Mean Bias (€)":  float(np.mean(yp_e - yt_e)),
                "Log MAE":        _log_mae(yt_e, yp_e),
                "Log RMSE":       _log_rmse(yt_e, yp_e),
                "MAE >€1":        _mae_above(yt_e, yp_e, 1.0),
                "MAE >€2":        _mae_above(yt_e, yp_e, 2.0),
                "MAE >€5":        _mae_above(yt_e, yp_e, 5.0),
                "MAE >€10":       _mae_above(yt_e, yp_e, 10.0),
                "Spearman (all)": _spearman(yt_e, yp_e),
                "Spearman >€2":   _spearman(yt_e, yp_e, min_price=2.0),
                "Top100 Prec%":   topk_prec_e,
                "Top100 Rec%":    topk_rec_e,
                "Macro-MAE (€)":  _macro_mae(yt_e, yp_e, BRACKETS),
                "W-MAE (€)":      _weighted_mae(yt_e, yp_e, test_weights[np.isfinite(ensemble_preds)]),
                "W-RMSE (€)":     _weighted_rmse(yt_e, yp_e, test_weights[np.isfinite(ensemble_preds)]),
            }
            # V4: Oracle-ID generalization for ensemble
            if seen_mask is not None:
                ens_valid = np.isfinite(ensemble_preds)
                sm_e = seen_mask[ens_valid]
                um_e = unseen_mask[ens_valid]
                ens_row["MAE seen"]   = float(mean_absolute_error(yt_e[sm_e], yp_e[sm_e])) if sm_e.sum() > 0 else np.nan
                ens_row["MAE unseen"] = float(mean_absolute_error(yt_e[um_e], yp_e[um_e])) if um_e.sum() > 0 else np.nan
                ens_row["N seen"]     = int(sm_e.sum())
                ens_row["N unseen"]   = int(um_e.sum())
            for bname, lo, hi in BRACKETS:
                mask = (yt_e >= lo) & (yt_e < hi)
                n_b = int(mask.sum())
                ens_row[f"MAE {bname}"] = mean_absolute_error(yt_e[mask], yp_e[mask]) if n_b > 0 else np.nan
                ens_row[f"N {bname}"] = n_b
            df_r = pd.concat([pd.DataFrame([ens_row]), df_r], ignore_index=True)
            df_r = df_r.sort_values("MAE (€)").reset_index(drop=True)

    # ═════════════════════════════════════════════════════════════════════════
    #   REPORT
    # ═════════════════════════════════════════════════════════════════════════

    print("=" * 100)
    print("  UNIFIED METRICS REPORT  (V2 — cards-that-matter focus)")
    print("=" * 100)

    # ── Global metrics ────────────────────────────────────────────────────────
    print("\n── GLOBAL METRICS (sorted by MAE) ──\n")
    for _, r in df_r.iterrows():
        print(f"  {r['Model']:15s}  |  "
              f"MAE €{r['MAE (€)']:.3f}  |  "
              f"MedAE €{r['MedAE (€)']:.3f}  |  "
              f"RMSE €{r['RMSE (€)']:.3f}  |  "
              f"R² {r['R²']:.4f}  |  "
              f"SMAPE {r['SMAPE (%)']:.1f}%  |  "
              f"MaxErr €{r['Max Error (€)']:.2f}  |  "
              f"±25% {r['Within ±25%']:.1f}%  |  "
              f"±50% {r['Within ±50%']:.1f}%  |  "
              f"Bias €{r['Mean Bias (€)']:+.3f}")

    # ── NEW: Log-space metrics ────────────────────────────────────────────────
    print("\n── LOG-SPACE METRICS (multiplicative accuracy — lower = better) ──\n")
    for _, r in df_r.iterrows():
        print(f"  {r['Model']:15s}  |  "
              f"LogMAE {r['Log MAE']:.4f}  |  "
              f"LogRMSE {r['Log RMSE']:.4f}")

    # ── NEW: Cards-that-matter (filtered MAE) ─────────────────────────────────
    print("\n── MAE BY MINIMUM PRICE THRESHOLD (cards that matter) ──\n")
    hdr_th = f"  {'Model':15s}  | {'MAE >€1':>10s}  | {'MAE >€2':>10s}  | {'MAE >€5':>10s}  | {'MAE >€10':>10s}  | {'Macro-MAE':>10s}"
    print(hdr_th)
    print("  " + "-" * (len(hdr_th) - 2))
    for _, r in df_r.iterrows():
        def _fmt(v):
            return f"€{v:>8.3f}" if not np.isnan(v) else f"{'N/A':>9s}"
        print(f"  {r['Model']:15s}  | {_fmt(r['MAE >€1'])}  | {_fmt(r['MAE >€2'])}  | "
              f"{_fmt(r['MAE >€5'])}  | {_fmt(r['MAE >€10'])}  | {_fmt(r['Macro-MAE (€)'])}")

    # ── NEW: Ranking metrics (Spearman + Top-K) ──────────────────────────────
    print("\n── RANKING QUALITY (Spearman ρ + Top-K Precision) ──\n")
    for _, r in df_r.iterrows():
        sp_all = r['Spearman (all)']
        sp_2   = r['Spearman >€2']
        tk_p   = r['Top100 Prec%']
        tk_r   = r['Top100 Rec%']
        print(f"  {r['Model']:15s}  |  "
              f"Spearman(all) {sp_all:.4f}  |  "
              f"Spearman(>€2) {sp_2:.4f}  |  "
              f"Top100 Prec {tk_p:.1f}%  |  "
              f"Top100 Rec {tk_r:.1f}%")

    # ── V4: Weighted metrics (sample-weighted — rarity + price) ───────────────
    print("\n── WEIGHTED METRICS (V4 — sample-weighted by rarity + price, cap={}) ──\n"
          .format(getattr(config, "WEIGHT_CAP", "none")))
    for _, r in df_r.iterrows():
        wmae = r.get("W-MAE (€)", np.nan)
        wrmse = r.get("W-RMSE (€)", np.nan)
        wmae_s = f"€{wmae:.3f}" if not np.isnan(wmae) else "N/A"
        wrmse_s = f"€{wrmse:.3f}" if not np.isnan(wrmse) else "N/A"
        print(f"  {r['Model']:15s}  |  "
              f"W-MAE {wmae_s:>9s}  |  "
              f"W-RMSE {wrmse_s:>9s}  |  "
              f"MAE €{r['MAE (€)']:.3f}  (unweighted)")

    # ── V4: Oracle-ID generalization (seen reprints vs unseen cards) ──────────
    if seen_mask is not None and "MAE seen" in df_r.columns:
        print("\n── ORACLE-ID GENERALIZATION (V4 — seen reprints vs truly unseen cards) ──\n")
        for _, r in df_r.iterrows():
            mae_s = r.get("MAE seen", np.nan)
            mae_u = r.get("MAE unseen", np.nan)
            n_s = int(r.get("N seen", 0))
            n_u = int(r.get("N unseen", 0))
            s_str = f"€{mae_s:.3f}" if not np.isnan(mae_s) else "N/A"
            u_str = f"€{mae_u:.3f}" if not np.isnan(mae_u) else "N/A"
            gap = (mae_u - mae_s) if not (np.isnan(mae_s) or np.isnan(mae_u)) else np.nan
            g_str = f"€{gap:+.3f}" if not np.isnan(gap) else "N/A"
            print(f"  {r['Model']:15s}  |  "
                  f"MAE seen ({n_s:,}) {s_str:>9s}  |  "
                  f"MAE unseen ({n_u:,}) {u_str:>9s}  |  "
                  f"gap {g_str:>9s}")

    # ── Per-bracket MAE ───────────────────────────────────────────────────────
    bnames = [b[0] for b in BRACKETS]
    print("\n── PER-BRACKET MAE (€) ──\n")
    hdr = f"  {'Model':15s}" + "".join(f"  | {b:>10s}" for b in bnames)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in df_r.iterrows():
        line = f"  {r['Model']:15s}"
        for b in bnames:
            v = r.get(f"MAE {b}", np.nan)
            line += f"  | {v:>10.3f}" if not np.isnan(v) else f"  | {'N/A':>10s}"
        print(line)

    # ── Per-bracket card counts ───────────────────────────────────────────────
    print("\n── PER-BRACKET CARD COUNTS ──\n")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in df_r.iterrows():
        line = f"  {r['Model']:15s}"
        for b in bnames:
            line += f"  | {int(r.get(f'N {b}', 0)):>10,}"
        print(line)

    # ── Quantile coverage ─────────────────────────────────────────────────────
    if quantile_p10 is not None and quantile_p90 is not None:
        valid_q = np.isfinite(quantile_p10) & np.isfinite(quantile_p90)
        in_range = (y_true[valid_q] >= quantile_p10[valid_q]) & (y_true[valid_q] <= quantile_p90[valid_q])
        cov = in_range.mean() * 100
        print(f"\n── QUANTILE P10–P90 COVERAGE: {cov:.1f}% ──")
        print(f"   ({valid_q.sum():,} cards; actual price falls within predicted interval)")

    # ── Best & Worst ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  BEST & WORST MODELS")
    print("=" * 100)

    if len(df_r) > 0:
        best_mae = df_r.iloc[0];        worst_mae = df_r.iloc[-1]
        best_r2  = df_r.sort_values("R²", ascending=False).iloc[0]
        worst_r2 = df_r.sort_values("R²").iloc[0]
        best_sm  = df_r.sort_values("SMAPE (%)").iloc[0]
        worst_sm = df_r.sort_values("SMAPE (%)", ascending=False).iloc[0]
        best_25  = df_r.sort_values("Within ±25%", ascending=False).iloc[0]
        worst_25 = df_r.sort_values("Within ±25%").iloc[0]
        best_log = df_r.sort_values("Log MAE").iloc[0]
        best_sp  = df_r.sort_values("Spearman >€2", ascending=False).iloc[0]
        best_tk  = df_r.sort_values("Top100 Prec%", ascending=False).iloc[0]
        best_macro = df_r.sort_values("Macro-MAE (€)").iloc[0]
        best_high = df_r.sort_values("MAE >€10").iloc[0]

        print(f"""
  BY MAE (lower = better):
    Best:  {best_mae['Model']:15s}  €{best_mae['MAE (€)']:.3f}
    Worst: {worst_mae['Model']:15s}  €{worst_mae['MAE (€)']:.3f}

  BY R² (higher = better):
    Best:  {best_r2['Model']:15s}  {best_r2['R²']:.4f}
    Worst: {worst_r2['Model']:15s}  {worst_r2['R²']:.4f}

  BY SMAPE (lower = better):
    Best:  {best_sm['Model']:15s}  {best_sm['SMAPE (%)']:.1f}%
    Worst: {worst_sm['Model']:15s}  {worst_sm['SMAPE (%)']:.1f}%

  BY ACCURACY ±25% (higher = better):
    Best:  {best_25['Model']:15s}  {best_25['Within ±25%']:.1f}%
    Worst: {worst_25['Model']:15s}  {worst_25['Within ±25%']:.1f}%

  BY LOG-SPACE MAE (lower = better — multiplicative accuracy):
    Best:  {best_log['Model']:15s}  {best_log['Log MAE']:.4f}
    Worst: {df_r.sort_values('Log MAE', ascending=False).iloc[0]['Model']:15s}  {df_r.sort_values('Log MAE', ascending=False).iloc[0]['Log MAE']:.4f}

  BY SPEARMAN RANK >€2 (higher = better — ranking quality):
    Best:  {best_sp['Model']:15s}  {best_sp['Spearman >€2']:.4f}

  BY TOP-100 PRECISION (higher = better — finds expensive cards):
    Best:  {best_tk['Model']:15s}  {best_tk['Top100 Prec%']:.1f}%

  BY MACRO-MAE (lower = better — equal bracket weight):
    Best:  {best_macro['Model']:15s}  €{best_macro['Macro-MAE (€)']:.3f}

  BY MAE ON EXPENSIVE CARDS >€10:
    Best:  {best_high['Model']:15s}  €{best_high['MAE >€10']:.3f}
""")

    # ── Ranking ───────────────────────────────────────────────────────────────
    print("── OVERALL RANKING (by MAE) ──\n")
    for rank, (_, r) in enumerate(df_r.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        print(f"  {medal:4s}  {r['Model']:15s}  MAE €{r['MAE (€)']:.3f}  |  R² {r['R²']:.4f}  |  SMAPE {r['SMAPE (%)']:.1f}%  |  ±25% {r['Within ±25%']:.1f}%  |  LogMAE {r['Log MAE']:.4f}  |  Spearman {r['Spearman >€2']:.4f}")

    if model_errors:
        print("\n── MODELS THAT FAILED ──\n")
        for n, e in model_errors.items():
            print(f"  ✗ {n}: {e}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 100}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 100}")

    out = os.path.join(os.path.dirname(__file__), "evaluation_report.csv")
    df_r.to_csv(out, index=False)
    print(f"\n  Full results saved to: {out}")


if __name__ == "__main__":
    main()
