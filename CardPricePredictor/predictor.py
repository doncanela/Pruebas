"""
predictor.py — Predict the Cardmarket EUR price of a single card.

Given a card name (or a Scryfall card dict), this module:
    1. Fetches the card from Scryfall (if only a name is given)
    2. Runs feature engineering on it
    3. Scales and feeds it to the trained model
    4. Returns the predicted price in EUR

This is the inference entry point — the model must have been trained first.
"""

import json
import os
import time
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import requests

import config
from feature_engineer import _card_to_row, apply_tfidf_svd
from model import load_model


# ─── Public API ──────────────────────────────────────────────────────────────

def predict_card(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict the Cardmarket EUR price for a given card.

    Parameters
    ----------
    card_name : English card name (looked up via Scryfall fuzzy search).
    card_dict : Already-fetched Scryfall JSON dict (skips API call).
    set_code  : Optional 3-letter set code to disambiguate reprints.
    verbose   : Print a summary.

    Returns
    -------
    dict with keys:
        card_name, set, predicted_price_eur, current_price_eur, features
    """
    # 1. Resolve card data
    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    # Enrich with reprint info
    card_dict = _enrich_single_card(card_dict)

    # 2. Feature-engineer
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    # 3. Load model — use Reserved List model if the card is on the RL
    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.RL_MODEL_PATH):
        from model import load_reserved_list_model
        model, scaler, feature_cols = load_reserved_list_model()
        model_used = "reserved_list"
    else:
        model, scaler, feature_cols = load_model()
        model_used = "main"

    # Ensure columns match training order (fill missing with 0)
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    # 4. Scale & predict
    X = scaler.transform(row_df)
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))
    pred_price = max(pred_price, 0.01)  # floor at 1 cent

    # Current price for comparison
    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
    }

    if verbose:
        _print_prediction(result)

    # Log prediction to history (local file + Neon DB)
    try:
        from price_history import log_prediction
        log_prediction(result)
    except Exception:
        pass  # Don't let logging failures break predictions
    try:
        from db import insert_prediction
        insert_prediction(result)
    except Exception:
        pass

    return result


def predict_batch(card_names: list[str], set_code: Optional[str] = None) -> list[dict]:
    """Predict prices for multiple cards."""
    results = []
    for name in card_names:
        try:
            r = predict_card(card_name=name, set_code=set_code, verbose=False)
            results.append(r)
        except Exception as e:
            results.append({"card_name": name, "error": str(e)})
        time.sleep(config.SCRYFALL_DELAY_SEC)
    return results


# ─── Lasso prediction ───────────────────────────────────────────────────────

def predict_card_lasso(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict the Cardmarket EUR price using the Lasso model.
    Same interface as predict_card but uses Lasso instead of XGBoost.
    """
    from model_lasso import load_lasso_model, load_lasso_reserved_list_model

    # 1. Resolve card data
    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)

    # 2. Feature-engineer
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    # 3. Load Lasso model — use RL variant if the card is on the Reserved List
    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.LASSO_RL_MODEL_PATH):
        model, scaler, feature_cols = load_lasso_reserved_list_model()
        model_used = "lasso_reserved_list"
    else:
        model, scaler, feature_cols = load_lasso_model()
        model_used = "lasso_main"

    # Ensure columns match training order
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    # 4. Scale & predict
    X = scaler.transform(row_df)
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))
    pred_price = max(pred_price, 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [Lasso model: {model_used}]")
        _print_prediction(result)

    return result


# ─── Random Forest prediction ───────────────────────────────────────────────

def predict_card_rf(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the Random Forest model."""
    from model_rf import load_rf_model, load_rf_reserved_list_model

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.RF_RL_MODEL_PATH):
        model, scaler, feature_cols = load_rf_reserved_list_model()
        model_used = "rf_reserved_list"
    else:
        model, scaler, feature_cols = load_rf_model()
        model_used = "rf_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df)
    pred_log = model.predict(X)[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [Random Forest model: {model_used}]")
        _print_prediction(result)

    return result


# ─── TabNet prediction ──────────────────────────────────────────────────────

def predict_card_tabnet(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the TabNet model."""
    from model_tabnet import load_tabnet_model, load_tabnet_reserved_list_model

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(os.path.join(config.TABNET_RL_MODEL_DIR, "tabnet_rl.zip")):
        model, scaler, feature_cols = load_tabnet_reserved_list_model()
        model_used = "tabnet_reserved_list"
    else:
        model, scaler, feature_cols = load_tabnet_model()
        model_used = "tabnet_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df).astype(np.float32)
    pred_log = model.predict(X).flatten()[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [TabNet model: {model_used}]")
        _print_prediction(result)

    return result


# ─── Elastic Net prediction ─────────────────────────────────────────────────

def predict_card_elasticnet(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the Elastic Net (TF-IDF) model."""
    from model_elasticnet import load_elasticnet_model, load_elasticnet_reserved_list_model
    import scipy.sparse as sp

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.ELASTICNET_RL_MODEL_PATH):
        model, scaler, feature_cols, tfidf = load_elasticnet_reserved_list_model()
        model_used = "elasticnet_reserved_list"
    else:
        model, scaler, feature_cols, tfidf = load_elasticnet_model()
        model_used = "elasticnet_main"

    # Build sparse TF-IDF + dense feature matrix
    oracle_text = card_dict.get("oracle_text", "")
    tfidf_sparse = tfidf.transform([oracle_text])

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    dense = scaler.transform(row_df[feature_cols])
    dense_sparse = sp.csr_matrix(dense)
    X = sp.hstack([tfidf_sparse, dense_sparse], format="csr")

    pred_log = model.predict(X)[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [Elastic Net model: {model_used}]")
        _print_prediction(result)

    return result


# ─── LightGBM prediction ────────────────────────────────────────────────────

def predict_card_lgbm(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the LightGBM model."""
    from model_lgbm import load_lgbm_model, load_lgbm_reserved_list_model

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.LGBM_RL_MODEL_PATH):
        model, scaler, feature_cols = load_lgbm_reserved_list_model()
        model_used = "lgbm_reserved_list"
    else:
        model, scaler, feature_cols = load_lgbm_model()
        model_used = "lgbm_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df)
    pred_log = model.predict(X)[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [LightGBM model: {model_used}]")
        _print_prediction(result)

    return result


# ─── CatBoost prediction ────────────────────────────────────────────────────

def predict_card_catboost(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the CatBoost model."""
    from model_catboost import load_catboost_model, load_catboost_reserved_list_model

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.CATBOOST_RL_MODEL_PATH):
        model, feature_cols, cat_indices = load_catboost_reserved_list_model()
        model_used = "catboost_reserved_list"
    else:
        model, feature_cols, cat_indices = load_catboost_model()
        model_used = "catboost_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols].copy()

    # Cast categorical columns to str for CatBoost
    for idx in cat_indices:
        col = feature_cols[idx]
        row_df[col] = row_df[col].astype(str)

    from catboost import Pool
    pool = Pool(row_df, cat_features=cat_indices)
    pred_log = model.predict(pool)[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [CatBoost model: {model_used}]")
        _print_prediction(result)

    return result


# ─── Two-Stage prediction ───────────────────────────────────────────────────

def predict_card_twostage(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict Cardmarket EUR price using the Two-Stage bulk/non-bulk model."""
    from model_twostage import load_twostage_model, load_twostage_reserved_list_model

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.TWOSTAGE_RL_CLASSIFIER_PATH):
        clf, reg, scaler, feature_cols = load_twostage_reserved_list_model()
        model_used = "twostage_reserved_list"
    else:
        clf, reg, scaler, feature_cols = load_twostage_model()
        model_used = "twostage_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df)
    is_bulk = clf.predict(X)[0]

    if is_bulk == 1:
        # Use the training-set bulk median (persisted during training)
        if os.path.exists(config.TWOSTAGE_BULK_MEDIAN_PATH):
            pred_price = joblib.load(config.TWOSTAGE_BULK_MEDIAN_PATH)
        else:
            pred_price = config.TWOSTAGE_BULK_THRESHOLD * 0.20  # fallback ~€0.10
    else:
        pred_log = reg.predict(X)[0]
        pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
        "is_bulk": bool(is_bulk),
    }

    if verbose:
        bulk_tag = " [BULK]" if is_bulk else ""
        print(f"\n  [Two-Stage model: {model_used}{bulk_tag}]")
        _print_prediction(result)

    return result


# ─── Quantile prediction ────────────────────────────────────────────────────

def predict_card_quantile(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict price range using quantile regression (P10/P50/P90).
    Returns predicted_price_eur (P50) plus predicted_p10 and predicted_p90.
    """
    from model_quantile import load_quantile_models, load_quantile_reserved_list_models

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(config.QUANTILE_RL_MODEL_P50_PATH):
        models, scaler, feature_cols = load_quantile_reserved_list_models()
        model_used = "quantile_reserved_list"
    else:
        models, scaler, feature_cols = load_quantile_models()
        model_used = "quantile_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df)
    p10 = max(float(np.expm1(models["P10"].predict(X)[0])), 0.01)
    p50 = max(float(np.expm1(models["P50"].predict(X)[0])), 0.01)
    p90 = max(float(np.expm1(models["P90"].predict(X)[0])), 0.01)

    # Ensure monotonicity: P10 ≤ P50 ≤ P90
    p10 = min(p10, p50)
    p90 = max(p90, p50)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(p50, 2),
        "predicted_p10": round(p10, 2),
        "predicted_p90": round(p90, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [Quantile model: {model_used}]")
        _print_prediction(result)
        print(f"  Price range: €{p10:.2f} (P10) – €{p50:.2f} (P50) – €{p90:.2f} (P90)")
        print()

    return result


def predict_card_highend(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Predict price using the High-End specialist model (trained on >€10)."""
    from model_highend import (
        load_highend_model, load_highend_reserved_list_model,
        HIGHEND_MODEL_PATH, HIGHEND_RL_MODEL_PATH,
    )

    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)
    row = _card_to_row(card_dict)
    row_df = pd.DataFrame([row])
    row_df = apply_tfidf_svd(row_df, card_dict.get("oracle_text", ""))

    is_reserved = card_dict.get("reserved", False)
    if is_reserved and os.path.exists(HIGHEND_RL_MODEL_PATH):
        model, scaler, feature_cols = load_highend_reserved_list_model()
        model_used = "highend_reserved_list"
    else:
        model, scaler, feature_cols = load_highend_model()
        model_used = "highend_main"

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    X = scaler.transform(row_df)
    pred_log = model.predict(X)[0]
    pred_price = max(float(np.expm1(pred_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": model_used,
    }

    if verbose:
        print(f"\n  [High-End model: {model_used}]")
        _print_prediction(result)

    return result


def predict_card_ensemble(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict price using a log-space weighted ensemble of top models.
    Uses weights from the evaluation optimization (ensemble_weights.joblib).
    Falls back to default weights if the optimized file doesn't exist.
    """
    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)

    # Load weights
    ens_weights_path = os.path.join(config.MODEL_DIR, "ensemble_weights.joblib")
    if os.path.exists(ens_weights_path):
        info = joblib.load(ens_weights_path)
        names = info["names"]
        weights = np.array(info["weights"])
    else:
        # Default fallback
        names = ["Quantile", "LightGBM", "Random Forest", "CatBoost"]
        weights = np.array([0.40, 0.25, 0.20, 0.15])

    # Map model names → predict functions
    _name_to_func = {
        "Quantile":      predict_card_quantile,
        "LightGBM":      predict_card_lgbm,
        "Random Forest":  predict_card_rf,
        "CatBoost":      predict_card_catboost,
        "XGBoost":       predict_card,
        "Two-Stage":     predict_card_twostage,
    }

    log_preds = []
    used_weights = []
    used_names = []

    for name, w in zip(names, weights):
        fn = _name_to_func.get(name)
        if fn is None:
            continue
        try:
            r = fn(card_dict=card_dict, verbose=False)
            pred = r.get("predicted_price_eur")
            if pred is not None and pred > 0:
                log_preds.append(np.log1p(pred))
                used_weights.append(w)
                used_names.append(name)
        except Exception:
            continue

    if not log_preds:
        raise RuntimeError("No ensemble models available.")

    # Normalise weights in case some models failed
    w_arr = np.array(used_weights)
    w_arr = w_arr / w_arr.sum()

    blend_log = np.dot(log_preds, w_arr)
    pred_price = max(float(np.expm1(blend_log)), 0.01)

    current = card_dict.get("prices", {}).get(config.PRICE_FIELD)
    current_price = float(current) if current else None

    result = {
        "card_name": card_dict.get("name", "Unknown"),
        "set": card_dict.get("set_name", "Unknown"),
        "set_code": card_dict.get("set", "???"),
        "rarity": card_dict.get("rarity", "unknown"),
        "predicted_price_eur": round(pred_price, 2),
        "current_price_eur": round(current_price, 2) if current_price else None,
        "mana_cost": card_dict.get("mana_cost", ""),
        "type_line": card_dict.get("type_line", ""),
        "model_used": f"ensemble({'+'.join(used_names)})",
        "ensemble_weights": dict(zip(used_names, w_arr.tolist())),
    }

    if verbose:
        print(f"\n  [Ensemble: {' + '.join(f'{n}({w:.0%})' for n, w in zip(used_names, w_arr))}]")
        _print_prediction(result)

    return result


# ─── All-models prediction ──────────────────────────────────────────────────

_MODEL_REGISTRY = [
    ("XGBoost",    "predict_card",           config.MODEL_PATH),
    ("RF",         "predict_card_rf",        config.RF_MODEL_PATH),
    ("TabNet",     "predict_card_tabnet",    config.TABNET_MODEL_DIR),
    ("Lasso",      "predict_card_lasso",     config.LASSO_MODEL_PATH),
    ("ElasticNet", "predict_card_elasticnet", config.ELASTICNET_MODEL_PATH),
    ("LightGBM",   "predict_card_lgbm",     config.LGBM_MODEL_PATH),
    ("CatBoost",   "predict_card_catboost",  config.CATBOOST_MODEL_PATH),
    ("TwoStage",   "predict_card_twostage",  config.TWOSTAGE_CLASSIFIER_PATH),
    ("Quantile",   "predict_card_quantile",  config.QUANTILE_MODEL_P50_PATH),
    ("HighEnd",    "predict_card_highend",
     os.path.join(config.MODEL_DIR, "highend_model.joblib")),
    ("Ensemble",   "predict_card_ensemble",  config.LGBM_MODEL_PATH),  # needs ≥2 base models
]


def predict_card_all(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run every available model on a single card and return all predictions.
    Fetches the card once and reuses the dict for every model.
    """
    # 1. Resolve card data once
    if card_dict is None:
        if card_name is None:
            raise ValueError("Provide either `card_name` or `card_dict`.")
        card_dict = _fetch_card(card_name, set_code)

    card_dict = _enrich_single_card(card_dict)

    # 2. Run each model that has trained artifacts
    results: list[dict] = []
    for label, func_name, artifact_path in _MODEL_REGISTRY:
        if not os.path.exists(artifact_path):
            continue
        try:
            fn = globals()[func_name]
            r = fn(card_dict=card_dict, verbose=False)
            r["model_label"] = label
            results.append(r)
        except Exception as e:
            results.append({
                "model_label": label,
                "predicted_price_eur": None,
                "error": str(e),
            })

    if not results:
        raise RuntimeError("No trained models found. Run a train command first.")

    # 3. Pretty-print comparison table
    if verbose:
        _print_all_predictions(card_dict, results)

    # 4. Log best prediction (LightGBM > XGBoost > first)
    best = (
        next((r for r in results if r.get("model_label") == "LightGBM"), None)
        or next((r for r in results if r.get("model_label") == "XGBoost"), None)
        or results[0]
    )
    try:
        from price_history import log_prediction
        log_prediction(best)
    except Exception:
        pass
    try:
        from db import insert_prediction
        insert_prediction(best)
    except Exception:
        pass

    return results


def _print_all_predictions(card: dict, results: list[dict]) -> None:
    """Print a single card header followed by a multi-model price table."""
    name     = card.get("name", "Unknown")
    set_name = card.get("set_name", "Unknown")
    mana     = card.get("mana_cost", "")
    type_l   = card.get("type_line", "")
    rarity   = card.get("rarity", "unknown")
    current  = card.get("prices", {}).get(config.PRICE_FIELD)
    cur_str  = f"€{float(current):.2f}" if current else "N/A"

    w = 78  # box width (wider for quantile range)
    bar = "═" * (w - 2)
    mid = "─" * (w - 2)

    print()
    print(f"╔{bar}╗")
    print(f"║  Card : {name:<{w-12}}║")
    print(f"║  Set  : {set_name:<{w-12}}║")
    print(f"║  Mana : {mana:<{w-12}}║")
    print(f"║  Type : {type_l:<{w-12}}║")
    print(f"║  Rare : {rarity:<{w-12}}║")
    print(f"╠{bar}╣")
    print(f"║  Current Cardmarket price : {cur_str:<{w-32}}║")
    print(f"╠{bar}╣")

    # Table header
    hdr = f"  {'Model':<12} {'Predicted':>12}  {'Diff':>10}  {'Diff %':>8}"
    print(f"║{hdr:<{w-2}}║")
    print(f"║  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}  ║")

    cur_val = float(current) if current else None
    for r in results:
        label = r.get("model_label", "?")
        pred  = r.get("predicted_price_eur")
        if pred is None:
            row = f"  {label:<12} {'ERROR':>12}  {'':>10}  {'':>8}"
        else:
            pred_s = f"€{pred:.2f}"
            if cur_val and cur_val > 0:
                diff   = pred - cur_val
                pct    = diff / cur_val * 100
                diff_s = f"{'+' if diff >= 0 else ''}{diff:.2f}"
                pct_s  = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
            else:
                diff_s = "—"
                pct_s  = "—"
            # Show range for Quantile model
            p10 = r.get("predicted_p10")
            p90 = r.get("predicted_p90")
            if p10 is not None and p90 is not None:
                range_s = f"  (€{p10:.2f}–€{p90:.2f})"
            else:
                range_s = ""
            row = f"  {label:<12} {pred_s:>12}  {diff_s:>10}  {pct_s:>8}{range_s}"
        print(f"║{row:<{w-2}}║")

    print(f"╚{bar}╝")
    print()


# ─── Scryfall fetch ──────────────────────────────────────────────────────────

def _fetch_card(name: str, set_code: Optional[str] = None) -> dict:
    """Fetch a card by name from Scryfall (fuzzy match)."""
    params = {"fuzzy": name}
    if set_code:
        params["set"] = set_code
    time.sleep(config.SCRYFALL_DELAY_SEC)
    resp = requests.get(config.SCRYFALL_CARD_URL, params=params, timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"Card not found: '{name}'")
    resp.raise_for_status()
    return resp.json()


def _enrich_single_card(card: dict) -> dict:
    """
    Look up all printings of this card and compute reprint stats.
    Uses the Scryfall prints_search_uri.
    """
    from datetime import datetime as _dt
    from statistics import stdev

    defaults = {
        "reprint_count": 0, "avg_prev_price": 0.0,
        "min_prev_price": 0.0, "max_prev_price": 0.0,
        "std_prev_price": 0.0, "price_spread": 0.0,
        "days_since_last_reprint": 0, "oldest_printing_days": 0,
        "set_card_count": 0,
    }

    prints_uri = card.get("prints_search_uri")
    if not prints_uri:
        for k, v in defaults.items():
            card.setdefault(k, v)
        return card

    # Fetch all printings
    time.sleep(config.SCRYFALL_DELAY_SEC)
    resp = requests.get(prints_uri, timeout=15)
    if resp.status_code != 200:
        for k, v in defaults.items():
            card.setdefault(k, v)
        return card

    printings = resp.json().get("data", [])
    card_release = card.get("released_at", "9999-99-99")

    other_prices = []
    other_releases = []
    for p in printings:
        if p.get("id") == card.get("id"):
            continue
        p_release = p.get("released_at", "9999-99-99")
        p_price = p.get("prices", {}).get(config.PRICE_FIELD)
        if p_release <= card_release:
            other_releases.append(p_release)
            if p_price is not None:
                other_prices.append(float(p_price))

    card["reprint_count"] = max(0, len(printings) - 1)
    card["avg_prev_price"] = sum(other_prices) / len(other_prices) if other_prices else 0.0
    card["min_prev_price"] = min(other_prices) if other_prices else 0.0
    card["max_prev_price"] = max(other_prices) if other_prices else 0.0
    card["std_prev_price"] = stdev(other_prices) if len(other_prices) >= 2 else 0.0
    card["price_spread"] = card["max_prev_price"] - card["min_prev_price"]

    # Days since last reprint
    if other_releases:
        most_recent = max(other_releases)
        try:
            delta = _dt.strptime(card_release, "%Y-%m-%d") - _dt.strptime(most_recent, "%Y-%m-%d")
            card["days_since_last_reprint"] = max(0, delta.days)
        except ValueError:
            card["days_since_last_reprint"] = 0
    else:
        card["days_since_last_reprint"] = 0

    # Age of oldest printing
    if other_releases:
        oldest = min(other_releases)
        try:
            delta = _dt.strptime(card_release, "%Y-%m-%d") - _dt.strptime(oldest, "%Y-%m-%d")
            card["oldest_printing_days"] = max(0, delta.days)
        except ValueError:
            card["oldest_printing_days"] = 0
    else:
        card["oldest_printing_days"] = 0

    card["set_card_count"] = 0  # not critical for single-card inference

    # Metagame enrichment: look up tournament usage from MTGGoldfish cache
    try:
        from metagame_collector import fetch_metagame_data, get_card_metagame
        metagame = fetch_metagame_data()  # uses cache if fresh
        card_name = card.get("name", "")
        meta_info = get_card_metagame(card_name, metagame)

        for fmt in ["standard", "pioneer", "modern", "legacy", "vintage"]:
            fmt_data = meta_info.get(fmt, {})
            card[f"meta_{fmt}_pct"] = fmt_data.get("pct", 0.0)
            card[f"meta_{fmt}_copies"] = fmt_data.get("copies", 0.0)

        card["meta_formats_played"] = len(meta_info)
        all_pcts = [v["pct"] for v in meta_info.values()] if meta_info else []
        card["meta_max_usage"] = max(all_pcts) if all_pcts else 0.0
        card["meta_avg_usage"] = (
            sum(all_pcts) / len(all_pcts) if all_pcts else 0.0
        )
        card["meta_total_usage"] = sum(all_pcts)
    except Exception:
        # Metagame data not available; features will default to 0
        pass

    return card


# ─── Pretty print ────────────────────────────────────────────────────────────

def _print_prediction(r: dict) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Card : {r['card_name']:<52s}║")
    print(f"║  Set  : {r['set']:<52s}║")
    print(f"║  Mana : {r['mana_cost']:<52s}║")
    print(f"║  Type : {r['type_line']:<52s}║")
    print(f"║  Rare : {r['rarity']:<52s}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    current_str = f"€{r['current_price_eur']}" if r['current_price_eur'] else "N/A"
    predicted_str = f"€{r['predicted_price_eur']}"
    print(f"║  Current Cardmarket price : {current_str:<33s}║")
    print(f"║  Predicted price          : {predicted_str:<33s}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a card's Cardmarket price")
    parser.add_argument("name", type=str, help="Card name (English)")
    parser.add_argument("--set", type=str, default=None, help="3-letter set code")
    args = parser.parse_args()

    predict_card(card_name=args.name, set_code=args.set)
