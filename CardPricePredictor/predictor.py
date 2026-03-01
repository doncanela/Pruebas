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
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests

import config
from feature_engineer import _card_to_row
from model import load_model


# ─── Public API ──────────────────────────────────────────────────────────────

def predict_card(
    card_name: Optional[str] = None,
    card_dict: Optional[dict] = None,
    set_code: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Predict the Cardmarket EUR price ~2 months after release.

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

    # 3. Load model
    model, scaler, feature_cols = load_model()

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
    print(f"║  Predicted price (≈2 mo.) : {predicted_str:<33s}║")
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
