"""
price_history.py — Prediction logging and periodic price-snapshot tracking.

Two independent stores:

1. **Prediction log** (``predictions.jsonl``):
   Every call to ``predict_card()`` is appended here with timestamp,
   predicted price, and actual Cardmarket price *at the time of prediction*.
   You can later compare to see how accurate predictions were.

2. **Price snapshots** (``price_snapshots.csv``):
   A periodic dump of current Cardmarket prices for *all* cards in the
   local database.  Run ``python main.py snapshot`` weekly/monthly to
   build a multi-date price history you can chart.

Both stores are append-only and stored in ``data/``.
"""

import csv
import json
import os
import time
from datetime import datetime
from typing import Optional

import requests

import config

# ─── Paths ───────────────────────────────────────────────────────────────────
PREDICTION_LOG = os.path.join(config.DATA_DIR, "predictions.jsonl")
SNAPSHOT_CSV   = os.path.join(config.DATA_DIR, "price_snapshots.csv")

SNAPSHOT_FIELDS = [
    "date", "card_name", "set_code", "set_name", "rarity",
    "price_eur", "price_eur_foil", "collector_number",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  PREDICTION LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_prediction(result: dict) -> None:
    """Append a prediction result to the JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "card_name": result.get("card_name"),
        "set_code": result.get("set_code"),
        "set": result.get("set"),
        "rarity": result.get("rarity"),
        "mana_cost": result.get("mana_cost"),
        "predicted_price_eur": result.get("predicted_price_eur"),
        "current_price_eur": result.get("current_price_eur"),
    }
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_predictions(card_name: Optional[str] = None) -> list[dict]:
    """Load all logged predictions, optionally filtered by card name."""
    if not os.path.exists(PREDICTION_LOG):
        return []
    entries = []
    with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if card_name and entry.get("card_name", "").lower() != card_name.lower():
                continue
            entries.append(entry)
    return entries


def check_prediction_accuracy() -> list[dict]:
    """
    For every logged prediction, fetch the card's CURRENT price from
    Scryfall and compare it to the predicted price.
    Returns a list of dicts with original + accuracy info.
    """
    predictions = load_predictions()
    if not predictions:
        return []

    # De-duplicate to avoid hammering the API for the same card
    seen: dict[str, float | None] = {}
    results = []

    for pred in predictions:
        key = f"{pred['card_name']}||{pred.get('set_code', '')}"
        if key not in seen:
            # Fetch current price
            time.sleep(config.SCRYFALL_DELAY_SEC)
            try:
                params = {"fuzzy": pred["card_name"]}
                if pred.get("set_code") and pred["set_code"] != "???":
                    params["set"] = pred["set_code"]
                resp = requests.get(config.SCRYFALL_CARD_URL,
                                    params=params, timeout=15)
                if resp.status_code == 200:
                    p = resp.json().get("prices", {}).get(config.PRICE_FIELD)
                    seen[key] = float(p) if p else None
                else:
                    seen[key] = None
            except Exception:
                seen[key] = None

        current_now = seen[key]
        entry = {**pred, "actual_price_now": current_now}
        if current_now is not None and pred.get("predicted_price_eur") is not None:
            entry["error"] = round(abs(pred["predicted_price_eur"] - current_now), 2)
        else:
            entry["error"] = None
        results.append(entry)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  PRICE SNAPSHOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def take_snapshot(cards: Optional[list[dict]] = None) -> int:
    """
    Append today's prices for all cards to the snapshot CSV
    AND to the Neon PostgreSQL database.

    Parameters
    ----------
    cards : list of Scryfall card dicts.  If None, loads from raw_cards.json.

    Returns
    -------
    Number of rows written.
    """
    if cards is None:
        if not os.path.exists(config.RAW_DATA_PATH):
            raise FileNotFoundError(
                f"No card data at {config.RAW_DATA_PATH}. Run 'collect' first."
            )
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")
    write_header = not os.path.exists(SNAPSHOT_CSV)
    n = 0
    total = len(cards)

    with open(SNAPSHOT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS)
        if write_header:
            writer.writeheader()

        for i, card in enumerate(cards):
            price_str = card.get("prices", {}).get(config.PRICE_FIELD)
            if price_str is None:
                continue
            foil_str = card.get("prices", {}).get("eur_foil")
            writer.writerow({
                "date": today,
                "card_name": card.get("name", ""),
                "set_code": card.get("set", ""),
                "set_name": card.get("set_name", ""),
                "rarity": card.get("rarity", ""),
                "price_eur": price_str,
                "price_eur_foil": foil_str or "",
                "collector_number": card.get("collector_number", ""),
            })
            n += 1

            if (i + 1) % 10000 == 0 or i == total - 1:
                pct = (i + 1) / total * 100
                print(f"  Snapshotting… {pct:5.1f}% ({i+1:,}/{total:,})", flush=True)

    # ── Also write to Neon DB ──
    try:
        from db import insert_snapshot_batch
        db_rows = []
        for card in cards:
            price_str = card.get("prices", {}).get(config.PRICE_FIELD)
            if price_str is None:
                continue
            foil_str = card.get("prices", {}).get("eur_foil")
            db_rows.append({
                "card_name": card.get("name", ""),
                "set_code": card.get("set", ""),
                "set_name": card.get("set_name", ""),
                "rarity": card.get("rarity", ""),
                "price_eur": float(price_str),
                "price_eur_foil": float(foil_str) if foil_str else None,
                "collector_number": card.get("collector_number", ""),
            })
        inserted = insert_snapshot_batch(db_rows)
        print(f"  ✓ Also saved {inserted:,} rows to Neon DB.", flush=True)
    except Exception as e:
        print(f"  ⚠ DB snapshot failed (local CSV still saved): {e}", flush=True)

    return n


def load_snapshots(card_name: Optional[str] = None,
                   set_code: Optional[str] = None) -> list[dict]:
    """Load snapshot rows, optionally filtered by card_name and/or set_code."""
    if not os.path.exists(SNAPSHOT_CSV):
        return []
    rows = []
    with open(SNAPSHOT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if card_name and row["card_name"].lower() != card_name.lower():
                continue
            if set_code and row["set_code"].lower() != set_code.lower():
                continue
            rows.append(row)
    return rows


def get_snapshot_dates() -> list[str]:
    """Return a sorted list of unique snapshot dates on file."""
    if not os.path.exists(SNAPSHOT_CSV):
        return []
    dates = set()
    with open(SNAPSHOT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.add(row["date"])
    return sorted(dates)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  DISPLAY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_card_history(card_name: str, set_code: Optional[str] = None) -> None:
    """Pretty-print both prediction log and price snapshots for one card."""

    # ── Prediction log ──
    preds = load_predictions(card_name)
    if preds:
        print(f"\n{'═' * 64}")
        print(f"  Prediction history for: {card_name}")
        print(f"{'═' * 64}")
        print(f"  {'Date':<22s} {'Predicted':>10s} {'At-time Cardmkt':>16s}")
        print(f"  {'─' * 52}")
        for p in preds:
            ts = p.get("timestamp", "?")[:19]
            pred = f"€{p['predicted_price_eur']:.2f}" if p.get("predicted_price_eur") is not None else "N/A"
            curr = f"€{p['current_price_eur']:.2f}" if p.get("current_price_eur") is not None else "N/A"
            print(f"  {ts:<22s} {pred:>10s} {curr:>16s}")
    else:
        print(f"\n  No predictions logged for '{card_name}'.")

    # ── Price snapshots ──
    snaps = load_snapshots(card_name, set_code)
    if snaps:
        # Group by date (could have multiple sets)
        print(f"\n  Price snapshot history:")
        print(f"  {'Date':<12s} {'Set':<8s} {'Rarity':<10s} {'Price':>8s} {'Foil':>8s}")
        print(f"  {'─' * 50}")
        for s in snaps:
            price = f"€{float(s['price_eur']):.2f}" if s.get("price_eur") else "N/A"
            foil = f"€{float(s['price_eur_foil']):.2f}" if s.get("price_eur_foil") else "—"
            print(f"  {s['date']:<12s} {s['set_code']:<8s} {s['rarity']:<10s} {price:>8s} {foil:>8s}")
    else:
        print(f"  No price snapshots yet. Run 'python main.py snapshot' to start tracking.")

    print()


def print_accuracy_report() -> None:
    """Fetch current prices for all past predictions and show accuracy."""
    print("\n  Fetching current prices for past predictions…\n")
    results = check_prediction_accuracy()

    if not results:
        print("  No predictions logged yet.")
        return

    print(f"  {'Card':<35s} {'Predicted':>10s} {'At-time':>10s} {'Now':>10s} {'Error':>8s}")
    print(f"  {'─' * 77}")
    errors = []
    for r in results:
        pred = f"€{r['predicted_price_eur']:.2f}" if r.get("predicted_price_eur") is not None else "N/A"
        at_time = f"€{r['current_price_eur']:.2f}" if r.get("current_price_eur") is not None else "N/A"
        now = f"€{r['actual_price_now']:.2f}" if r.get("actual_price_now") is not None else "N/A"
        err = f"€{r['error']:.2f}" if r.get("error") is not None else "—"
        name = r.get("card_name", "?")[:34]
        print(f"  {name:<35s} {pred:>10s} {at_time:>10s} {now:>10s} {err:>8s}")
        if r.get("error") is not None:
            errors.append(r["error"])

    if errors:
        avg_err = sum(errors) / len(errors)
        print(f"\n  Average error: €{avg_err:.2f}  ({len(errors)} predictions)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Price history tools")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("snapshot", help="Take a price snapshot for all cards")
    p_hist = sub.add_parser("history", help="Show history for a card")
    p_hist.add_argument("name", help="Card name")
    p_hist.add_argument("--set", default=None, help="Set code filter")
    sub.add_parser("accuracy", help="Check accuracy of past predictions")

    args = p.parse_args()
    if args.cmd == "snapshot":
        n = take_snapshot()
        print(f"  ✓ Snapshot saved: {n:,} cards → {SNAPSHOT_CSV}")
    elif args.cmd == "history":
        print_card_history(args.name, getattr(args, "set", None))
    elif args.cmd == "accuracy":
        print_accuracy_report()
    else:
        p.print_help()
