"""
weekly_snapshot.py — Automated weekly pipeline.

This script is meant to be called by Windows Task Scheduler, GitHub Actions,
or cron.  It:
  1. Downloads fresh bulk card data from Scryfall  (prices + EDHREC ranks)
  2. Refreshes MTGGoldfish metagame staples        (5 competitive formats)
  3. Takes a price snapshot                        (local CSV + Neon DB)
  4. Syncs card metadata to Neon DB
  5. Syncs EDHREC ranks to Neon DB                 (weekly rank history)
  6. Syncs metagame staples to Neon DB             (weekly metagame history)
  7. Retrains the model on the newest data          (optional, --retrain flag)

Exit codes:
  0 = success
  1 = partial failure (non-critical step failed)
  2 = fatal error (bulk download failed)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(config.BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "weekly_snapshot.log")

logger = logging.getLogger("weekly_snapshot")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(handler)

# Also print to console (useful when running manually)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console)


def run(retrain: bool = False):
    exit_code = 0
    total_steps = 7 if retrain else 6
    start = datetime.now()
    logger.info("=" * 50)
    logger.info(f"Weekly snapshot started (retrain={'yes' if retrain else 'no'})")
    meta = None  # will hold metagame dict if step 2 succeeds

    # ── 1. Download fresh bulk data (Scryfall → Cardmarket EUR prices + EDHREC)
    step = 1
    logger.info(f"Step {step}/{total_steps}: Downloading bulk card data from Scryfall…")
    try:
        from data_collector import download_bulk
        cards = download_bulk(force=True)
        logger.info(f"  Downloaded {len(cards):,} cards.")
    except Exception as e:
        logger.error(f"  FATAL: Bulk download failed: {e}")
        return 2

    # ── 2. Refresh MTGGoldfish metagame data
    step = 2
    logger.info(f"Step {step}/{total_steps}: Refreshing MTGGoldfish metagame staples…")
    try:
        from metagame_collector import fetch_metagame_data
        meta = fetch_metagame_data(force=True)
        logger.info(f"  Metagame refreshed: {len(meta):,} unique cards across 5 formats.")
    except Exception as e:
        logger.warning(f"  Metagame refresh failed (non-fatal): {e}")
        if exit_code == 0:
            exit_code = 1

    # ── 3. Take price snapshot (local CSV + Neon DB)
    step = 3
    logger.info(f"Step {step}/{total_steps}: Taking price snapshot…")
    try:
        from price_history import take_snapshot
        n = take_snapshot(cards)
        logger.info(f"  Snapshot saved: {n:,} card prices.")
    except Exception as e:
        logger.error(f"  Snapshot failed: {e}")
        if exit_code == 0:
            exit_code = 1

    # ── 4. Sync card metadata to Neon DB
    step = 4
    logger.info(f"Step {step}/{total_steps}: Syncing card metadata to Neon DB…")
    try:
        from db import init_db, upsert_cards_batch, upsert_edhrec_batch, upsert_metagame_batch
        init_db()
        n = upsert_cards_batch(cards)
        logger.info(f"  Synced {n:,} cards to Neon DB.")
    except Exception as e:
        logger.error(f"  DB card sync failed: {e}")
        if exit_code == 0:
            exit_code = 1

    # ── 5. Sync EDHREC ranks to Neon DB
    step = 5
    logger.info(f"Step {step}/{total_steps}: Syncing EDHREC ranks to Neon DB…")
    try:
        n = upsert_edhrec_batch(cards)
        logger.info(f"  EDHREC ranks synced: {n:,} cards with ranks.")
    except Exception as e:
        logger.error(f"  EDHREC sync failed: {e}")
        if exit_code == 0:
            exit_code = 1

    # ── 6. Sync metagame staples to Neon DB
    step = 6
    logger.info(f"Step {step}/{total_steps}: Syncing metagame staples to Neon DB…")
    if meta:
        try:
            n = upsert_metagame_batch(meta)
            logger.info(f"  Metagame synced: {n:,} format-card rows.")
        except Exception as e:
            logger.error(f"  Metagame DB sync failed: {e}")
            if exit_code == 0:
                exit_code = 1
    else:
        logger.warning("  Skipped — no metagame data available (step 2 failed).")

    # ── 7. Retrain model (optional)
    if retrain:
        step = 7
        logger.info(f"Step {step}/{total_steps}: Retraining model on fresh data…")
        try:
            from model import train as train_model
            metrics = train_model(cards=cards)
            logger.info(f"  Model retrained — R²={metrics.get('r2', '?'):.3f}, "
                        f"MAE=€{metrics.get('mae', '?'):.2f}")
        except Exception as e:
            logger.error(f"  Model retraining failed: {e}")
            if exit_code == 0:
                exit_code = 1

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Weekly snapshot finished in {elapsed:.0f}s (exit code {exit_code})")
    logger.info("=" * 50)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly MTG data snapshot & optional retrain")
    parser.add_argument("--retrain", action="store_true",
                        help="Also retrain the model after refreshing data")
    args = parser.parse_args()
    sys.exit(run(retrain=args.retrain))
