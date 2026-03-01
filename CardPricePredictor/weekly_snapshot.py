"""
weekly_snapshot.py — Automated weekly price snapshot.

This script is meant to be called by Windows Task Scheduler (or cron).
It:
  1. Downloads fresh bulk card data from Scryfall
  2. Takes a price snapshot (local CSV + Neon DB)
  3. Syncs card metadata to Neon DB
  4. Logs everything to a rotating log file

Exit codes:
  0 = success
  1 = partial failure (snapshot taken but DB sync failed)
  2 = fatal error
"""

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


def run():
    exit_code = 0
    start = datetime.now()
    logger.info("=" * 50)
    logger.info("Weekly snapshot started")

    # 1. Download fresh bulk data
    logger.info("Step 1/3: Downloading bulk card data from Scryfall…")
    try:
        from data_collector import download_bulk
        cards = download_bulk(force=True)
        logger.info(f"  Downloaded {len(cards):,} cards.")
    except Exception as e:
        logger.error(f"  FATAL: Bulk download failed: {e}")
        return 2

    # 2. Take price snapshot (local CSV + Neon DB)
    logger.info("Step 2/3: Taking price snapshot…")
    try:
        from price_history import take_snapshot
        n = take_snapshot(cards)
        logger.info(f"  Snapshot saved: {n:,} card prices.")
    except Exception as e:
        logger.error(f"  Snapshot failed: {e}")
        exit_code = 1

    # 3. Sync card metadata to Neon DB
    logger.info("Step 3/3: Syncing card metadata to Neon DB…")
    try:
        from db import init_db, upsert_cards_batch
        init_db()
        n = upsert_cards_batch(cards)
        logger.info(f"  Synced {n:,} cards to Neon DB.")
    except Exception as e:
        logger.error(f"  DB card sync failed: {e}")
        if exit_code == 0:
            exit_code = 1

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Weekly snapshot finished in {elapsed:.0f}s (exit code {exit_code})")
    logger.info("=" * 50)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
