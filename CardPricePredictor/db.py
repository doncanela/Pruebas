"""
db.py — Neon PostgreSQL database layer for the MTG Card Price Predictor.

Stores:
  • predictions       — every prediction with timestamp
  • price_snapshots   — periodic snapshots of all card prices
  • cards             — card metadata (name, set, rarity, mana cost, type, etc.)

Connection is via the DATABASE_URL in config.py (Neon serverless PostgreSQL).
"""

import os
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional

import psycopg

import config

# ─── Connection ──────────────────────────────────────────────────────────────

def get_conn():
    """Return a new psycopg connection to Neon."""
    return psycopg.connect(config.DATABASE_URL, autocommit=True)


@contextmanager
def cursor():
    """Context-managed cursor that auto-closes the connection."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
    finally:
        conn.close()


# ─── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id              SERIAL PRIMARY KEY,
    scryfall_id     TEXT UNIQUE,
    name            TEXT NOT NULL,
    set_code        TEXT,
    set_name        TEXT,
    rarity          TEXT,
    mana_cost       TEXT,
    cmc             REAL,
    type_line       TEXT,
    oracle_text     TEXT,
    collector_number TEXT,
    released_at     DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cards_name ON cards (name);
CREATE INDEX IF NOT EXISTS idx_cards_set  ON cards (set_code);

CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    card_name           TEXT NOT NULL,
    set_code            TEXT,
    set_name            TEXT,
    rarity              TEXT,
    mana_cost           TEXT,
    predicted_price_eur REAL,
    current_price_eur   REAL,
    model_version       TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_card   ON predictions (card_name);
CREATE INDEX IF NOT EXISTS idx_pred_ts     ON predictions (timestamp);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id               SERIAL PRIMARY KEY,
    snapshot_date    DATE NOT NULL,
    card_name        TEXT NOT NULL,
    set_code         TEXT,
    set_name         TEXT,
    rarity           TEXT,
    price_eur        REAL,
    price_eur_foil   REAL,
    collector_number TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snap_card ON price_snapshots (card_name);
CREATE INDEX IF NOT EXISTS idx_snap_date ON price_snapshots (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snap_card_date ON price_snapshots (card_name, snapshot_date);
"""


def init_db() -> None:
    """Create tables if they don't exist."""
    with cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    print("  ✓ Database schema initialized.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insert_prediction(result: dict, model_version: str = "v1") -> None:
    """Insert a prediction into the database."""
    with cursor() as cur:
        cur.execute(
            """INSERT INTO predictions
               (card_name, set_code, set_name, rarity, mana_cost,
                predicted_price_eur, current_price_eur, model_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                result.get("card_name"),
                result.get("set_code"),
                result.get("set"),
                result.get("rarity"),
                result.get("mana_cost"),
                result.get("predicted_price_eur"),
                result.get("current_price_eur"),
                model_version,
            ),
        )


def get_predictions(card_name: Optional[str] = None,
                    limit: int = 200) -> list[dict]:
    """Retrieve predictions, optionally filtered by card name."""
    with cursor() as cur:
        if card_name:
            cur.execute(
                """SELECT timestamp, card_name, set_code, set_name, rarity,
                          mana_cost, predicted_price_eur, current_price_eur,
                          model_version
                   FROM predictions
                   WHERE LOWER(card_name) = LOWER(%s)
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (card_name, limit),
            )
        else:
            cur.execute(
                """SELECT timestamp, card_name, set_code, set_name, rarity,
                          mana_cost, predicted_price_eur, current_price_eur,
                          model_version
                   FROM predictions
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (limit,),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRICE SNAPSHOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insert_snapshot_batch(rows: list[dict], snapshot_date: Optional[str] = None) -> int:
    """
    Bulk-insert price snapshot rows.

    Parameters
    ----------
    rows : list of dicts with keys: card_name, set_code, set_name, rarity,
           price_eur, price_eur_foil, collector_number
    snapshot_date : ISO date string.  Default = today.

    Returns
    -------
    Number of rows inserted.
    """
    if not rows:
        return 0
    snap_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")

    BATCH_SIZE = 500
    insert_sql = """
        INSERT INTO price_snapshots
           (snapshot_date, card_name, set_code, set_name, rarity,
            price_eur, price_eur_foil, collector_number)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Prepare tuples
    prepared = []
    for r in rows:
        foil = r.get("price_eur_foil")
        prepared.append((
            snap_date,
            r.get("card_name", ""),
            r.get("set_code", ""),
            r.get("set_name", ""),
            r.get("rarity", ""),
            r.get("price_eur"),
            foil if foil else None,
            r.get("collector_number", ""),
        ))

    total = len(prepared)
    n = 0
    conn = psycopg.connect(config.DATABASE_URL)  # autocommit OFF for batching
    try:
        with conn.cursor() as cur:
            for start in range(0, total, BATCH_SIZE):
                batch = prepared[start : start + BATCH_SIZE]
                cur.executemany(insert_sql, batch)
                conn.commit()
                n += len(batch)
                if n % 5000 < BATCH_SIZE or n == total:
                    pct = n / total * 100
                    print(f"  DB snapshot… {pct:5.1f}% ({n:,}/{total:,})", flush=True)
    finally:
        conn.close()
    return n


def get_snapshots(card_name: Optional[str] = None,
                  set_code: Optional[str] = None,
                  limit: int = 500) -> list[dict]:
    """Retrieve price snapshot rows."""
    with cursor() as cur:
        clauses = []
        params: list = []
        if card_name:
            clauses.append("LOWER(card_name) = LOWER(%s)")
            params.append(card_name)
        if set_code:
            clauses.append("LOWER(set_code) = LOWER(%s)")
            params.append(set_code)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(
            f"""SELECT snapshot_date, card_name, set_code, set_name, rarity,
                       price_eur, price_eur_foil, collector_number
                FROM price_snapshots
                {where}
                ORDER BY snapshot_date DESC, card_name
                LIMIT %s""",
            params + [limit],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_snapshot_dates() -> list[str]:
    """Return sorted unique snapshot dates."""
    with cursor() as cur:
        cur.execute(
            "SELECT DISTINCT snapshot_date FROM price_snapshots ORDER BY snapshot_date"
        )
        return [str(row[0]) for row in cur.fetchall()]


def get_price_timeline(card_name: str,
                       set_code: Optional[str] = None) -> list[dict]:
    """
    Get the price over time for a single card — one row per snapshot date.
    Perfect for plotting price evolution.
    """
    with cursor() as cur:
        if set_code:
            cur.execute(
                """SELECT snapshot_date, price_eur, price_eur_foil
                   FROM price_snapshots
                   WHERE LOWER(card_name) = LOWER(%s)
                     AND LOWER(set_code) = LOWER(%s)
                   ORDER BY snapshot_date""",
                (card_name, set_code),
            )
        else:
            cur.execute(
                """SELECT snapshot_date, price_eur, price_eur_foil
                   FROM price_snapshots
                   WHERE LOWER(card_name) = LOWER(%s)
                   ORDER BY snapshot_date""",
                (card_name,),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CARDS METADATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def upsert_cards_batch(cards: list[dict]) -> int:
    """
    Bulk upsert card metadata from Scryfall dicts.
    Uses executemany with batches for speed over network.
    Returns number of cards processed.
    """
    if not cards:
        return 0

    BATCH_SIZE = 500
    upsert_sql = """
        INSERT INTO cards
           (scryfall_id, name, set_code, set_name, rarity,
            mana_cost, cmc, type_line, oracle_text,
            collector_number, released_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scryfall_id) DO UPDATE SET
            name = EXCLUDED.name,
            set_name = EXCLUDED.set_name,
            rarity = EXCLUDED.rarity,
            mana_cost = EXCLUDED.mana_cost,
            cmc = EXCLUDED.cmc,
            type_line = EXCLUDED.type_line,
            oracle_text = EXCLUDED.oracle_text,
            collector_number = EXCLUDED.collector_number,
            released_at = EXCLUDED.released_at
    """

    # Prepare all rows first
    rows = []
    for card in cards:
        scryfall_id = card.get("id")
        if not scryfall_id:
            continue
        rows.append((
            scryfall_id,
            card.get("name", ""),
            card.get("set", ""),
            card.get("set_name", ""),
            card.get("rarity", ""),
            card.get("mana_cost", ""),
            card.get("cmc", 0),
            card.get("type_line", ""),
            card.get("oracle_text", ""),
            card.get("collector_number", ""),
            card.get("released_at"),
        ))

    total = len(rows)
    n = 0
    conn = psycopg.connect(config.DATABASE_URL)  # autocommit OFF for batching
    try:
        with conn.cursor() as cur:
            for start in range(0, total, BATCH_SIZE):
                batch = rows[start : start + BATCH_SIZE]
                cur.executemany(upsert_sql, batch)
                conn.commit()
                n += len(batch)
                pct = n / total * 100
                print(f"  Syncing cards to DB… {pct:5.1f}% ({n:,}/{total:,})", flush=True)
    finally:
        conn.close()
    return n


def get_card_count() -> int:
    """Return total number of cards in the DB."""
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM cards")
        return cur.fetchone()[0]


def get_prediction_count() -> int:
    """Return total number of predictions logged."""
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM predictions")
        return cur.fetchone()[0]


def get_snapshot_count() -> int:
    """Return total number of snapshot rows."""
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM price_snapshots")
        return cur.fetchone()[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATS / DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_db_stats() -> None:
    """Print a summary of what's in the database."""
    print(f"\n  Neon Database: {config.DATABASE_URL[:40]}…")
    print(f"  Cards in DB       : {get_card_count():,}")
    print(f"  Predictions logged: {get_prediction_count():,}")
    print(f"  Snapshot rows     : {get_snapshot_count():,}")
    dates = get_snapshot_dates()
    if dates:
        print(f"  Snapshot dates    : {len(dates)} ({dates[0]} → {dates[-1]})")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Database utilities")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("init", help="Create tables")
    sub.add_parser("stats", help="Show DB statistics")
    args = p.parse_args()
    if args.cmd == "init":
        init_db()
    elif args.cmd == "stats":
        print_db_stats()
    else:
        p.print_help()
