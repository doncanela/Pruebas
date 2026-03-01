"""
data_collector.py — Fetch MTG card data from the Scryfall API.

Scryfall provides Cardmarket EUR prices (prices.eur / prices.eur_foil)
alongside every card attribute: mana cost, oracle text, type line, rarity,
set, legalities, keywords, and more.

Two collection strategies are offered:
    1. Bulk download  — fast, grabs every card at once (~300 MB JSON).
    2. Set-by-set     — slower, but lets you target specific sets.

We keep only cards that have a non-null Cardmarket EUR price.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
from tqdm import tqdm

import config


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[dict] = None) -> dict:
    """GET with polite rate-limiting and error handling."""
    time.sleep(config.SCRYFALL_DELAY_SEC)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _cards_have_price(card: dict) -> bool:
    """Return True if the card has a Cardmarket EUR price."""
    prices = card.get("prices", {})
    return prices.get(config.PRICE_FIELD) is not None


# ─── Reprint helper ─────────────────────────────────────────────────────────

def _enrich_reprint_info(cards: list[dict]) -> list[dict]:
    """
    For every card, compute:
      - reprint_count       : how many other printings exist (by oracle_id)
      - avg_prev_price      : average Cardmarket EUR price of older printings
      - min_prev_price / max_prev_price / std_prev_price / price_spread
      - days_since_last_reprint : days between this printing and the previous one
      - oldest_printing_days    : age of the very first printing
    This uses only the cards we already downloaded (no extra API calls).
    """
    from collections import defaultdict
    from statistics import stdev

    oracle_groups: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        oid = c.get("oracle_id")
        if oid:
            oracle_groups[oid].append(c)

    # Pre-compute set sizes (number of unique oracle_ids per set)
    set_sizes: dict[str, int] = defaultdict(int)
    seen_per_set: dict[str, set] = defaultdict(set)
    for c in cards:
        s = c.get("set", "???")
        oid = c.get("oracle_id", c.get("id", ""))
        if oid not in seen_per_set[s]:
            seen_per_set[s].add(oid)
            set_sizes[s] += 1

    total = len(cards)
    for i, card in enumerate(cards):
        if i % 5000 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"  Enriching reprint info… {pct:5.1f}% ({i+1:,}/{total:,})", flush=True)
        oid = card.get("oracle_id")
        siblings = oracle_groups.get(oid, [])
        card_release = card.get("released_at", "9999-99-99")

        other_prices = []
        other_releases = []
        for sib in siblings:
            if sib.get("id") == card.get("id"):
                continue
            sib_release = sib.get("released_at", "9999-99-99")
            sib_price = sib.get("prices", {}).get(config.PRICE_FIELD)
            if sib_release <= card_release:
                other_releases.append(sib_release)
                if sib_price is not None:
                    other_prices.append(float(sib_price))

        card["reprint_count"] = max(0, len(siblings) - 1)
        card["avg_prev_price"] = (
            sum(other_prices) / len(other_prices) if other_prices else 0.0
        )
        card["min_prev_price"] = min(other_prices) if other_prices else 0.0
        card["max_prev_price"] = max(other_prices) if other_prices else 0.0
        card["std_prev_price"] = (
            stdev(other_prices) if len(other_prices) >= 2 else 0.0
        )
        card["price_spread"] = card["max_prev_price"] - card["min_prev_price"]

        # Days since last reprint
        if other_releases:
            most_recent = max(other_releases)
            try:
                from datetime import datetime as _dt
                delta = _dt.strptime(card_release, "%Y-%m-%d") - _dt.strptime(most_recent, "%Y-%m-%d")
                card["days_since_last_reprint"] = max(0, delta.days)
            except ValueError:
                card["days_since_last_reprint"] = 0
        else:
            card["days_since_last_reprint"] = 0

        # Age of the oldest printing
        if other_releases:
            oldest = min(other_releases)
            try:
                from datetime import datetime as _dt
                delta = _dt.strptime(card_release, "%Y-%m-%d") - _dt.strptime(oldest, "%Y-%m-%d")
                card["oldest_printing_days"] = max(0, delta.days)
            except ValueError:
                card["oldest_printing_days"] = 0
        else:
            card["oldest_printing_days"] = 0

        # Set size
        card["set_card_count"] = set_sizes.get(card.get("set", "???"), 0)

    return cards


# ─── Strategy 1: Bulk download ──────────────────────────────────────────────

def download_bulk(force: bool = False) -> list[dict]:
    """
    Download the Scryfall 'default_cards' bulk file (all printings).
    Filters for paper cards with a Cardmarket price.
    Results are cached in DATA_DIR.
    """
    if not force and _load_cache() is not None:
        print("  → Using cached data.")
        return _load_cache()

    print("[1/3] Fetching Scryfall bulk-data index …")
    bulk_index = _get(config.SCRYFALL_BULK_URL)
    default_entry = next(
        (e for e in bulk_index["data"] if e["type"] == "default_cards"), None
    )
    if default_entry is None:
        raise RuntimeError("Could not find 'default_cards' in Scryfall bulk data.")

    dl_url = default_entry["download_uri"]
    print(f"[2/3] Downloading bulk cards from:\n      {dl_url}")
    print("       (this is ~300 MB — may take a few minutes)")
    resp = requests.get(dl_url, timeout=600, stream=True)
    resp.raise_for_status()

    # Stream into a temp list
    raw = resp.json()  # big but manageable on modern machines
    print(f"       Downloaded {len(raw):,} raw cards.")

    print("[3/3] Filtering for paper cards with Cardmarket EUR price …")
    total_raw = len(raw)
    cards = []
    for i, c in enumerate(raw):
        if (i + 1) % 10000 == 0 or i == total_raw - 1:
            pct = (i + 1) / total_raw * 100
            print(f"  Filtering… {pct:5.1f}% ({i+1:,}/{total_raw:,}) — kept {len(cards):,} so far", flush=True)
        if (c.get("games") and "paper" in c["games"]
                and _cards_have_price(c)
                and c.get("lang", "en") == "en"):
            cards.append(c)
    print(f"       Kept {len(cards):,} cards with Cardmarket price.\n")

    cards = _enrich_reprint_info(cards)
    cards = _enrich_metagame_info(cards)
    _save_cache(cards)
    return cards


# ─── Strategy 2: Set-by-set download ────────────────────────────────────────

def download_sets(
    set_codes: Optional[list[str]] = None,
    released_after: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """
    Download cards set by set via the Scryfall search API.

    Parameters
    ----------
    set_codes : list of 3-letter set codes (e.g. ["mkm", "otj"]).
                If None, all sets released after `released_after` are used.
    released_after : ISO date string. Sets released on or after this date
                     are included. Default: 25 years ago (full reprint history).
    force : re-download even if cache exists.
    """
    if not force and _load_cache() is not None:
        print("  → Using cached data.")
        return _load_cache()

    if set_codes is None:
        if released_after is None:
            released_after = (datetime.now() - timedelta(days=365 * 25)).strftime("%Y-%m-%d")
        set_codes = _get_recent_set_codes(released_after)

    all_cards: list[dict] = []
    for code in tqdm(set_codes, desc="Downloading sets"):
        cards = _download_set(code)
        all_cards.extend(cards)

    print(f"\nTotal cards collected: {len(all_cards):,}")
    all_cards = _enrich_reprint_info(all_cards)
    all_cards = _enrich_metagame_info(all_cards)
    _save_cache(all_cards)
    return all_cards


def _get_recent_set_codes(released_after: str) -> list[str]:
    """Return set codes for sets released since `released_after`.

    Includes expansions, core sets, masters/remaster sets, commander products,
    and draft innovation — everything that carries meaningful reprint data.
    """
    data = _get(config.SCRYFALL_SETS_URL)
    valid_types = {
        "expansion", "core", "masters", "draft_innovation",
        "commander", "masterpiece", "arsenal",
    }
    codes = [
        s["code"]
        for s in data["data"]
        if s.get("set_type") in valid_types
        and s.get("released_at", "") >= released_after
    ]
    print(f"Found {len(codes)} sets (since {released_after}).")
    return codes


def _download_set(set_code: str) -> list[dict]:
    """Download all English, paper cards with Cardmarket price in a set."""
    cards: list[dict] = []
    url = config.SCRYFALL_SEARCH_URL
    params = {"q": f"set:{set_code} lang:en game:paper", "unique": "prints"}

    while url:
        try:
            data = _get(url, params=params)
        except requests.exceptions.HTTPError as e:
            # 404 means no cards match the query (empty / promo-only set)
            if e.response is not None and e.response.status_code == 404:
                return cards
            raise
        for c in data.get("data", []):
            if _cards_have_price(c):
                cards.append(c)
        url = data.get("next_page")
        params = None  # next_page already has params

    return cards


# ─── Metagame enrichment ─────────────────────────────────────────────────────

def _enrich_metagame_info(cards: list[dict]) -> list[dict]:
    """
    Merge MTGGoldfish competitive metagame data into each card dict.
    Adds per-format usage percentages and aggregate metagame signals.
    """
    try:
        from metagame_collector import fetch_metagame_data, _normalize_name
    except ImportError:
        print("  ⚠ metagame_collector not available, skipping metagame enrichment.")
        return cards

    print("\nFetching metagame data from MTGGoldfish …")
    metagame = fetch_metagame_data()

    matched = 0
    for card in cards:
        card_name = card.get("name", "")
        norm_name = _normalize_name(card_name)
        meta_info = metagame.get(norm_name, {})

        # Per-format usage: 0.0 if not in the top-50 for that format
        for fmt in ["standard", "pioneer", "modern", "legacy", "vintage"]:
            fmt_data = meta_info.get(fmt, {})
            card[f"meta_{fmt}_pct"] = fmt_data.get("pct", 0.0)
            card[f"meta_{fmt}_copies"] = fmt_data.get("copies", 0.0)

        # Aggregate signals
        card["meta_formats_played"] = len(meta_info)
        all_pcts = [v["pct"] for v in meta_info.values()] if meta_info else []
        card["meta_max_usage"] = max(all_pcts) if all_pcts else 0.0
        card["meta_avg_usage"] = (
            sum(all_pcts) / len(all_pcts) if all_pcts else 0.0
        )
        card["meta_total_usage"] = sum(all_pcts)

        if meta_info:
            matched += 1

    print(f"  Metagame: matched {matched:,} card printings to tournament data.")
    return cards


# ─── Cache management ────────────────────────────────────────────────────────

def _save_cache(cards: list[dict]) -> None:
    with open(config.RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(cards, f)
    print(f"  → Saved {len(cards):,} cards to {config.RAW_DATA_PATH}")


def _load_cache() -> Optional[list[dict]]:
    try:
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ─── CLI convenience ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MTG card data from Scryfall")
    parser.add_argument(
        "--mode", choices=["bulk", "sets"], default="sets",
        help="'bulk' for a full download (~300 MB), 'sets' for recent sets only.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument(
        "--since", type=str, default=None,
        help="ISO date (YYYY-MM-DD). Fetch sets released since this date.",
    )
    args = parser.parse_args()

    if args.mode == "bulk":
        download_bulk(force=args.force)
    else:
        download_sets(released_after=args.since, force=args.force)
