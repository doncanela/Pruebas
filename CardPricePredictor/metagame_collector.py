"""
metagame_collector.py — Fetch competitive-format metagame data from MTGGoldfish.

Two-phase scraping strategy to maximize card coverage:

Phase 1 — Format Staples (precise top-50 per format)
  Source: https://www.mtggoldfish.com/format-staples/{format}/full/all
  - Top 50 most-played cards with exact % of decks and avg copies

Phase 2 — Archetype Decklists (deep coverage, 500+ per format)
  Source: https://www.mtggoldfish.com/metagame/{format}/full
  - All meta archetypes with their meta share %
  - Each archetype's decklist: card names, copies, % of archetype decks
  - Card-level stats are weighted by archetype meta share

Formats covered: Standard, Pioneer, Modern, Legacy, Vintage

The result is a dict keyed by normalized card name:
    {
        "ragavan, nimble pilferer": {
            "modern":  {"pct": 28.0, "copies": 3.8},
            "legacy":  {"pct":  5.0, "copies": 2.1},
        },
        ...
    }

pct  = estimated % of the overall metagame that plays this card
       (for staples: direct value; for archetype cards: sum of
        archetype_meta_share × card_pct_in_archetype / 100)
copies = average copies per deck that includes the card

Cached to data/metagame_cache.json with a TTL of 7 days.
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

import config

# ─── Configuration ───────────────────────────────────────────────────────────

METAGAME_CACHE_PATH = os.path.join(config.DATA_DIR, "metagame_cache.json")
CACHE_TTL_DAYS = 7

# Formats to scrape (MTGGoldfish URL slugs)
METAGAME_FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage"]

# Maximum cards to keep per format (ranked by pct)
MAX_CARDS_PER_FORMAT = 500

# Polite delay between requests (MTGGoldfish is ad-supported, be respectful)
_REQUEST_DELAY = 1.5  # seconds

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_STAPLES_URL = "https://www.mtggoldfish.com/format-staples/{fmt}/full/all"
_METAGAME_URL = "https://www.mtggoldfish.com/metagame/{fmt}/full"


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_metagame_data(force: bool = False) -> dict[str, dict]:
    """
    Fetch metagame data from MTGGoldfish for all competitive formats.

    Returns a dict keyed by normalized card name with per-format usage data:
        {
            "card name": {
                "modern": {"pct": 28.0, "copies": 3.8},
                ...
            }
        }

    Results are cached for CACHE_TTL_DAYS days.
    """
    if not force:
        cached = _load_cache()
        if cached is not None:
            print(f"  → Using cached metagame data ({len(cached):,} cards).")
            return cached

    all_data: dict[str, dict] = {}

    for fmt in METAGAME_FORMATS:
        print(f"\n  ── {fmt.upper()} ──")
        fmt_cards = _collect_format(fmt)
        print(f"  {fmt.capitalize()} total: {len(fmt_cards):,} unique cards")
        for norm_name, info in fmt_cards.items():
            if norm_name not in all_data:
                all_data[norm_name] = {}
            all_data[norm_name][fmt] = info

    _save_cache(all_data)
    print(f"\n  Metagame data: {len(all_data):,} unique cards across "
          f"{len(METAGAME_FORMATS)} formats.")
    return all_data


def get_card_metagame(card_name: str, metagame_data: dict) -> dict:
    """
    Look up metagame data for a specific card.
    Returns a dict of {format: {pct, copies}} or empty dict.
    """
    norm = _normalize_name(card_name)
    return metagame_data.get(norm, {})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLLECTION PIPELINE (per format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _collect_format(fmt: str) -> dict[str, dict]:
    """
    Collect metagame data for a single format.

    1. Phase 1: format-staples top-50 (precise pct/copies)
    2. Phase 2: archetype decklists (deep coverage)
    3. Merge: staples data overrides archetype-derived data (more precise)
    4. Trim to MAX_CARDS_PER_FORMAT
    """
    # Phase 1: Format Staples (top 50, precise data)
    staple_cards = _scrape_format_staples(fmt)

    # Phase 2: Archetype decklists (deep coverage)
    archetype_cards = _scrape_archetype_decklists(fmt)

    # Merge: archetype data as base, staples override (more precise)
    merged: dict[str, dict] = {}
    for norm_name, info in archetype_cards.items():
        merged[norm_name] = info
    for norm_name, info in staple_cards.items():
        merged[norm_name] = info  # staples take priority

    # Sort by pct descending, trim to MAX_CARDS_PER_FORMAT
    sorted_cards = sorted(merged.items(), key=lambda x: x[1]["pct"], reverse=True)
    return dict(sorted_cards[:MAX_CARDS_PER_FORMAT])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1: FORMAT STAPLES (top 50, precise)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scrape_format_staples(fmt: str) -> dict[str, dict]:
    """
    Scrape /format-staples/{fmt}/full/all for top-50 precise data.
    Returns {normalized_name: {"pct": float, "copies": float}}.
    """
    print(f"  Phase 1: Format staples …", end=" ", flush=True)
    url = _STAPLES_URL.format(fmt=fmt)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"FAILED ({e})")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    results: dict[str, dict] = {}

    # Parse table rows
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                parsed = _parse_staples_row(cells)
                if parsed:
                    name, pct, copies = parsed
                    norm = _normalize_name(name)
                    results[norm] = {"pct": pct, "copies": copies}

    # Fallback: regex on raw HTML
    if not results:
        for name, pct, copies in _regex_fallback(resp.text):
            norm = _normalize_name(name)
            results[norm] = {"pct": pct, "copies": copies}

    time.sleep(_REQUEST_DELAY)
    print(f"{len(results)} cards")
    return results


def _parse_staples_row(cells) -> tuple[str, float, float] | None:
    """Parse a format-staples table row."""
    try:
        name_cell = None
        pct_val = None
        copies_val = None

        for cell in cells:
            link = cell.find("a")
            if link and "/price/" in str(link.get("href", "")):
                name_cell = link.get_text(strip=True)

            text = cell.get_text(strip=True)
            if "%" in text and pct_val is None:
                m = re.search(r"([\d.]+)%", text)
                if m:
                    pct_val = float(m.group(1))

            if pct_val is not None and copies_val is None and "%" not in text:
                try:
                    val = float(text)
                    if 0.5 <= val <= 4.5:
                        copies_val = val
                except ValueError:
                    pass

        if name_cell and pct_val is not None:
            return (name_cell, pct_val, copies_val or 1.0)
    except Exception:
        pass
    return None


def _regex_fallback(html: str) -> list[tuple[str, float, float]]:
    """Fallback regex extraction for format-staples data."""
    results = []
    pattern = re.compile(
        r'href="/price/[^"]*"[^>]*>([^<]+)</a>'
        r'.*?'
        r'([\d]+(?:\.\d+)?)%'
        r'.*?'
        r'<td[^>]*>\s*([\d]+(?:\.\d+)?)\s*</td>',
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        name = match.group(1).strip()
        pct = float(match.group(2))
        copies = float(match.group(3))
        if name and pct > 0:
            results.append((name, pct, copies))
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: ARCHETYPE DECKLISTS (deep coverage)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scrape_archetype_decklists(fmt: str) -> dict[str, dict]:
    """
    Scrape /metagame/{fmt}/full to get archetype tiles, then scrape each
    archetype page for its decklist with per-card copies and % of decks.

    For each card, compute a weighted metagame-level pct:
        card_meta_pct = Σ (archetype_meta_share × card_deck_pct / 100)
    and a weighted average copies.

    Returns {normalized_name: {"pct": float, "copies": float}}.
    """
    print(f"  Phase 2: Archetype decklists …")

    # Step 1: get archetypes from metagame page
    archetypes = _get_archetypes(fmt)
    if not archetypes:
        print(f"    No archetypes found for {fmt}")
        return {}
    print(f"    Found {len(archetypes)} archetypes, scraping decklists …")

    # Step 2: scrape each archetype decklist
    # Accumulate per-card: list of (archetype_meta_share, deck_pct, copies)
    card_observations: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for i, (arch_name, arch_href, meta_share) in enumerate(archetypes):
        deck_cards = _scrape_archetype_deck(arch_href)
        for card_name, deck_pct, copies in deck_cards:
            norm = _normalize_name(card_name)
            card_observations[norm].append((meta_share, deck_pct, copies))

        if (i + 1) % 10 == 0 or (i + 1) == len(archetypes):
            print(f"    [{i+1}/{len(archetypes)}] scraped, "
                  f"{len(card_observations):,} unique cards so far",
                  flush=True)
        time.sleep(_REQUEST_DELAY)

    # Step 3: aggregate per card
    results: dict[str, dict] = {}
    for norm_name, obs in card_observations.items():
        # Weighted metagame pct: sum of archetype_share × (deck_pct / 100)
        total_pct = sum(meta_share * (deck_pct / 100.0) for meta_share, deck_pct, _ in obs)
        # Weighted average copies (weighted by meta_share × deck_pct)
        weight_sum = sum(meta_share * deck_pct for meta_share, deck_pct, _ in obs)
        if weight_sum > 0:
            avg_copies = sum(
                copies * meta_share * deck_pct
                for meta_share, deck_pct, copies in obs
            ) / weight_sum
        else:
            avg_copies = sum(c for _, _, c in obs) / len(obs)

        results[norm_name] = {
            "pct": round(total_pct, 2),
            "copies": round(avg_copies, 1),
        }

    return results


def _get_archetypes(fmt: str) -> list[tuple[str, str, float]]:
    """
    Scrape /metagame/{fmt}/full for archetype tiles.
    Returns list of (archetype_name, href, meta_share_pct).
    """
    url = _METAGAME_URL.format(fmt=fmt)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"    Metagame page failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    archetypes: list[tuple[str, str, float]] = []
    seen_hrefs: set[str] = set()

    for tile in soup.select(".archetype-tile"):
        # Archetype name + link (prefer paper link)
        link_el = tile.select_one(".deck-price-paper a")
        if not link_el:
            link_el = tile.select_one("a[href*='/archetype/']")
        if not link_el:
            continue

        name = link_el.get_text(strip=True)
        href = link_el.get("href", "")
        # Remove #paper/#online fragment
        href = href.split("#")[0]

        if not href or href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        # Meta share %
        pct_el = tile.select_one(
            ".metagame-percentage .archetype-tile-statistic-value"
        )
        meta_share = 0.0
        if pct_el:
            m = re.search(r"([\d.]+)%", pct_el.get_text())
            if m:
                meta_share = float(m.group(1))

        if meta_share > 0:
            archetypes.append((name, href, meta_share))

    time.sleep(_REQUEST_DELAY)
    return archetypes


def _scrape_archetype_deck(href: str) -> list[tuple[str, float, float]]:
    """
    Scrape an individual archetype page for its decklist.
    Returns list of (card_name, pct_of_archetype_decks, avg_copies).
    """
    url = "https://www.mtggoldfish.com" + href if href.startswith("/") else href
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[tuple[str, float, float]] = []

    # Card entries: <a href="/price/...">CardName</a> inside a div whose text
    # contains "X.X in Y% of decks"
    for link in soup.find_all("a", href=re.compile(r"/price/")):
        name = link.get_text(strip=True)
        if not name:
            continue
        parent = link.parent
        if not parent:
            continue
        text = parent.get_text(strip=True)
        m = re.search(r"([\d.]+)\s*in\s*([\d.]+)%\s*of\s*decks", text)
        if m:
            copies = float(m.group(1))
            deck_pct = float(m.group(2))
            results.append((name, deck_pct, copies))

    return results


# ─── Name normalization ──────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    Normalize a card name for matching between Scryfall and MTGGoldfish.
    """
    name = " ".join(name.strip().split())
    name = name.lower()
    name = name.replace("\u2019", "'").replace("\u2018", "'")
    name = name.replace("\u2014", "-").replace("\u2013", "-")
    return name


# ─── Cache management ────────────────────────────────────────────────────────

def _save_cache(data: dict) -> None:
    """Save metagame data to JSON cache."""
    cache_obj = {
        "fetched_at": datetime.now().isoformat(),
        "data": data,
    }
    os.makedirs(os.path.dirname(METAGAME_CACHE_PATH), exist_ok=True)
    with open(METAGAME_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, indent=2, ensure_ascii=False)
    print(f"  → Cached metagame data to {METAGAME_CACHE_PATH}")


def _load_cache() -> dict | None:
    """Load metagame cache if it exists and is fresh."""
    if not os.path.exists(METAGAME_CACHE_PATH):
        return None
    try:
        with open(METAGAME_CACHE_PATH, "r", encoding="utf-8") as f:
            cache_obj = json.load(f)
        fetched = datetime.fromisoformat(cache_obj["fetched_at"])
        if datetime.now() - fetched > timedelta(days=CACHE_TTL_DAYS):
            print("  Metagame cache expired, re-fetching …")
            return None
        return cache_obj["data"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch MTG metagame data from MTGGoldfish"
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache")
    parser.add_argument("--card", type=str, help="Look up a specific card")
    args = parser.parse_args()

    data = fetch_metagame_data(force=args.force)

    if args.card:
        card_data = get_card_metagame(args.card, data)
        if card_data:
            print(f"\nMetagame data for '{args.card}':")
            for fmt, info in sorted(card_data.items()):
                print(f"  {fmt:>10s}: {info['pct']:5.1f}% meta share, "
                      f"{info['copies']:.1f} avg copies")
        else:
            print(f"\n'{args.card}' not found in metagame data.")
    else:
        for fmt in METAGAME_FORMATS:
            card_count = sum(1 for v in data.values() if fmt in v)
            print(f"  {fmt:>10s}: {card_count} cards")
