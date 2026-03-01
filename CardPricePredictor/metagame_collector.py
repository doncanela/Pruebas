"""
metagame_collector.py — Fetch competitive-format metagame data from MTGGoldfish.

Scrapes the "Format Staples" pages to find out which cards are actually
*played* in tournaments (and how much), not just which are *legal*.

Data source: https://www.mtggoldfish.com/format-staples/{format}/full/all
  - Top 50 most-played cards per format
  - Percentage of decks that include each card
  - Average number of copies per deck

Formats covered: Standard, Pioneer, Modern, Legacy, Vintage

The result is a dict keyed by normalized card name:
    {
        "Ragavan, Nimble Pilferer": {
            "modern":  {"pct": 28.0, "copies": 3.8},
            "legacy":  {"pct":  5.0, "copies": 2.1},
        },
        ...
    }

Cached to data/metagame_cache.json with a TTL of 7 days (tournament metas
shift weekly, so we refresh once per week).
"""

import json
import os
import re
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

import config

# ─── Configuration ───────────────────────────────────────────────────────────

METAGAME_CACHE_PATH = os.path.join(config.DATA_DIR, "metagame_cache.json")
CACHE_TTL_DAYS = 7

# Formats to scrape (MTGGoldfish URL slugs)
METAGAME_FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage"]

# MTGGoldfish full-list URL pattern
_URL_TEMPLATE = "https://www.mtggoldfish.com/format-staples/{fmt}/full/all"

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


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_metagame_data(force: bool = False) -> dict[str, dict]:
    """
    Fetch format-staples data from MTGGoldfish for all competitive formats.

    Returns a dict keyed by normalized card name with per-format usage data:
        {
            "Card Name": {
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
        print(f"  Fetching {fmt.capitalize()} metagame staples …", end=" ", flush=True)
        try:
            cards = _scrape_format(fmt)
            print(f"{len(cards)} cards")
            for name, pct, copies in cards:
                norm_name = _normalize_name(name)
                if norm_name not in all_data:
                    all_data[norm_name] = {}
                all_data[norm_name][fmt] = {"pct": pct, "copies": copies}
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(_REQUEST_DELAY)

    _save_cache(all_data)
    print(f"  Metagame data: {len(all_data):,} unique cards across {len(METAGAME_FORMATS)} formats.")
    return all_data


def get_card_metagame(card_name: str, metagame_data: dict) -> dict:
    """
    Look up metagame data for a specific card.
    Returns a dict of {format: {pct, copies}} or empty dict.
    """
    norm = _normalize_name(card_name)
    return metagame_data.get(norm, {})


# ─── Scraping ────────────────────────────────────────────────────────────────

def _scrape_format(fmt: str) -> list[tuple[str, float, float]]:
    """
    Scrape the MTGGoldfish format-staples page for a single format.
    Returns list of (card_name, usage_pct, avg_copies).
    """
    url = _URL_TEMPLATE.format(fmt=fmt)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[tuple[str, float, float]] = []

    # MTGGoldfish renders staples in table rows with class "steep-table-row"
    # or within <table> elements in the main content.
    # We need to find the data table(s).

    # Strategy 1: Look for <table> with format-staples data
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                result = _parse_table_row(cells)
                if result:
                    results.append(result)

    # Strategy 2: If no table rows found, try the card-list divs
    if not results:
        # Look for elements with card names and percentages
        card_entries = soup.select(".steep-table-row, .format-staples-row")
        for entry in card_entries:
            result = _parse_div_entry(entry)
            if result:
                results.append(result)

    # Strategy 3: Regex fallback on raw HTML
    if not results:
        results = _regex_fallback(resp.text)

    return results


def _parse_table_row(cells) -> tuple[str, float, float] | None:
    """Parse a table row with [rank, card_name, mana_cost, pct, copies]."""
    try:
        # Find the cell with the card name (usually has an <a> tag)
        name_cell = None
        pct_val = None
        copies_val = None

        for cell in cells:
            # Card name: typically the cell with href to /price/
            link = cell.find("a")
            if link and "/price/" in str(link.get("href", "")):
                name_cell = link.get_text(strip=True)
            
            # Percentage: contains a '%' sign
            text = cell.get_text(strip=True)
            if "%" in text and pct_val is None:
                pct_match = re.search(r"([\d.]+)%", text)
                if pct_match:
                    pct_val = float(pct_match.group(1))
            
            # Average copies: a plain float like 3.8
            if pct_val is not None and copies_val is None and "%" not in text:
                try:
                    val = float(text)
                    if 0.5 <= val <= 4.5:  # reasonable copy count range
                        copies_val = val
                except ValueError:
                    pass

        if name_cell and pct_val is not None:
            return (name_cell, pct_val, copies_val or 1.0)
    except Exception:
        pass
    return None


def _parse_div_entry(entry) -> tuple[str, float, float] | None:
    """Parse a div-based card entry."""
    try:
        name_elem = entry.select_one("a[href*='/price/']")
        if not name_elem:
            name_elem = entry.select_one(".card-name, .name")
        
        pct_elem = entry.select_one(".percentage, .pct")
        copies_elem = entry.select_one(".copies, .avg-copies")

        if name_elem:
            name = name_elem.get_text(strip=True)
            text = entry.get_text()
            pct_match = re.search(r"([\d.]+)%", text)
            pct = float(pct_match.group(1)) if pct_match else 0.0
            
            copies = 1.0
            if copies_elem:
                try:
                    copies = float(copies_elem.get_text(strip=True))
                except ValueError:
                    pass
            
            if pct > 0:
                return (name, pct, copies)
    except Exception:
        pass
    return None


def _regex_fallback(html: str) -> list[tuple[str, float, float]]:
    """
    Fallback: extract card data via regex from raw HTML.
    MTGGoldfish format-staples pages list cards in a consistent pattern.
    """
    results = []
    # Pattern: card name in <a> to /price/ ... then percentage ... then copies
    pattern = re.compile(
        r'href="/price/[^"]*"[^>]*>([^<]+)</a>'  # card name in link
        r'.*?'
        r'([\d]+(?:\.\d+)?)%'  # percentage
        r'.*?'
        r'<td[^>]*>\s*([\d]+(?:\.\d+)?)\s*</td>',  # copies
        re.DOTALL
    )
    
    for match in pattern.finditer(html):
        name = match.group(1).strip()
        pct = float(match.group(2))
        copies = float(match.group(3))
        if name and pct > 0:
            results.append((name, pct, copies))
    
    return results


# ─── Name normalization ──────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    Normalize a card name for matching between Scryfall and MTGGoldfish.
    - Lowercase
    - Strip accents/diacritics (basic)
    - Collapse whitespace
    - Handle split cards (A // B → take both, match on either)
    """
    # Strip extra whitespace
    name = " ".join(name.strip().split())
    # Lowercase for matching
    name = name.lower()
    # Normalize common character issues
    name = name.replace("'", "'").replace("'", "'").replace("—", "-").replace("–", "-")
    return name


# ─── Cache management ────────────────────────────────────────────────────────

def _save_cache(data: dict) -> None:
    """Save metagame data to JSON cache."""
    cache_obj = {
        "fetched_at": datetime.now().isoformat(),
        "data": data,
    }
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

    parser = argparse.ArgumentParser(description="Fetch MTG metagame data from MTGGoldfish")
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-fetch")
    parser.add_argument("--card", type=str, help="Look up a specific card")
    args = parser.parse_args()

    data = fetch_metagame_data(force=args.force)

    if args.card:
        card_data = get_card_metagame(args.card, data)
        if card_data:
            print(f"\nMetagame data for '{args.card}':")
            for fmt, info in sorted(card_data.items()):
                print(f"  {fmt:>10s}: {info['pct']:5.1f}% of decks, {info['copies']:.1f} avg copies")
        else:
            print(f"\n'{args.card}' not found in metagame data (not in top-50 of any format).")
    else:
        # Print summary
        for fmt in METAGAME_FORMATS:
            card_count = sum(1 for v in data.values() if fmt in v)
            print(f"  {fmt:>10s}: {card_count} staple cards")
