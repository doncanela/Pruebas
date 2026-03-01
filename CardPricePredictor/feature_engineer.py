"""
feature_engineer.py — Transform raw Scryfall card JSON into a flat feature matrix.

Every card is turned into a numeric row with the following feature groups:

 1. Mana cost         : cmc, color counts, number of colors, is_colorless
 2. Card types        : binary flags for each major type
 3. Rarity            : one-hot encoded
 4. Stats             : power, toughness, loyalty (0 when absent)
 5. Text complexity   : oracle_text length, word count, line count
 6. Keywords          : binary flags for tracked keyword abilities
 7. Reprint / history : reprint_count, avg/min/max previous print price
 8. Set features      : set age in days, set_size (total cards in set)
 9. Legality          : how many formats the card is legal in
10. Miscellaneous     : is_legendary, is_modal_dfc, has_adventure, etc.
11. Serialized / Special treatments
12. Interaction terms  : rarity × demand signals
13. Color identity     : bitmask, size, multicolor flags
14. Alternative costs  : substitutive-cost keywords & text patterns

Target variable: Cardmarket EUR price  (prices.eur from Scryfall).
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import config


# ─── Public API ──────────────────────────────────────────────────────────────

def build_feature_dataframe(cards: list[dict]) -> pd.DataFrame:
    """Convert a list of Scryfall card dicts into a feature DataFrame."""
    total = len(cards)
    rows = []
    for i, c in enumerate(cards):
        rows.append(_card_to_row(c))
        if (i + 1) % 5000 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"  Engineering features… {pct:5.1f}% ({i+1:,}/{total:,})", flush=True)
    df = pd.DataFrame(rows)

    # Drop rows where target is missing
    df = df.dropna(subset=["price_eur"])

    # Impute EDHREC rank: replace -1 sentinel with the median of known values
    known_mask = df["edhrec_rank"] > 0
    if known_mask.any():
        median_rank = df.loc[known_mask, "edhrec_rank"].median()
        df.loc[~known_mask, "edhrec_rank"] = median_rank
        print(f"EDHREC rank: imputed {(~known_mask).sum():,} missing → median {median_rank:.0f}")

    # Filter out extreme outliers (price > €500) to reduce noise
    n_before = len(df)
    df = df[df["price_eur"] <= 500]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} cards with price > €500 (outliers)")

    df = df.reset_index(drop=True)

    # Save for inspection
    df.to_csv(config.FEATURES_PATH, index=False)
    print(f"Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Saved to {config.FEATURES_PATH}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except the target)."""
    return [c for c in df.columns if c != "price_eur"]


# ─── Row builder ─────────────────────────────────────────────────────────────

def _card_to_row(card: dict) -> dict:
    """Convert one Scryfall card dict into a flat feature dict."""
    row: dict = {}

    # ── Target ──────────────────────────────────────────────────
    price_str = card.get("prices", {}).get(config.PRICE_FIELD)
    row["price_eur"] = float(price_str) if price_str else np.nan

    # ── 1. Mana cost ────────────────────────────────────────────
    row["cmc"] = card.get("cmc", 0.0)
    mana_cost = card.get("mana_cost", "")
    for color in config.COLORS:
        row[f"mana_{color}"] = mana_cost.count(f"{{{color}}}")
    row["num_colors"] = len(card.get("colors", []))
    row["is_colorless"] = int(row["num_colors"] == 0)

    # Generic mana (e.g. {3})
    generic_matches = re.findall(r"\{(\d+)\}", mana_cost)
    row["generic_mana"] = sum(int(x) for x in generic_matches) if generic_matches else 0

    # Hybrid / Phyrexian indicators
    row["has_hybrid_mana"] = int(bool(re.search(r"\{[WUBRG]/[WUBRG]\}", mana_cost)))
    row["has_phyrexian_mana"] = int(bool(re.search(r"\{[WUBRG]/P\}", mana_cost)))
    row["has_X_cost"] = int("{X}" in mana_cost)

    # ── 2. Card types ──────────────────────────────────────────
    type_line = card.get("type_line", "")
    for t in config.CARD_TYPES:
        row[f"is_{t.lower()}"] = int(t.lower() in type_line.lower())

    # Subtypes count (text after "—")
    if "—" in type_line:
        subtypes = type_line.split("—")[1].strip().split()
        row["subtype_count"] = len(subtypes)
    else:
        row["subtype_count"] = 0

    row["is_legendary"] = int("Legendary" in type_line)
    row["is_snow"] = int("Snow" in type_line)
    row["is_tribal"] = int("Tribal" in type_line)
    row["is_token"] = int("Token" in type_line)
    row["is_equipment"] = int("Equipment" in type_line)
    row["is_aura"] = int("Aura" in type_line)
    row["is_vehicle"] = int("Vehicle" in type_line)

    # ── 3. Rarity ──────────────────────────────────────────────
    rarity = card.get("rarity", "common")
    for r in config.RARITIES:
        row[f"rarity_{r}"] = int(rarity == r)
    row["rarity_special"] = int(rarity == "special")
    # Numeric rarity for interaction terms
    rarity_num = {"common": 1, "uncommon": 2, "rare": 3, "mythic": 4, "special": 5}
    row["rarity_numeric"] = rarity_num.get(rarity, 1)

    # ── 4. Stats ───────────────────────────────────────────────
    row["power"] = _parse_stat(card.get("power"))
    row["toughness"] = _parse_stat(card.get("toughness"))
    row["loyalty"] = _parse_stat(card.get("loyalty"))
    row["has_variable_stats"] = int(
        "*" in str(card.get("power", "")) or "*" in str(card.get("toughness", ""))
    )

    # ── 5. Text complexity ─────────────────────────────────────
    oracle = card.get("oracle_text", "")
    row["text_length"] = len(oracle)
    row["text_word_count"] = len(oracle.split())
    row["text_line_count"] = oracle.count("\n") + (1 if oracle else 0)
    row["text_has_plus_ability"] = int(bool(re.search(r"\+\d+", oracle)))  # PW +N
    row["text_has_minus_ability"] = int(bool(re.search(r"−\d+", oracle)))  # PW −N
    row["text_targets"] = oracle.lower().count("target")
    row["text_draws"] = oracle.lower().count("draw")
    row["text_destroys"] = oracle.lower().count("destroy")
    row["text_exiles"] = oracle.lower().count("exile")
    row["text_counters"] = oracle.lower().count("counter")
    row["text_tokens"] = oracle.lower().count("token")
    row["text_life"] = oracle.lower().count("life")
    row["text_sacrifice"] = oracle.lower().count("sacrifice")
    row["text_search_library"] = int("search your library" in oracle.lower())
    row["text_creates"] = oracle.lower().count("create")
    row["text_copies"] = oracle.lower().count("copy")
    row["text_each_opponent"] = oracle.lower().count("each opponent")
    row["text_each_player"] = oracle.lower().count("each player")
    row["text_whenever"] = oracle.lower().count("whenever")
    row["text_beginning_upkeep"] = int("at the beginning" in oracle.lower())
    row["text_you_may"] = oracle.lower().count("you may")
    row["text_nonland"] = oracle.lower().count("nonland")
    row["text_graveyard"] = oracle.lower().count("graveyard")
    row["text_cant"] = oracle.lower().count("can't")
    row["text_additional_cost"] = int("additional cost" in oracle.lower())
    row["text_mana_ability"] = int("add {" in oracle.lower()) or int("add one mana" in oracle.lower())
    row["text_etb"] = int("enters the battlefield" in oracle.lower()) or int("enters, " in oracle.lower())
    row["text_unique_words"] = len(set(oracle.lower().split()))
    row["text_ability_count"] = oracle.count("\n") + (1 if oracle.strip() else 0)

    # ── 6. Keywords ────────────────────────────────────────────
    keywords = set(card.get("keywords", []))
    for kw in config.TRACKED_KEYWORDS:
        row[f"kw_{_sanitize(kw)}"] = int(kw in keywords)
    row["keyword_count"] = len(keywords)

    # ── 7. Reprint / pricing history ──────────────────────────
    row["reprint_count"] = card.get("reprint_count", 0)
    row["avg_prev_price"] = card.get("avg_prev_price", 0.0)
    row["min_prev_price"] = card.get("min_prev_price", 0.0)
    row["max_prev_price"] = card.get("max_prev_price", 0.0)
    row["std_prev_price"] = card.get("std_prev_price", 0.0)
    row["price_spread"] = card.get("price_spread", 0.0)
    row["is_reprint"] = int(card.get("reprint", False))
    row["days_since_last_reprint"] = card.get("days_since_last_reprint", 0)
    row["oldest_printing_days"] = card.get("oldest_printing_days", 0)

    # Ratio: avg previous price vs reprint count (demand signal)
    rc = row["reprint_count"]
    row["avg_price_per_reprint"] = row["avg_prev_price"] / rc if rc > 0 else 0.0

    # ── 8. Set features ────────────────────────────────────────
    released = card.get("released_at", "2020-01-01")
    try:
        release_date = datetime.strptime(released, "%Y-%m-%d")
        row["days_since_release"] = (datetime.now() - release_date).days
    except ValueError:
        row["days_since_release"] = 0

    row["set_card_count"] = card.get("set_card_count", 0)

    # ── 9. Legality ────────────────────────────────────────────
    legalities = card.get("legalities", {})
    row["num_formats_legal"] = sum(
        1 for v in legalities.values() if v == "legal"
    )
    row["is_commander_legal"] = int(legalities.get("commander") == "legal")
    row["is_modern_legal"] = int(legalities.get("modern") == "legal")
    row["is_standard_legal"] = int(legalities.get("standard") == "legal")
    row["is_pioneer_legal"] = int(legalities.get("pioneer") == "legal")
    row["is_legacy_legal"] = int(legalities.get("legacy") == "legal")
    row["is_vintage_legal"] = int(legalities.get("vintage") == "legal")
    row["is_pauper_legal"] = int(legalities.get("pauper") == "legal")

    # ── 10. Misc ───────────────────────────────────────────────
    layout = card.get("layout", "normal")
    row["is_modal_dfc"] = int(layout == "modal_dfc")
    row["is_transform"] = int(layout == "transform")
    row["is_adventure"] = int(layout == "adventure")
    row["is_split"] = int(layout == "split")
    row["is_saga"] = int("Saga" in type_line)
    row["has_foil"] = int(card.get("foil", False))
    row["has_nonfoil"] = int(card.get("nonfoil", True))
    row["is_full_art"] = int(card.get("full_art", False))
    row["is_extended_art"] = int(card.get("border_color") == "borderless")
    row["is_promo"] = int(card.get("promo", False))

    # EDHREC rank (commander staple indicator)
    # Use -1 as sentinel for missing; will be imputed to median during training
    row["edhrec_rank"] = card.get("edhrec_rank", -1)
    row["has_edhrec_rank"] = int(card.get("edhrec_rank") is not None)

    # Penny Dreadful legality as a cheapness proxy
    row["is_penny_legal"] = int(legalities.get("penny") == "legal")

    # Foil price as demand signal (even though we predict non-foil)
    foil_price_str = card.get("prices", {}).get("eur_foil")
    row["foil_price"] = float(foil_price_str) if foil_price_str else 0.0
    row["has_foil_price"] = int(foil_price_str is not None)
    row["foil_multiplier"] = (
        row["foil_price"] / row["price_eur"]
        if row["price_eur"] and row["price_eur"] > 0 and row["foil_price"] > 0
        else 0.0
    )

    # Collector number as proxy for position in set (chase cards often at end)
    collector_raw = card.get("collector_number", "0")
    try:
        row["collector_number"] = int(collector_raw)
    except ValueError:
        row["collector_number"] = 0

    # ── 11. Serialized / Special treatments ─────────────────────
    # Serialized cards have a non-numeric collector number suffix or specific flag
    row["is_serialized"] = int(
        bool(card.get("finishes") and "etched" in card.get("finishes", []))
        or "serialized" in str(card.get("promo_types", [])).lower()
    )
    promo_types = card.get("promo_types", []) or []
    row["is_buy_a_box"] = int("buyabox" in promo_types)
    row["is_prerelease"] = int("prerelease" in promo_types)
    row["is_bundle"] = int("bundle" in promo_types)
    row["is_datestamped"] = int("datestamped" in promo_types)

    frame_effects = card.get("frame_effects", []) or []
    row["is_showcase"] = int(
        "showcase" in promo_types or "showcase" in frame_effects
    )
    row["is_retro_frame"] = int("retro" in frame_effects)
    row["is_extended_art_frame"] = int("extendedart" in frame_effects)
    row["is_textless"] = int(card.get("textless", False))
    row["promo_type_count"] = len(promo_types)

    # Finishes
    finishes = card.get("finishes", [])
    row["has_etched"] = int("etched" in finishes)
    row["num_finishes"] = len(finishes)

    # Frame (year-based: 1993, 1997, 2003, 2015 — older frames = collectible)
    frame = card.get("frame", "2015")
    try:
        row["frame_year"] = int(frame)
    except ValueError:
        row["frame_year"] = 2015

    # Set type (premium sets tend to have higher prices)
    set_type = card.get("set_type", "expansion")
    row["is_premium_set"] = int(set_type in config.PREMIUM_SET_TYPES)
    row["is_commander_set"] = int(set_type == "commander")
    row["is_core_set"] = int(set_type == "core")
    row["is_expansion_set"] = int(set_type == "expansion")
    row["is_masters_set"] = int(set_type in {"masters", "masterpiece"})

    # ── 12. Interaction terms (rarity × demand signals) ─────────
    rn = row["rarity_numeric"]
    row["rarity_x_edhrec"] = rn * (1.0 / max(row["edhrec_rank"], 1)) * 10000
    row["rarity_x_reprint_count"] = rn * row["reprint_count"]
    row["rarity_x_formats"] = rn * row["num_formats_legal"]
    row["rarity_x_keywords"] = rn * row["keyword_count"]
    row["rarity_x_text_length"] = rn * row["text_length"]
    row["cmc_x_rarity"] = row["cmc"] * rn
    row["avg_prev_price_x_rarity"] = row["avg_prev_price"] * rn
    row["foil_mult_x_rarity"] = row["foil_multiplier"] * rn

    # ── 13. Color identity hash ─────────────────────────────────
    color_id = card.get("color_identity", [])
    # Encode as a bitmask: W=1, U=2, B=4, R=8, G=16
    color_bits = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}
    row["color_identity_mask"] = sum(color_bits.get(c, 0) for c in color_id)
    row["color_identity_size"] = len(color_id)

    # Multi-color premium: 2+ colors often command a premium
    row["is_multicolor"] = int(len(color_id) >= 2)
    row["is_five_color"] = int(len(color_id) == 5)

    # ── 14. Alternative / substitutive costs ────────────────────
    # Keyword-based: count how many alt-cost keywords this card has
    alt_kw_count = sum(1 for kw in config.ALT_COST_KEYWORDS if kw in keywords)

    # Text-based patterns (oracle text) for unnamed alternative costs:
    oracle_lower = oracle.lower()

    # "rather than pay its/this spell's mana cost" — pitch spells
    txt_rather_than_pay = int("rather than pay" in oracle_lower)

    # "without paying its/their mana cost" — free-casting
    txt_without_paying_mana = int(
        "without paying" in oracle_lower and "mana cost" in oracle_lower
    )

    # "pay N life rather than pay" — life as cost (Snuff Out, Force of Will)
    txt_pay_life_alt = int(bool(re.search(
        r"pay .{0,15} life rather than", oracle_lower
    )))

    # "exile .* from your hand" as a substitutive cost (pitch cards)
    txt_exile_from_hand = int(
        "exile" in oracle_lower
        and "from your hand" in oracle_lower
        and "rather than" in oracle_lower
    )

    # "you may cast this/it ... for ..." — conditional alternative cost
    txt_cast_for_alt = int(bool(re.search(
        r"you may cast (this spell|it) for", oracle_lower
    )))

    # "cast .* without paying" on self — cascade / suspend payoff on other
    txt_cast_without_paying = int(bool(re.search(
        r"cast (this spell|it) without paying", oracle_lower
    )))

    # "sacrifice .* rather than pay" — offering-style
    txt_sacrifice_alt = int(bool(re.search(
        r"sacrifice .{0,30} rather than pay", oracle_lower
    )))

    # Aggregate text-based alternative cost signals
    text_alt_signals = (
        txt_rather_than_pay + txt_without_paying_mana +
        txt_pay_life_alt + txt_exile_from_hand +
        txt_cast_for_alt + txt_cast_without_paying + txt_sacrifice_alt
    )

    # Individual text flags
    row["txt_rather_than_pay"] = txt_rather_than_pay
    row["txt_without_paying_mana"] = txt_without_paying_mana
    row["txt_pay_life_alt"] = txt_pay_life_alt
    row["txt_exile_from_hand"] = txt_exile_from_hand
    row["txt_cast_for_alt"] = txt_cast_for_alt
    row["txt_cast_without_paying"] = txt_cast_without_paying
    row["txt_sacrifice_alt"] = txt_sacrifice_alt

    # Master flags
    row["has_alt_cost"] = int(alt_kw_count > 0 or text_alt_signals > 0)
    row["alt_cost_keyword_count"] = alt_kw_count
    row["alt_cost_text_count"] = text_alt_signals
    row["alt_cost_total_count"] = alt_kw_count + text_alt_signals

    # Interaction terms — alt-cost is far more impactful on expensive cards
    rn = row["rarity_numeric"]
    row["alt_cost_x_rarity"] = row["has_alt_cost"] * rn
    row["alt_cost_x_cmc"] = row["has_alt_cost"] * row["cmc"]
    row["alt_cost_x_formats"] = row["has_alt_cost"] * row["num_formats_legal"]

    return row


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_stat(value) -> float:
    """Parse power / toughness / loyalty into a float. '*' → 0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _sanitize(name: str) -> str:
    """Lowercase, replace spaces with underscores, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))


# ─── CLI convenience ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Loading cached card data …")
    with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
        cards = json.load(f)
    build_feature_dataframe(cards)
