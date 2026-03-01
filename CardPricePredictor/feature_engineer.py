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
 8. Set features      : set_size (total cards in set)
 9. Legality          : how many formats the card is legal in
10. Miscellaneous     : is_legendary, is_modal_dfc, has_adventure, etc.
11. Serialized / Special treatments
12. Interaction terms  : rarity × demand signals
13. Color identity     : bitmask, size, multicolor flags
14. Alternative costs  : substitutive-cost keywords & text patterns
15. Format demand      : competitive vs casual demand profile, rotation risk
16. Supply / scarcity  : set era, print-run proxy, supply drought
17. Reprint risk       : Reserved List, reprint frequency, vulnerability
18. Ban / restrict     : per-format ban flags, ban ratio
19. Seasonality        : release month/quarter (calendar features only, no leakage)
20. Metagame demand    : actual tournament usage % from MTGGoldfish per format

Target variable: Cardmarket EUR price  (prices.eur from Scryfall, snapshot at download time).

Note: This is a cross-sectional model. All features (EDHREC rank, metagame,
reprint prices, legality) are snapshot-contemporaneous: they reflect the state
at the time of data collection, not at the card's release date.  See README
for full temporal caveats.
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
    # Ensure metagame data is merged into cards (even if loaded from cache)
    cards = _ensure_metagame_enriched(cards)

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
    """Return the list of feature column names (excludes target and metadata)."""
    return [
        c for c in df.columns
        if c != "price_eur" and not c.startswith("_")
    ]


# ─── Metagame enrichment ─────────────────────────────────────────────────────

def _ensure_metagame_enriched(cards: list[dict]) -> list[dict]:
    """
    If cards don't already have metagame data, merge it from MTGGoldfish cache.
    This ensures cached raw_cards.json still gets metagame features.
    """
    # Quick check: does the first card with a name have meta fields?
    sample = next((c for c in cards if c.get("name")), None)
    if sample and "meta_formats_played" in sample:
        return cards  # already enriched

    try:
        from metagame_collector import fetch_metagame_data, _normalize_name
    except ImportError:
        print("  ⚠ metagame_collector not available, skipping metagame enrichment.")
        return cards

    print("Enriching cards with MTGGoldfish metagame data …")
    metagame = fetch_metagame_data()

    matched = 0
    for card in cards:
        card_name = card.get("name", "")
        norm_name = _normalize_name(card_name)
        meta_info = metagame.get(norm_name, {})

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

        if meta_info:
            matched += 1

    print(f"  Metagame matched: {matched:,} card printings to tournament data.")
    return cards


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
        _days_since_release = (datetime.now() - release_date).days
    except ValueError:
        _days_since_release = 0

    # Metadata-only column (underscore prefix → excluded from features)
    # Used for temporal train/test split, NOT for prediction.
    row["_days_since_release"] = _days_since_release

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

    # NOTE: foil_price / foil_multiplier REMOVED — they are current market
    # prices and constitute leakage when predicting the non-foil price.

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

    # ── 15. Format demand profile ───────────────────────────────
    # Competitive formats: Standard, Pioneer, Modern, Legacy, Vintage
    competitive_legal = sum(
        1 for fmt in config.COMPETITIVE_FORMATS
        if legalities.get(fmt) == "legal"
    )
    casual_legal = sum(
        1 for fmt in config.CASUAL_FORMATS
        if legalities.get(fmt) == "legal"
    )
    row["competitive_formats_legal"] = competitive_legal
    row["casual_formats_legal"] = casual_legal
    row["is_multi_format_staple"] = int(competitive_legal >= 3)
    row["is_competitive_only"] = int(competitive_legal > 0 and casual_legal == 0)
    row["is_casual_only"] = int(casual_legal > 0 and competitive_legal == 0)

    # Commander demand score: EDHREC rank (lower=better) + color coverage
    edhrec = row["edhrec_rank"] if row["edhrec_rank"] > 0 else 100000
    row["commander_demand_score"] = (
        row["is_commander_legal"]
        * (1.0 / max(edhrec, 1))
        * 10000
    )
    # Commander staple tiers by EDHREC rank
    row["edhrec_top_100"] = int(0 < edhrec <= 100)
    row["edhrec_top_500"] = int(0 < edhrec <= 500)
    row["edhrec_top_1000"] = int(0 < edhrec <= 1000)

    # Legacy/Vintage scarcity premium: legal only in old formats + high rarity
    # Use oldest_printing_days (design age) instead of days_since_release
    oldest_yrs = row["oldest_printing_days"] / 365.0
    row["legacy_vintage_only"] = int(
        row["is_legacy_legal"] == 1
        and row["is_modern_legal"] == 0
        and row["is_pioneer_legal"] == 0
        and row["is_standard_legal"] == 0
    )
    row["legacy_vintage_premium"] = (
        row["legacy_vintage_only"] * rn * oldest_yrs
    )

    # Standard rotation risk: legal in Standard but could rotate out
    row["standard_rotation_risk"] = int(
        row["is_standard_legal"] == 1 and row["is_pioneer_legal"] == 0
    )

    # Format demand breadth × rarity (wider play + higher rarity = higher price)
    row["format_breadth_x_rarity"] = competitive_legal * rn
    row["competitive_x_commander"] = competitive_legal * row["is_commander_legal"]
    row["demand_diversity"] = competitive_legal + casual_legal

    # ── 16. Supply / scarcity ───────────────────────────────────
    # Determine set era from release year (proxy for print run size)
    try:
        release_year = int(released[:4])
    except (ValueError, IndexError):
        release_year = 2020

    era_score = 7  # default: current era
    for _era, (start, end, score) in config.SET_ERA_BOUNDARIES.items():
        if start <= release_year < end:
            era_score = score
            break
    row["set_era"] = era_score

    # Estimated print-run proxy: era × set_type factor
    set_type = card.get("set_type", "expansion")
    type_factor = {
        "expansion": 1.0, "core": 1.0,
        "masters": 0.4, "masterpiece": 0.1,
        "commander": 0.5, "draft_innovation": 0.6,
        "from_the_vault": 0.05, "premium_deck": 0.1,
        "spellbook": 0.1, "arsenal": 0.1,
        "box": 0.3,  # Secret Lair
    }.get(set_type, 0.7)
    # Lower era_score = older = smaller print runs → lower supply score
    row["print_run_proxy"] = era_score * type_factor
    row["supply_scarcity"] = 1.0 / max(row["print_run_proxy"], 0.01)

    # Old + rare + few reprints = genuinely scarce
    # Use oldest_printing_days (design age, known at release) — NOT days_since_release
    oldest_yrs_supply = row["oldest_printing_days"] / 365.0
    row["is_old_rare"] = int(
        oldest_yrs_supply > 15 and rn >= 3 and row["reprint_count"] <= 2
    )
    row["is_old_scarce"] = int(
        oldest_yrs_supply > 10 and row["reprint_count"] <= 1
    )

    # Supply drought: how long without a reprint, weighted by total reprints
    row["reprint_drought_intensity"] = (
        row["days_since_last_reprint"] / max(row["reprint_count"] + 1, 1)
    )
    row["supply_drought_years"] = row["days_since_last_reprint"] / 365.0

    # Secret Lair and Universes Beyond detection
    set_code = card.get("set", "")
    row["is_secret_lair"] = int(
        set_type == "box" or "sld" in set_code.lower()
    )
    row["is_universes_beyond"] = int(
        set_code.lower() in config.UNIVERSES_BEYOND_MARKERS
        or "universes beyond" in card.get("set_name", "").lower()
    )

    # Scarcity interaction: old × rare × low-supply (using design age)
    row["scarcity_x_rarity"] = row["supply_scarcity"] * rn
    row["scarcity_x_demand"] = row["supply_scarcity"] * row["num_formats_legal"]
    row["card_age_x_rarity"] = oldest_yrs_supply * rn

    # ── 17. Reprint risk ────────────────────────────────────────
    # Reserved List: Scryfall provides this directly — reprint is impossible
    row["is_reserved_list"] = int(card.get("reserved", False))

    # Reprint frequency: reprints per year of card existence (design age)
    card_age_yrs = row["oldest_printing_days"] / 365.0
    row["reprint_frequency"] = (
        row["reprint_count"] / max(card_age_yrs, 0.1)
    )

    # Reprint vulnerability: high-demand, NOT reserved, been a while
    row["reprint_vulnerable"] = int(
        row["is_reserved_list"] == 0
        and rn >= 3
        and row["num_formats_legal"] >= 3
        and row["days_since_last_reprint"] > 365
    )

    # Reprint immunity: reserved OR very old + mythic + rare reprints
    row["reprint_immune"] = int(
        row["is_reserved_list"] == 1
        or (card_age_yrs > 20 and rn >= 4 and row["reprint_count"] == 0)
    )

    # Reprint ceiling pressure: non-reserved, popular, reprintable
    row["reprint_ceiling_pressure"] = (
        (1 - row["is_reserved_list"])
        * row["num_formats_legal"]
        * (1.0 / max(row["days_since_last_reprint"], 1) * 365)
    )

    # Reserved list interaction terms (using design age, not printing age)
    row["reserved_x_age"] = row["is_reserved_list"] * card_age_yrs
    row["reserved_x_rarity"] = row["is_reserved_list"] * rn
    row["reserved_x_formats"] = row["is_reserved_list"] * row["num_formats_legal"]
    row["reserved_x_scarcity"] = row["is_reserved_list"] * row["supply_scarcity"]

    # ── 18. Ban / restrict status ───────────────────────────────
    banned_formats = []
    restricted_formats = []
    for fmt, status in legalities.items():
        if status == "banned":
            banned_formats.append(fmt)
        elif status == "restricted":
            restricted_formats.append(fmt)

    row["num_formats_banned"] = len(banned_formats)
    row["num_formats_restricted"] = len(restricted_formats)
    row["is_banned_anywhere"] = int(len(banned_formats) > 0)
    row["is_restricted_anywhere"] = int(len(restricted_formats) > 0)

    # Per-format ban flags (competitive formats most impactful on price)
    row["is_banned_in_standard"] = int(legalities.get("standard") == "banned")
    row["is_banned_in_pioneer"] = int(legalities.get("pioneer") == "banned")
    row["is_banned_in_modern"] = int(legalities.get("modern") == "banned")
    row["is_banned_in_legacy"] = int(legalities.get("legacy") == "banned")
    row["is_banned_in_commander"] = int(legalities.get("commander") == "banned")
    row["is_restricted_in_vintage"] = int(legalities.get("vintage") == "restricted")

    # Ban impact score: banned in more competitive formats = bigger price hit
    # (unless it's a collector piece → handled by rarity/age interactions)
    row["ban_impact_score"] = (
        row["is_banned_in_standard"] * 1.0
        + row["is_banned_in_pioneer"] * 1.5
        + row["is_banned_in_modern"] * 2.0
        + row["is_banned_in_legacy"] * 2.5
        + row["is_banned_in_commander"] * 3.0
    )

    # Ban ratio: fraction of formats where card is banned
    total_formats = len(legalities) if legalities else 1
    row["ban_ratio"] = len(banned_formats) / total_formats

    # Restricted = powerful but not banned → often high value
    row["restricted_power_signal"] = (
        row["is_restricted_anywhere"] * rn
    )

    # ── 19. Seasonality (release-date-knowable features only) ──
    # NOTE: days/months/years_since_release, hype_decay, recency buckets,
    # rotation proximity — all REMOVED as leakage.  At t+60 these would be
    # constants; keeping them variable in training conflates card age with
    # card quality.  We keep ONLY calendar features that are known at release.
    try:
        release_dt = datetime.strptime(released, "%Y-%m-%d")
        row["release_month"] = release_dt.month
        row["release_quarter"] = (release_dt.month - 1) // 3 + 1
        row["release_day_of_year"] = release_dt.timetuple().tm_yday
    except ValueError:
        row["release_month"] = 1
        row["release_quarter"] = 1
        row["release_day_of_year"] = 1

    # Fall sets (Q4) tend to have holiday-driven sales
    row["is_fall_set"] = int(row["release_quarter"] == 4)
    # Spring sets (Q1-Q2) with Pro Tour season
    row["is_competitive_season"] = int(row["release_quarter"] in (1, 2))

    # ── 20. Metagame demand (MTGGoldfish tournament data) ───────
    # Per-format usage percentage (0–100; 0 = not in top staples)
    for fmt in ["standard", "pioneer", "modern", "legacy", "vintage"]:
        row[f"meta_{fmt}_pct"] = card.get(f"meta_{fmt}_pct", 0.0)
        row[f"meta_{fmt}_copies"] = card.get(f"meta_{fmt}_copies", 0.0)
        # Missingness flag: 1 = card is legal but NOT in metagame data
        # (could be a playable card outside top-N staples, vs 0 = not legal)
        is_legal_in_fmt = int(legalities.get(fmt) == "legal")
        has_meta = int(card.get(f"meta_{fmt}_pct", 0.0) > 0)
        row[f"meta_{fmt}_unknown"] = int(is_legal_in_fmt == 1 and has_meta == 0)

    # Aggregate metagame signals
    row["meta_formats_played"] = card.get("meta_formats_played", 0)
    row["meta_max_usage"] = card.get("meta_max_usage", 0.0)
    row["meta_avg_usage"] = card.get("meta_avg_usage", 0.0)
    row["meta_total_usage"] = card.get("meta_total_usage", 0.0)

    # Metagame tier flags
    row["is_meta_staple"] = int(row["meta_max_usage"] >= 20.0)
    row["is_meta_allstar"] = int(row["meta_max_usage"] >= 40.0)
    row["is_multi_format_meta"] = int(row["meta_formats_played"] >= 2)
    row["is_cross_format_staple"] = int(
        row["meta_formats_played"] >= 3 and row["meta_avg_usage"] >= 15.0
    )

    # Competitive demand score: weighted sum across formats
    # (Modern/Pioneer have more players → higher weight than Vintage)
    row["competitive_demand_score"] = (
        row["meta_standard_pct"] * 1.0
        + row["meta_pioneer_pct"] * 1.2
        + row["meta_modern_pct"] * 1.5
        + row["meta_legacy_pct"] * 0.8
        + row["meta_vintage_pct"] * 0.5
    )

    # Metagame × rarity interaction (meta-played rares/mythics = premium)
    row["meta_x_rarity"] = row["meta_max_usage"] * rn
    row["meta_x_scarcity"] = row["meta_max_usage"] * row.get("supply_scarcity", 1.0)
    row["meta_x_commander"] = (
        row["meta_formats_played"] * row["is_commander_legal"]
    )

    # Specific format demand profiles
    row["modern_pioneer_demand"] = (
        row["meta_modern_pct"] + row["meta_pioneer_pct"]
    )
    row["eternal_format_demand"] = (
        row["meta_legacy_pct"] + row["meta_vintage_pct"]
    )

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
