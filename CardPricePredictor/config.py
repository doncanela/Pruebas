"""
Configuration constants for the MTG Card Price Predictor.
Scryfall API is used as the data source — it includes Cardmarket EUR prices
and comprehensive card metadata (mana cost, oracle text, reprints, etc.).
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_cards.json")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
PREDICTION_LOG_PATH = os.path.join(DATA_DIR, "predictions.jsonl")
SNAPSHOT_CSV_PATH = os.path.join(DATA_DIR, "price_snapshots.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

# ─── Scryfall API ────────────────────────────────────────────────────────────
SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data"
SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
SCRYFALL_CARD_URL = "https://api.scryfall.com/cards/named"
SCRYFALL_SETS_URL = "https://api.scryfall.com/sets"

# Scryfall asks for 50-100 ms between requests
SCRYFALL_DELAY_SEC = 0.1

# ─── Price field (Cardmarket EUR) ────────────────────────────────────────────
# Scryfall exposes Cardmarket prices under "prices.eur"
PRICE_FIELD = "eur"  # non-foil Cardmarket EUR price

# ─── Feature engineering ─────────────────────────────────────────────────────
# MTG color identities
COLORS = ["W", "U", "B", "R", "G"]

# Card supertypes / types we track
CARD_TYPES = [
    "Creature", "Instant", "Sorcery", "Enchantment",
    "Artifact", "Planeswalker", "Land", "Battle",
]

# Keyword abilities that historically correlate with value
TRACKED_KEYWORDS = [
    "Flying", "Trample", "Deathtouch", "Lifelink", "Haste",
    "First strike", "Double strike", "Hexproof", "Indestructible",
    "Vigilance", "Flash", "Menace", "Reach", "Ward",
    "Cascade", "Convoke", "Affinity", "Annihilator",
    "Delve", "Dredge", "Storm", "Madness", "Flashback",
    "Retrace", "Escape", "Foretell", "Disturb", "Prototype",
    # Alternative-cost mechanics (added for alt-cost feature group)
    "Evoke", "Emerge", "Dash", "Overload", "Spectacle",
    "Miracle", "Mutate", "Morph", "Megamorph", "Disguise",
    "Ninjutsu", "Prowl", "Surge", "Blitz", "Plot",
    "Craft", "Channel", "Suspend", "Bestow", "Encore",
    "Offering", "Aftermath",
]

# ─── Alternative / substitutive cost keywords ────────────────────────────────
# Keywords that let you cast or play a card for a DIFFERENT cost than its mana
# cost (not additional costs like Kicker or Entwine).
ALT_COST_KEYWORDS = {
    "Evoke", "Emerge", "Dash", "Overload", "Spectacle",
    "Madness", "Miracle", "Mutate", "Morph", "Megamorph",
    "Disguise", "Ninjutsu", "Prowl", "Surge", "Blitz",
    "Plot", "Craft", "Channel", "Suspend", "Bestow",
    "Encore", "Offering", "Aftermath",
    "Flashback", "Retrace", "Escape", "Foretell", "Disturb",
}

# Rarities
RARITIES = ["common", "uncommon", "rare", "mythic"]

# ─── Sample weights — give rare/mythic more influence during training ─────────
RARITY_WEIGHTS = {
    "common": 1.0,
    "uncommon": 2.0,
    "rare": 6.0,
    "mythic": 10.0,
    "special": 8.0,
}
# Additional price-based weight: cards above this get extra weight
PRICE_WEIGHT_THRESHOLD = 2.0   # € — cards above this get boosted
PRICE_WEIGHT_FACTOR = 3.0      # multiplier for expensive cards

# ─── Model hyper-parameters (XGBoost) ────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 9,
    "learning_rate": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "colsample_bylevel": 0.65,
    "colsample_bynode": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "min_child_weight": 3,
    "gamma": 0.15,
    "random_state": 42,
    "early_stopping_rounds": 80,
    "objective": "reg:pseudohubererror",   # robust to outliers
    "huber_slope": 1.0,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Set types considered "premium" (higher expected prices) ─────────────────
PREMIUM_SET_TYPES = {
    "masters", "masterpiece", "from_the_vault", "premium_deck",
    "spellbook", "arsenal",
}

# ─── Ensure directories exist ────────────────────────────────────────────────
for d in (DATA_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ─── Neon PostgreSQL ─────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_jVy8dY5PCHOR@ep-proud-shape-abz5266u"
    ".eu-west-2.aws.neon.tech/neondb?sslmode=require",
)
