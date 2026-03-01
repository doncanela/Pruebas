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
METAGAME_CACHE_PATH = os.path.join(DATA_DIR, "metagame_cache.json")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

# Reserved List specialist model
RL_MODEL_PATH = os.path.join(MODEL_DIR, "reserved_list_model.joblib")
RL_SCALER_PATH = os.path.join(MODEL_DIR, "reserved_list_scaler.joblib")
RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "reserved_list_feature_columns.joblib")

# Lasso regression model
LASSO_MODEL_PATH = os.path.join(MODEL_DIR, "lasso_model.joblib")
LASSO_SCALER_PATH = os.path.join(MODEL_DIR, "lasso_scaler.joblib")
LASSO_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "lasso_feature_columns.joblib")
LASSO_RL_MODEL_PATH = os.path.join(MODEL_DIR, "lasso_reserved_list_model.joblib")
LASSO_RL_SCALER_PATH = os.path.join(MODEL_DIR, "lasso_reserved_list_scaler.joblib")
LASSO_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "lasso_reserved_list_feature_columns.joblib")

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

# ─── Reserved List model hyper-parameters ─────────────────────────────────────
# Smaller dataset (~870 cards) → shallower trees, more regularisation, no
# outlier cap so the model can learn the full €0–€30k range.
RL_XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "colsample_bylevel": 0.60,
    "colsample_bynode": 0.80,
    "reg_alpha": 3.0,
    "reg_lambda": 8.0,
    "min_child_weight": 5,
    "gamma": 0.3,
    "random_state": 42,
    "early_stopping_rounds": 60,
    "objective": "reg:squarederror",
}

# ─── Lasso regression hyper-parameters ───────────────────────────────────────
# LassoCV auto-tunes alpha via cross-validation. We set a search grid and
# let sklearn pick the best regularisation strength.
LASSO_PARAMS = {
    "alphas": 100,           # number of alpha values to try in CV
    "cv": 5,                 # cross-validation folds
    "max_iter": 20_000,      # convergence iterations
    "tol": 1e-4,
    "random_state": 42,
    "n_jobs": -1,            # use all cores for CV
}

# ─── Set types considered "premium" (higher expected prices) ─────────────────
PREMIUM_SET_TYPES = {
    "masters", "masterpiece", "from_the_vault", "premium_deck",
    "spellbook", "arsenal",
}

# ─── Format categories ──────────────────────────────────────────────────────
# Competitive formats where metagame shifts / bans move prices sharply
COMPETITIVE_FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage"]
# Casual / eternal formats with stickier demand
CASUAL_FORMATS = ["commander", "pauper", "oathbreaker", "penny"]

# ─── Set era boundaries (for print-run size proxy) ──────────────────────────
# Smaller print runs → more scarcity. Approximate era boundaries:
SET_ERA_BOUNDARIES = {
    # (start_year, end_year, era_label, print_run_proxy)
    "alpha_beta":     (1993, 1994, 1),   # tiny runs
    "early":          (1994, 1998, 2),   # small
    "pre_modern":     (1998, 2003, 3),   # medium
    "modern_frame":   (2003, 2008, 4),   # growing
    "mythic_era":     (2008, 2015, 5),   # mythic rarity introduced
    "modern_era":     (2015, 2020, 6),   # high volume
    "current":        (2020, 2030, 7),   # very high volume
}

# ─── Secret Lair / Universes Beyond set types or codes ──────────────────────
SECRET_LAIR_SET_TYPES = {"box", "funny"}  # Scryfall uses various types
UNIVERSES_BEYOND_MARKERS = {
    "who", "ltr", "ltc", "40k", "pip", "acr",  # known UB set codes
}

# ─── Standard rotation year-set mapping ─────────────────────────────────────
# Standard rotates annually in the fall; sets within ~2 years are "safe"
STANDARD_ROTATION_MONTHS = 24  # approximate months a set stays in Standard

# ─── Ensure directories exist ────────────────────────────────────────────────
for d in (DATA_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ─── Neon PostgreSQL ─────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_jVy8dY5PCHOR@ep-proud-shape-abz5266u"
    ".eu-west-2.aws.neon.tech/neondb?sslmode=require",
)
