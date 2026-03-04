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

# Random Forest model
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
RF_SCALER_PATH = os.path.join(MODEL_DIR, "rf_scaler.joblib")
RF_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "rf_feature_columns.joblib")
RF_RL_MODEL_PATH = os.path.join(MODEL_DIR, "rf_reserved_list_model.joblib")
RF_RL_SCALER_PATH = os.path.join(MODEL_DIR, "rf_reserved_list_scaler.joblib")
RF_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "rf_reserved_list_feature_columns.joblib")

# TabNet model
TABNET_MODEL_DIR = os.path.join(MODEL_DIR, "tabnet")
TABNET_RL_MODEL_DIR = os.path.join(MODEL_DIR, "tabnet_rl")
TABNET_SCALER_PATH = os.path.join(MODEL_DIR, "tabnet_scaler.joblib")
TABNET_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "tabnet_feature_columns.joblib")
TABNET_RL_SCALER_PATH = os.path.join(MODEL_DIR, "tabnet_rl_scaler.joblib")
TABNET_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "tabnet_rl_feature_columns.joblib")

# Elastic Net model (TF-IDF text baseline)
ELASTICNET_MODEL_PATH = os.path.join(MODEL_DIR, "elasticnet_model.joblib")
ELASTICNET_SCALER_PATH = os.path.join(MODEL_DIR, "elasticnet_scaler.joblib")
ELASTICNET_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "elasticnet_feature_columns.joblib")
ELASTICNET_TFIDF_PATH = os.path.join(MODEL_DIR, "elasticnet_tfidf.joblib")
ELASTICNET_RL_MODEL_PATH = os.path.join(MODEL_DIR, "elasticnet_rl_model.joblib")
ELASTICNET_RL_SCALER_PATH = os.path.join(MODEL_DIR, "elasticnet_rl_scaler.joblib")
ELASTICNET_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "elasticnet_rl_feature_columns.joblib")
ELASTICNET_RL_TFIDF_PATH = os.path.join(MODEL_DIR, "elasticnet_rl_tfidf.joblib")

# LightGBM model
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.joblib")
LGBM_SCALER_PATH = os.path.join(MODEL_DIR, "lgbm_scaler.joblib")
LGBM_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "lgbm_feature_columns.joblib")
LGBM_RL_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_rl_model.joblib")
LGBM_RL_SCALER_PATH = os.path.join(MODEL_DIR, "lgbm_rl_scaler.joblib")
LGBM_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "lgbm_rl_feature_columns.joblib")

# CatBoost model
CATBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")
CATBOOST_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "catboost_feature_columns.joblib")
CATBOOST_RL_MODEL_PATH = os.path.join(MODEL_DIR, "catboost_rl_model.cbm")
CATBOOST_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "catboost_rl_feature_columns.joblib")

# Two-stage bulk/non-bulk model
TWOSTAGE_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "twostage_classifier.joblib")
TWOSTAGE_REGRESSOR_PATH = os.path.join(MODEL_DIR, "twostage_regressor.joblib")
TWOSTAGE_SCALER_PATH = os.path.join(MODEL_DIR, "twostage_scaler.joblib")
TWOSTAGE_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "twostage_feature_columns.joblib")
TWOSTAGE_RL_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "twostage_rl_classifier.joblib")
TWOSTAGE_RL_REGRESSOR_PATH = os.path.join(MODEL_DIR, "twostage_rl_regressor.joblib")
TWOSTAGE_RL_SCALER_PATH = os.path.join(MODEL_DIR, "twostage_rl_scaler.joblib")
TWOSTAGE_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "twostage_rl_feature_columns.joblib")
TWOSTAGE_BULK_THRESHOLD = 0.50  # € — bulk vs non-bulk boundary

# Quantile regression model (LightGBM quantile)
QUANTILE_MODEL_P10_PATH = os.path.join(MODEL_DIR, "quantile_p10.joblib")
QUANTILE_MODEL_P50_PATH = os.path.join(MODEL_DIR, "quantile_p50.joblib")
QUANTILE_MODEL_P90_PATH = os.path.join(MODEL_DIR, "quantile_p90.joblib")
QUANTILE_SCALER_PATH = os.path.join(MODEL_DIR, "quantile_scaler.joblib")
QUANTILE_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "quantile_feature_columns.joblib")
QUANTILE_RL_MODEL_P10_PATH = os.path.join(MODEL_DIR, "quantile_rl_p10.joblib")
QUANTILE_RL_MODEL_P50_PATH = os.path.join(MODEL_DIR, "quantile_rl_p50.joblib")
QUANTILE_RL_MODEL_P90_PATH = os.path.join(MODEL_DIR, "quantile_rl_p90.joblib")
QUANTILE_RL_SCALER_PATH = os.path.join(MODEL_DIR, "quantile_rl_scaler.joblib")
QUANTILE_RL_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "quantile_rl_feature_columns.joblib")

# TF-IDF + SVD text pipeline (shared by all models)
TFIDF_SVD_PATH = os.path.join(MODEL_DIR, "tfidf_svd_pipeline.joblib")
TFIDF_N_COMPONENTS = 50  # number of SVD dimensions

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
    "Offering", "Aftermath", "Transmute", "Split second",
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
    "Transmute",
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

# ─── Random Forest hyper-parameters ──────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": 42,
}

RF_RL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": 0.6,
    "n_jobs": -1,
    "random_state": 42,
}

# ─── TabNet hyper-parameters ─────────────────────────────────────────────────
TABNET_PARAMS = {
    "n_d": 32,               # width of decision prediction layer
    "n_a": 32,               # width of attention embedding layer
    "n_steps": 5,            # number of sequential attention steps
    "gamma": 1.5,            # coefficient for feature reusage in attention
    "n_independent": 2,      # number of independent GLU layers
    "n_shared": 2,           # number of shared GLU layers
    "lambda_sparse": 1e-3,   # sparsity regularisation
    "momentum": 0.3,
    "mask_type": "entmax",
}

TABNET_FIT_PARAMS = {
    "max_epochs": 200,
    "patience": 30,           # early stopping patience
    "batch_size": 2048,
    "virtual_batch_size": 256,
}

TABNET_RL_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 3,
    "gamma": 1.5,
    "n_independent": 1,
    "n_shared": 1,
    "lambda_sparse": 1e-2,
    "momentum": 0.3,
    "mask_type": "entmax",
}

TABNET_RL_FIT_PARAMS = {
    "max_epochs": 300,
    "patience": 40,
    "batch_size": 128,
    "virtual_batch_size": 64,
}

# ─── Elastic Net hyper-parameters ────────────────────────────────────────────
ELASTICNET_PARAMS = {
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],  # mix L1 / L2
    "cv": 5,
    "max_iter": 20_000,
    "tol": 1e-4,
    "random_state": 42,
    "n_jobs": -1,
}

# ─── LightGBM hyper-parameters ──────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "huber",
}

LGBM_RL_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_alpha": 3.0,
    "reg_lambda": 8.0,
    "min_child_samples": 5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "objective": "regression",
}

# ─── CatBoost hyper-parameters ──────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations": 3000,
    "depth": 8,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 200,
    "loss_function": "RMSE",
    "early_stopping_rounds": 80,
}

CATBOOST_RL_PARAMS = {
    "iterations": 2000,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 8.0,
    "random_seed": 42,
    "verbose": 100,
    "loss_function": "RMSE",
    "early_stopping_rounds": 60,
}

# CatBoost categorical columns (these start with _cat_ in the feature matrix)
CATBOOST_CAT_FEATURES = [
    "_cat_rarity", "_cat_color_identity", "_cat_type_bucket",
]

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
    "who", "ltr", "ltc", "40k", "pip", "acr",    # known UB set codes
    "scd", "rex", "fin", "spg",                    # Final Fantasy, Jurassic World, etc.
    "pltr", "p40k",                                  # promo variants
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
