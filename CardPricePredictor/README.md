# MTG Card Price Predictor

Predict **Cardmarket EUR Trend prices** of Magic: The Gathering cards using machine learning (XGBoost, Random Forest, TabNet, Lasso).

This is a **cross-sectional price model**: it learns the relationship between card attributes and current market value across ~77,000 cards. For newly released cards it provides a useful price estimate based on how similar attributes are valued across the rest of the market.

Data is sourced from the **Scryfall API** (Cardmarket prices + card metadata) and **MTGGoldfish** (competitive metagame tournament data).

**XGBoost model**: 279 features · R² 0.53 · MAE €1.13 · SMAPE 46% · temporal split · zero leakage · trained on 77,000+ cards

**Random Forest model**: 1000 trees · R² 0.56 · MAE €1.07 · SMAPE 44% · impurity-based feature importance · strong ensemble baseline

**TabNet model**: attention-based deep learning · R² 0.29 · MAE €1.53 · SMAPE 64% · learned feature selection via attention masks · interpretable neural network

**Lasso model**: 29/277 features selected · R² 0.23 · MAE €2.25 · SMAPE 106% · auto-tuned alpha via LassoCV · interpretable linear baseline

**Two-model architecture**: Reserved List cards (869 cards, €0–€31K) are routed to a dedicated specialist model, while all other cards use the main model. All four model families (XGBoost, RF, TabNet, Lasso) maintain their own RL specialists.

---

## Features used by the model (277 total)

| # | Category | Features |
|---|---|---|
| 1 | **Mana cost** | CMC, per-color pips, generic mana, hybrid/Phyrexian/X indicators |
| 2 | **Card types** | Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle, subtypes |
| 3 | **Rarity** | Common, Uncommon, Rare, Mythic (one-hot + numeric) |
| 4 | **Stats** | Power, Toughness, Loyalty, variable stats flag |
| 5 | **Oracle text** | Length, word count, keyword counts (target, draw, destroy, exile, graveyard, etc.) |
| 6 | **Keywords** | 50+ tracked keywords (Flying, Cascade, Storm, Flashback, Evoke, Ninjutsu, …) |
| 7 | **Reprint history** | Reprint count, avg/min/max/std price of prior printings, days since last reprint |
| 8 | **Set info** | Set size |
| 9 | **Legality** | # of legal formats, per-format legal flags (Commander, Modern, Standard, Pioneer, Legacy, Vintage, Pauper) |
| 10 | **Misc** | Legendary, modal DFC, adventure, saga, full art, promo, EDHREC rank |
| 11 | **Special treatments** | Serialized, showcase, retro frame, extended art, etched, promo types, frame year |
| 12 | **Interaction terms** | Rarity × EDHREC, rarity × reprint count, rarity × formats, CMC × rarity |
| 13 | **Color identity** | Bitmask, identity size, multicolor/five-color flags |
| 14 | **Alternative costs** | 22 alt-cost keywords (Evoke, Emerge, Dash, etc.), 7 text patterns, interaction terms |
| 15 | **Format demand** | Competitive vs casual format counts, multi-format staple flag, Commander demand score, EDHREC tiers (top 100/500/1000), Legacy/Vintage-only premium, Standard rotation risk |
| 16 | **Supply / scarcity** | Set era (7 tiers), print-run proxy, supply scarcity score, old-rare/scarce flags (design age), reprint drought intensity, Secret Lair & Universes Beyond detection |
| 17 | **Reprint risk** | Reserved List flag, reprint frequency (design age), vulnerability/immunity, ceiling pressure, Reserved List interaction terms |
| 18 | **Ban / restrict** | Per-format ban & restrict flags, ban count, ban impact score (weighted by format), ban ratio, restricted power signal |
| 19 | **Seasonality** | Release month/quarter, fall set & competitive season flags (calendar features only, no time leakage) |
| 20 | **Metagame (MTGGoldfish)** | Per-format tournament usage % and avg copies, **missingness flags** (legal-but-not-in-data vs not-legal), staple/allstar tiers, competitive demand score, cross-format interactions |

### Leakage audit (removed features)

The following features were **removed** to eliminate data leakage:

- **Foil price features** (`foil_price`, `foil_multiplier`, `has_foil_price`, `foil_mult_x_rarity`): current Cardmarket market prices — using the answer to predict the answer.
- **Time-since-release features** (`days_since_release`, `months_since_release`, `years_since_release`, recency buckets, `hype_decay`, `rotation_proximity`, etc.): at prediction time these are constants (~60 days), so training with variable values conflates card age with card quality.
- **Age-dependent features** were replaced with `oldest_printing_days` (the card's *design age* — known at release) instead of `days_since_release` (this printing's age).

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect card data

```bash
# Download recent sets (last 25 years)
python main.py collect

# Or download ALL cards (~300 MB bulk, slower)
python main.py collect --mode bulk

# Only sets since a given date
python main.py collect --since 2024-06-01

# Skip syncing to Neon DB
python main.py collect --no-sync-db
```

### 3. Train the model

```bash
# Train using cached features (fast if features.csv exists)
python main.py train

# Force re-engineering all features from raw data
python main.py train --rebuild-features
```

This will:
- Engineer 279 features from the raw card data (zero leakage)
- Enrich with MTGGoldfish metagame tournament usage data
- **Stage 1 — Main model**: Temporal train/test split, XGBoost with PseudoHuber loss, sample weights, expensive-card specialist (two-stage for cards ≥€20). Reserved List cards are excluded.
- **Stage 2 — Reserved List model**: A dedicated XGBoost model trained only on ~869 RL cards with no outlier cap (handles €0–€31K), squared-error loss, and stronger regularisation for the small dataset.
- Print evaluation metrics (MAE, MedAE, RMSE, R², MAPE, SMAPE), per-rarity and per-price-bracket breakdowns
- Show top-25 feature importances
- Save all models to `models/`

### 3b. Train the Random Forest model

```bash
python main.py train-rf
python main.py train-rf --rebuild-features
```

Trains a **Random Forest** (1000 trees, max_depth=20) with the same two-model architecture:
- Main model on ~76K non-RL cards + RL specialist on ~869 cards
- Same temporal split, sample weights, and evaluation as XGBoost
- Impurity-based feature importance ranking

### 3c. Train the TabNet model

```bash
python main.py train-tabnet
python main.py train-tabnet --rebuild-features
```

Trains a **TabNet** (attention-based deep learning) model:
- n_d=32, n_a=32, 5 attention steps, entmax masking
- Early stopping on validation MAE (patience=30)
- Learns feature importance via attention masks
- Both main model and RL specialist

### 3d. Train the Lasso model (linear baseline)

```bash
python main.py train-lasso
python main.py train-lasso --rebuild-features
```

This trains a **LassoCV** (L1-regularised linear regression) as a simpler, interpretable baseline:
- Automatically selects the best regularisation alpha via 5-fold cross-validation
- Performs built-in feature selection (drives irrelevant coefficients to zero)
- Trains both a main model (non-RL) and a Reserved List specialist
- Same temporal split, sample weights, and evaluation as the XGBoost pipeline

### 4. Predict a card's price

```bash
# Single card
python main.py predict "Sheoldred, the Apocalypse"

# Specific printing (by set code)
python main.py predict "Lightning Bolt" --set lea

# Multiple cards at once
python main.py batch "Ragavan, Nimble Pilferer" "The One Ring" "Orcish Bowmasters"

# Save batch results to JSON
python main.py batch "Force of Will" "Mana Crypt" -o results.json

# Predict using the Random Forest model
python main.py predict-rf "Sheoldred, the Apocalypse"
python main.py predict-rf "Lightning Bolt" --set lea

# Predict using the TabNet model
python main.py predict-tabnet "Sheoldred, the Apocalypse"
python main.py predict-tabnet "Lightning Bolt" --set lea

# Predict using the Lasso model
python main.py predict-lasso "Sheoldred, the Apocalypse"
python main.py predict-lasso "Lightning Bolt" --set lea
```

### 5. Metagame data

```bash
# Fetch/refresh tournament metagame data from MTGGoldfish
python main.py metagame --force

# Look up a specific card's metagame usage
python main.py metagame --card "Force of Will"
```

### 6. Price tracking & history

```bash
# Take a price snapshot of all cards (stores locally + Neon DB)
python main.py snapshot

# View prediction log and price history for a card
python main.py history "Sheoldred, the Apocalypse"

# Check accuracy of past predictions vs current prices
python main.py accuracy
```

### 7. Database (Neon PostgreSQL)

```bash
# Initialize the database tables
python main.py db-init

# Show database statistics
python main.py db-stats

# Sync local card data + snapshot to Neon DB
python main.py sync-db
```

### 8. Batch predictions & report

```bash
# Predict all rare/mythic cards (~32,000) — auto-routes RL cards to specialist model
python batch_predict_all.py
# Output: output/rare_mythic_predictions.csv

# Generate an 11-page PDF report of biggest price differences
python generate_diff_report.py
# Output: output/report/price_difference_report.pdf + individual PNGs
```

### 9. Status

```bash
python main.py info
```

---

## Project structure

```
CardPricePredictor/
├── main.py                # CLI entry point (14 commands)
├── config.py              # Paths, API URLs, hyperparameters, constants
├── data_collector.py      # Scryfall API data fetching + reprint/metagame enrichment
├── feature_engineer.py    # Raw JSON → 279 numeric features (leakage-free)
├── model.py               # XGBoost training & evaluation (main + Reserved List models)
├── model_rf.py            # Random Forest training & evaluation (main + Reserved List)
├── model_tabnet.py        # TabNet (deep learning) training & evaluation (main + RL)
├── model_lasso.py         # Lasso (L1) training & evaluation (main + Reserved List)
├── predictor.py           # Single-card inference (XGBoost/RF/TabNet/Lasso, auto-routes RL)
├── metagame_collector.py  # MTGGoldfish scraper (Standard/Pioneer/Modern/Legacy/Vintage)
├── price_history.py       # Prediction logging + price snapshots (local + DB)
├── db.py                  # Neon PostgreSQL integration (cards, predictions, snapshots)
├── generate_report.py     # PDF validation report generator (sample-based)
├── generate_diff_report.py # PDF report of biggest price differences (batch predictions)
├── batch_predict_all.py   # Batch-predict all rare/mythic cards (routes RL automatically)
├── weekly_snapshot.py     # Automated weekly snapshot script
├── requirements.txt
├── data/                  # (auto-created) raw data, features CSV, metagame cache
├── models/                # (auto-created) trained model artifacts
├── output/                # (auto-created) batch predictions CSV + visual report PDFs
├── reports/               # (auto-created) PDF reports
└── logs/                  # (auto-created) automation logs
```

---

## How it works

1. **Data collection**: The Scryfall bulk API provides every Magic card ever printed, including Cardmarket EUR prices (`prices.eur`) as of the download date. We filter for English paper cards with a known price (~77,000 cards). Prices reflect the **current** market state, not historical values.

2. **Reprint enrichment**: For each card, we compute reprint count, average/min/max/std price of older printings (using current Scryfall prices, not historical), days since last reprint, and oldest printing age.

3. **Metagame enrichment**: MTGGoldfish "Format Staples" pages are scraped to get the top-50 most-played cards in each competitive format (Standard, Pioneer, Modern, Legacy, Vintage) with their usage percentages and average copies per deck. Cached for 7 days.

4. **Feature engineering**: Each card is converted to 277 numeric features across 20 groups — from basic stats to metagame demand, supply scarcity, Reserved List status, ban impact, and seasonality signals. All features are audited for intra-snapshot leakage (no foil prices, no time-since-release). See the Temporal Caveats section for inter-snapshot considerations (EDHREC, metagame, reprint prices).

5. **Model (two-model architecture)**:
   - **Main model (XGBoost)**: Trained on ~76,000 non-Reserved-List cards with log-transformed prices, PseudoHuber loss, sample weights, temporal split, and a two-stage expensive-card specialist (≥€20) blended 60/40.
   - **Reserved List model (XGBoost)**: A dedicated XGBoost regressor trained on ~869 RL cards with squared-error loss, no outlier cap (prices span €0.02–€31,000), stronger regularisation, and heavier weighting on expensive cards.
   - **Random Forest**: 1000 trees (max_depth=20) with the same temporal split, sample weights, and log-target pipeline. Provides a strong tree-ensemble baseline without boosting; uses impurity-based feature importance.
   - **TabNet**: Attention-based deep learning (n_d=32, n_a=32, 5 steps, entmax masking). Trains with early stopping on validation MAE. Learns sparse feature selection via sequential attention — the attention masks reveal which features each card "looks at".
   - **Lasso alternative**: A LassoCV linear model (L1 regularisation) trained with the same pipeline. Auto-tunes alpha, selects ~29 features from 277, and provides an interpretable baseline.
   - At inference, cards are automatically routed to the correct RL/main model based on the `reserved` flag.

6. **Inference**: Given a card name, we fetch its data from Scryfall, enrich with reprint info and metagame data, engineer the same 277 features, and run the model. The prediction is automatically logged to both a local JSONL file and the Neon PostgreSQL database.

7. **Automated tracking**: Weekly snapshots of all card prices can run automatically via Windows Task Scheduler (local) or GitHub Actions (cloud), storing data in Neon DB for long-term price trend analysis.

---

## Automation

### Windows Task Scheduler
A scheduled task `MTG_CardPrice_Weekly_Snapshot` runs `weekly_snapshot.py` every Sunday at 10:00 AM, taking a price snapshot and syncing to the database.

### GitHub Actions
A workflow (`.github/workflows/weekly_snapshot.yml`) runs the same snapshot every Sunday at 10:00 UTC without requiring your PC to be on. Uses the `NEON_DATABASE_URL` secret for database access.

---

## Model performance

Evaluated on a **temporal test set** (most recent ~2.4 years of cards, trained on older sets). This is a harder, more honest evaluation than random splitting.

### Main model (non-Reserved-List cards, ~76,258 cards)

| Metric | Value |
|---|---|
| **MAE** | €1.13 |
| **Median AE** | €0.14 |
| **RMSE** | €4.63 |
| **R²** | 0.53 |
| **MAPE (all ≥€0.05)** | 64.5% |
| **MAPE (≥€1 only)** | 48.8% |
| **SMAPE** | 46.1% |

> **On MAPE**: The 64.5% headline number is inflated by ultra-cheap cards (€0–€0.50, n=8,980) where dividing a €0.14 error by a €0.10 price produces 140%. For cards ≥€1, MAPE drops to **48.8%** — a more meaningful number.

### Two-stage model (with expensive-card specialist)

For cards ≥€20, a specialist model trained on expensive cards only is blended with the base model (60/40), reducing high-value prediction error.

| Expensive-card MAE | Base | Specialist |
|---|---|---|
| Cards ≥€20 | €14.70 | **€9.92** |

### Per-rarity MAE

| Rarity | MAE |
|---|---|
| Common | €0.13 |
| Uncommon | €0.27 |
| Rare | €1.26 |
| Mythic | €3.66 |

### Per-price-bracket breakdown

| Bracket | MAE | nMAE | MAPE | SMAPE | n |
|---|---|---|---|---|---|
| €0–€0.50 | €0.14 | 58% | 73.0% | 47.9% | 8,980 |
| €0.50–€1 | €0.48 | 64% | 67.7% | 44.4% | 1,631 |
| €1–€2 | €0.85 | 57% | 60.5% | 44.0% | 1,260 |
| €2–€5 | €1.62 | 46% | 50.0% | 43.1% | 1,595 |
| €5–€10 | €3.03 | 40% | 43.1% | 40.6% | 951 |
| €10–€20 | €5.11 | 34% | 37.0% | 40.8% | 608 |
| €20–€50 | €10.64 | 30% | 35.4% | 47.5% | 336 |
| €50+ | €46.08 | 46% | 52.0% | 83.1% | 39 |

**nMAE** = bracket-normalized MAE (MAE ÷ bracket midpoint), comparable across price tiers.

### Bracket classification accuracy

| Metric | Value |
|---|---|
| Exact bracket match | 71.1% |
| Within ±1 bracket | 94.2% |

### Within-X% accuracy (cards ≥€1, n=4,789)

| Threshold | Accuracy |
|---|---|
| Within ±25% | 43.4% |
| Within ±50% | 67.6% |
| Within ±75% | 83.0% |
| Within ±100% | 91.1% |

### Reserved List specialist model (~869 cards)

RL cards span €0.02–€30,975 and behave very differently from standard cards. A dedicated model with squared-error loss and stronger regularisation handles these:

| Metric | Value |
|---|---|
| **MAE** | €56.57 |
| **Median AE** | €1.43 |
| **RMSE** | €604.49 |
| **R²** | −2.24 |

> **Why R² is negative**: With only 869 samples spanning 5 orders of magnitude and a strict temporal split, the model's squared-error predictions are worse than predicting the mean for the test fold. However, the model correctly captures pricing structure for high-value staples (e.g. Black Lotus predicted ~€17K vs old main model's €287). The negative R² reflects the extreme difficulty of the problem, not uselessness.

**Top features**: `meta_legacy_unknown`, `is_expansion_set`, `century_plus_reprints`, `meta_vintage_unknown`, `edhrec_rank`

### Random Forest model (~76,258 non-RL cards)

| Metric | XGBoost | RF |
|---|---|---|
| **MAE** | €1.13 | €1.07 |
| **MedAE** | €0.14 | €0.13 |
| **RMSE** | €4.63 | €4.53 |
| **R²** | 0.53 | 0.56 |
| **SMAPE** | 46% | 44% |
| **Bracket accuracy** | 71.1% | 71.6% |
| **Within ±1 bracket** | 94.2% | 94.2% |

**Per-rarity MAE**: Common €0.12 · Uncommon €0.26 · Rare €1.19 · Mythic €3.55

**Top RF features** (impurity decrease): `min_prev_price`, `avg_prev_price_x_rarity`, `avg_prev_price`, `rarity_x_edhrec`, `scarcity_x_rarity`, `supply_scarcity`, `max_prev_price`, `set_card_count`, `edhrec_rank`, `print_run_proxy`

### RF Reserved List model (~869 cards)

| Metric | Value |
|---|---|
| **MAE** | €49.55 |
| **Median AE** | €9.91 |
| **R²** | −1.24 |
| **Bracket accuracy** | 34.5% |

### TabNet model (deep learning, ~76,258 non-RL cards)

| Metric | XGBoost | TabNet |
|---|---|---|
| **MAE** | €1.13 | €1.53 |
| **MedAE** | €0.14 | €0.22 |
| **RMSE** | €4.63 | €5.77 |
| **R²** | 0.53 | 0.29 |
| **SMAPE** | 46% | 64% |
| **Bracket accuracy** | 71.1% | 62.4% |
| **Within ±1 bracket** | 94.2% | 89.3% |

**Per-rarity MAE**: Common €0.17 · Uncommon €0.42 · Rare €1.70 · Mythic €5.00

**Top TabNet features** (attention importance): `txt_sacrifice_alt`, `supply_scarcity`, `edhrec_rank`, `min_prev_price`, `rarity_numeric`, `text_you_may`, `is_colorless`, `supply_drought_years`, `is_prerelease`, `kw_miracle`

> TabNet's attention mechanism picks up different signals than tree models — notably, it focuses heavily on oracle text patterns (`txt_sacrifice_alt`, `text_you_may`) and supply indicators. Its main value is as a complementary model with different feature emphasis.

### TabNet Reserved List model (~869 cards)

| Metric | Value |
|---|---|
| **MAE** | €40.50 |
| **Median AE** | €8.41 |
| **R²** | −0.56 |
| **Bracket accuracy** | 21.3% |

### Lasso model (linear baseline, ~76,258 non-RL cards)

LassoCV automatically selects the best L1 regularisation strength and drives irrelevant coefficients to zero. This produces a simpler, more interpretable model at the cost of accuracy.

| Metric | XGBoost | RF | TabNet | Lasso |
|---|---|---|---|---|
| **MAE** | €1.13 | **€1.07** | €1.53 | €2.25 |
| **MedAE** | €0.14 | **€0.13** | €0.22 | €0.91 |
| **RMSE** | €4.63 | **€4.53** | €5.77 | €6.00 |
| **R²** | 0.53 | **0.56** | 0.29 | 0.23 |
| **SMAPE** | 46% | **44%** | 64% | 106% |
| **Features used** | 277 | 277 | 277 | 29 (10.5%) |
| **Bracket accuracy** | 71.1% | **71.6%** | 62.4% | 26.2% |

**Top Lasso features** (by absolute coefficient): `edhrec_rank` (−), `set_era` (−), `edhrec_top_1000` (+), `rarity_mythic` (+), `avg_price_per_reprint` (+), `meta_modern_unknown` (−), `rarity_numeric` (+), `scarcity_x_rarity` (+), `is_full_art` (+), `is_penny_legal` (−)

> **Model roles**: Random Forest slightly outperforms XGBoost and serves as a strong ensemble baseline. TabNet offers a complementary deep-learning perspective with attention-based feature selection. Lasso provides maximum interpretability with only 29 features. XGBoost remains the production default for its balance of accuracy and speed.

### Lasso Reserved List model (~869 cards)

| Metric | Value |
|---|---|
| **MAE** | €30.44 |
| **Median AE** | €7.97 |
| **R²** | −0.06 |
| **Features used** | 19/277 |

**Top features**: `set_era` (−), `is_core_set` (+), `edhrec_rank` (−), `is_reprint` (−), `meta_max_usage` (+)

> **Note on R²**: The previous R² of 0.71 was inflated by (a) random train/test split mixing eras, (b) foil-price features that leaked current market info, and (c) time-since-release features. The current R² of 0.53 on a strict temporal split with zero leakage is a far more honest measure of real-world prediction quality.

---

## Notes

- **Data sources**: Scryfall (free, includes Cardmarket EUR prices) + MTGGoldfish (free, format staples). No API keys needed.
- **Rate limiting**: Scryfall requires 50–100 ms between requests; MTGGoldfish gets 1.5 s delays. Both are respected.
- **Database**: Optional Neon PostgreSQL stores cards, predictions, and price snapshots for long-term tracking.
- **Price target**: The model predicts `prices.eur` (Cardmarket Trend price at download time). This is a **cross-sectional** model, not a temporal forecast — see the Temporal Caveats section below.
- **Reserved List**: Cards on the Reserved List (~869 printings) are detected via Scryfall's `reserved` field and routed to a dedicated specialist model. The main model no longer trains on RL cards, allowing each model to specialise in its price range (main: €0–€500, RL: €0–€31K).
- **Metagame**: Tournament usage from MTGGoldfish covers the top-50 staples per format. Now includes **missingness flags** (`meta_{fmt}_unknown`) to distinguish "legal but not in data" from "not legal" — the model can learn that being legal-but-not-a-staple is different from being banned/illegal.

---

## Temporal caveats (important for interpretation)

This model is a **cross-sectional price model**, not a true temporal forecaster. Understanding what this means:

### What the model actually predicts

The label (`price_eur`) is the Cardmarket Trend price **at the time Scryfall data was downloaded**, regardless of when the card was released. A 2018 card's label is its 2026 price. The model learns: "given these current attributes and current market context, what should this card cost right now?"

### Why this is useful for new cards

At inference time, you give the model a newly released card's attributes (rarity, text, keywords, metagame demand, etc.), and it compares against the learned price structure of all ~77,000 cards. This produces a reasonable estimate for what the card "should" cost based on comparable cards.

### Features that use snapshot-contemporaneous data

| Feature group | What it reflects | Consequence |
|---|---|---|
| **EDHREC rank** | Current commander popularity (from Scryfall) | A 2018 card's EDHREC rank reflects 8 years of deck data. At release, its rank would have been very different. |
| **MTGGoldfish metagame** | Current tournament metagame scraped today | Cards that are popular today may not have been popular at release. |
| **Reprint price stats** (avg/min/max/std) | Current Scryfall prices of older printings | The `sib_release <= card_release` filter ensures only older printings, but uses their *current* prices — not prices at the card's release date. |
| **Legality / bans** | Current format legality | Cards banned after release have their current (banned) status. |

### Why the temporal split is still valuable

Despite all features being snapshot-contemporaneous, the temporal train/test split tests whether price patterns learned from older sets **generalize to newer sets** (which have different mechanics, power levels, and design philosophies). This is a harder test than random splitting and avoids overfitting to set-specific patterns.

### What would be needed for a true +60-day forecast

1. **Historical price snapshots** — store `price_at(release_date + 60)` as the label
2. **Point-in-time EDHREC ranks** — version ranks by date
3. **Point-in-time metagame** — historical MTGGoldfish snapshots
4. **Point-in-time reprint prices** — prices of older printings as of the prediction date

The weekly snapshot system (Neon DB + GitHub Actions) is a step toward accumulating this data.
