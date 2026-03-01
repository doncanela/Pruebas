# MTG Card Price Predictor

Predict **Cardmarket EUR prices** of Magic: The Gathering cards **~2 months after release** using machine learning (XGBoost).

Data is sourced from the **Scryfall API** (Cardmarket prices + card metadata) and **MTGGoldfish** (competitive metagame tournament data).

**Current model**: 277 features · R² 0.54 · MAE €1.12 · SMAPE 46% · temporal split · zero leakage · trained on 77,000+ cards

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
- Engineer 277 features from the raw card data (zero leakage)
- Enrich with MTGGoldfish metagame tournament usage data
- Temporal train/test split (older sets → train, newer → test)
- Train an XGBoost regressor with log-transformed prices and sample weights
- Train an expensive-card specialist (two-stage model for cards ≥€20)
- Print evaluation metrics (MAE, MedAE, RMSE, R², MAPE, SMAPE), per-rarity and per-price-bracket breakdowns
- Show top-25 feature importances
- Save the model + specialist to `models/`

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

### 8. Status

```bash
python main.py info
```

---

## Project structure

```
CardPricePredictor/
├── main.py                # CLI entry point (12 commands)
├── config.py              # Paths, API URLs, hyperparameters, constants
├── data_collector.py      # Scryfall API data fetching + reprint/metagame enrichment
├── feature_engineer.py    # Raw JSON → 277 numeric features (leakage-free)
├── model.py               # XGBoost training & evaluation
├── predictor.py           # Single-card inference (auto-enriches with metagame)
├── metagame_collector.py  # MTGGoldfish scraper (Standard/Pioneer/Modern/Legacy/Vintage)
├── price_history.py       # Prediction logging + price snapshots (local + DB)
├── db.py                  # Neon PostgreSQL integration (cards, predictions, snapshots)
├── generate_report.py     # PDF validation report generator
├── weekly_snapshot.py     # Automated weekly snapshot script
├── requirements.txt
├── data/                  # (auto-created) raw data, features CSV, metagame cache
├── models/                # (auto-created) trained model artifacts
├── reports/               # (auto-created) PDF reports
└── logs/                  # (auto-created) automation logs
```

---

## How it works

1. **Data collection**: The Scryfall bulk API provides every Magic card ever printed, including Cardmarket EUR prices (`prices.eur`). We filter for English paper cards with a known price (~77,000 cards).

2. **Reprint enrichment**: For each card, we compute reprint count, average/min/max/std price of older printings, days since last reprint, and oldest printing age — strong signals for predicting new print prices.

3. **Metagame enrichment**: MTGGoldfish "Format Staples" pages are scraped to get the top-50 most-played cards in each competitive format (Standard, Pioneer, Modern, Legacy, Vintage) with their usage percentages and average copies per deck. Cached for 7 days.

4. **Feature engineering**: Each card is converted to 277 numeric features across 20 groups — from basic stats to metagame demand, supply scarcity, Reserved List status, ban impact, and seasonality signals. All features are audited for leakage: no foil prices, no time-since-release, no current-market-price dependencies.

5. **Model**: An XGBoost gradient-boosted tree regressor is trained on log-transformed prices with PseudoHuber loss function (robust to outliers). Sample weights boost rare/mythic and expensive cards. **Temporal train/test split** (train on older sets, test on newer) prevents era-mixing leakage. A **two-stage specialist** re-predicts expensive cards (≥€20) with a dedicated model blended 60/40 with the base predictions.

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

### Base model

| Metric | Value |
|---|---|
| **MAE** | €1.12 |
| **Median AE** | €0.14 |
| **RMSE** | €4.63 |
| **R²** | 0.54 |
| **MAPE** | 64.5% |
| **SMAPE** | 46.1% |

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

| Bracket | MAE | MAPE | SMAPE | n |
|---|---|---|---|---|
| €0–€0.50 | €0.14 | 73.0% | 47.9% | 8,980 |
| €0.50–€1 | €0.48 | 67.7% | 44.4% | 1,631 |
| €1–€2 | €0.85 | 60.5% | 44.0% | 1,260 |
| €2–€5 | €1.62 | 50.0% | 43.1% | 1,595 |
| €5–€10 | €3.03 | 43.1% | 40.6% | 951 |
| €10–€20 | €5.11 | 37.0% | 40.8% | 608 |
| €20–€50 | €10.64 | 35.4% | 47.5% | 336 |
| €50+ | €46.08 | 52.0% | 83.1% | 39 |

> **Note on R²**: The previous R² of 0.71 was inflated by (a) random train/test split mixing eras, (b) foil-price features that leaked current market info, and (c) time-since-release features. The current R² of 0.54 on a strict temporal split with zero leakage is a far more honest measure of real-world prediction quality.

---

## Notes

- **Data sources**: Scryfall (free, includes Cardmarket EUR prices) + MTGGoldfish (free, format staples). No API keys needed.
- **Rate limiting**: Scryfall requires 50–100 ms between requests; MTGGoldfish gets 1.5 s delays. Both are respected.
- **Database**: Optional Neon PostgreSQL stores cards, predictions, and price snapshots for long-term tracking.
- **Price target**: The model predicts `prices.eur` (Cardmarket trend price). For new cards, this approximates the ~2-month settled price.
- **Reserved List**: Cards on the Reserved List are detected via Scryfall's `reserved` field — the model's #1 most important interaction feature.
- **Metagame**: Tournament usage from MTGGoldfish covers the top-50 staples per format. Now includes **missingness flags** (`meta_{fmt}_unknown`) to distinguish "legal but not in data" from "not legal" — the model can learn that being legal-but-not-a-staple is different from being banned/illegal.
