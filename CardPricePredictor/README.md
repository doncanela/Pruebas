# MTG Card Price Predictor

Predict **Cardmarket EUR prices** of Magic: The Gathering cards **~2 months after release** using machine learning (XGBoost).

Data is sourced from the **Scryfall API** (Cardmarket prices + card metadata) and **MTGGoldfish** (competitive metagame tournament data).

**Current model**: 290 features · R² 0.71 · MAE €1.10 · trained on 77,000+ cards

---

## Features used by the model (290 total)

| # | Category | Features |
|---|---|---|
| 1 | **Mana cost** | CMC, per-color pips, generic mana, hybrid/Phyrexian/X indicators |
| 2 | **Card types** | Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle, subtypes |
| 3 | **Rarity** | Common, Uncommon, Rare, Mythic (one-hot + numeric) |
| 4 | **Stats** | Power, Toughness, Loyalty, variable stats flag |
| 5 | **Oracle text** | Length, word count, keyword counts (target, draw, destroy, exile, graveyard, etc.) |
| 6 | **Keywords** | 50+ tracked keywords (Flying, Cascade, Storm, Flashback, Evoke, Ninjutsu, …) |
| 7 | **Reprint history** | Reprint count, avg/min/max/std price of prior printings, days since last reprint |
| 8 | **Set info** | Days since release, set size |
| 9 | **Legality** | # of legal formats, per-format legal flags (Commander, Modern, Standard, Pioneer, Legacy, Vintage, Pauper) |
| 10 | **Misc** | Legendary, modal DFC, adventure, saga, full art, promo, EDHREC rank, foil price/multiplier |
| 11 | **Special treatments** | Serialized, showcase, retro frame, extended art, etched, promo types, frame year |
| 12 | **Interaction terms** | Rarity × EDHREC, rarity × reprint count, rarity × formats, CMC × rarity |
| 13 | **Color identity** | Bitmask, identity size, multicolor/five-color flags |
| 14 | **Alternative costs** | 22 alt-cost keywords (Evoke, Emerge, Dash, etc.), 7 text patterns, interaction terms |
| 15 | **Format demand** | Competitive vs casual format counts, multi-format staple flag, Commander demand score, EDHREC tiers (top 100/500/1000), Legacy/Vintage-only premium, Standard rotation risk |
| 16 | **Supply / scarcity** | Set era (7 tiers), print-run proxy, supply scarcity score, old-rare/scarce flags, reprint drought intensity, Secret Lair & Universes Beyond detection |
| 17 | **Reprint risk** | Reserved List flag, reprint frequency, reprint vulnerability/immunity, reprint ceiling pressure, Reserved List interaction terms |
| 18 | **Ban / restrict** | Per-format ban & restrict flags, ban count, ban impact score (weighted by format), ban ratio, restricted power signal |
| 19 | **Seasonality** | Release month/quarter, fall set & competitive season flags, recency buckets, Standard rotation proximity, hype decay curve |
| 20 | **Metagame (MTGGoldfish)** | Per-format tournament usage % and avg copies for Standard/Pioneer/Modern/Legacy/Vintage, staple/allstar tiers, competitive demand score, cross-format interactions |

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
- Engineer 290 features from the raw card data
- Enrich with MTGGoldfish metagame tournament usage data
- Train an XGBoost regressor with log-transformed prices and sample weights
- Print evaluation metrics (MAE, MedAE, RMSE, R²), per-rarity and per-price-bracket breakdowns
- Show top-25 feature importances
- Save the model to `models/`

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
├── feature_engineer.py    # Raw JSON → 290 numeric features
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

4. **Feature engineering**: Each card is converted to 290 numeric features across 20 groups — from basic stats to metagame demand, supply scarcity, Reserved List status, ban impact, and seasonality signals.

5. **Model**: An XGBoost gradient-boosted tree regressor is trained on log-transformed prices with PseudoHuber loss function (robust to outliers). Sample weights boost rare/mythic and expensive cards. Stratified train/test split ensures fair evaluation.

6. **Inference**: Given a card name, we fetch its data from Scryfall, enrich with reprint info and metagame data, engineer the same 290 features, and run the model. The prediction is automatically logged to both a local JSONL file and the Neon PostgreSQL database.

7. **Automated tracking**: Weekly snapshots of all card prices can run automatically via Windows Task Scheduler (local) or GitHub Actions (cloud), storing data in Neon DB for long-term price trend analysis.

---

## Automation

### Windows Task Scheduler
A scheduled task `MTG_CardPrice_Weekly_Snapshot` runs `weekly_snapshot.py` every Sunday at 10:00 AM, taking a price snapshot and syncing to the database.

### GitHub Actions
A workflow (`.github/workflows/weekly_snapshot.yml`) runs the same snapshot every Sunday at 10:00 UTC without requiring your PC to be on. Uses the `NEON_DATABASE_URL` secret for database access.

---

## Model performance

| Metric | Value |
|---|---|
| **MAE** | €1.10 |
| **Median AE** | €0.05 |
| **RMSE** | €12.68 |
| **R²** | 0.71 |

### Per-rarity MAE

| Rarity | MAE |
|---|---|
| Common | €0.81 |
| Uncommon | €0.74 |
| Rare | €1.58 |
| Mythic | €1.45 |

### Per-price-bracket MAE

| Bracket | MAE |
|---|---|
| €0–€0.50 | €0.11 |
| €0.50–€1 | €0.21 |
| €1–€2 | €0.31 |
| €2–€5 | €0.60 |
| €5–€10 | €1.13 |
| €10–€20 | €2.90 |
| €20–€50 | €6.82 |
| €50+ | €81.27 |

---

## Notes

- **Data sources**: Scryfall (free, includes Cardmarket EUR prices) + MTGGoldfish (free, format staples). No API keys needed.
- **Rate limiting**: Scryfall requires 50–100 ms between requests; MTGGoldfish gets 1.5 s delays. Both are respected.
- **Database**: Optional Neon PostgreSQL stores cards, predictions, and price snapshots for long-term tracking.
- **Price target**: The model predicts `prices.eur` (Cardmarket trend price). For new cards, this approximates the ~2-month settled price.
- **Reserved List**: Cards on the Reserved List are detected via Scryfall's `reserved` field — the model's #1 most important interaction feature.
- **Metagame**: Tournament usage from MTGGoldfish covers the top-50 staples per format. Cards not in the top-50 get 0% usage (not a staple signal).
