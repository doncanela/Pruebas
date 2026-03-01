# MTG Card Price Predictor

Predict **Cardmarket EUR prices** of Magic: The Gathering cards **~2 months after release** using machine learning (XGBoost).

Data is sourced from the **Scryfall API**, which includes Cardmarket prices alongside comprehensive card metadata.

---

## Features used by the model

| Category | Features |
|---|---|
| **Mana cost** | CMC, per-color pips, generic mana, hybrid/Phyrexian/X indicators |
| **Card types** | Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle |
| **Rarity** | Common, Uncommon, Rare, Mythic (one-hot) |
| **Stats** | Power, Toughness, Loyalty, variable stats flag |
| **Oracle text** | Length, word count, keyword counts (target, draw, destroy, exile, …) |
| **Keywords** | 30 tracked keywords (Flying, Cascade, Storm, …) |
| **Reprint history** | Reprint count, avg/min/max price of prior printings |
| **Set info** | Days since release |
| **Legality** | # of legal formats, Commander/Modern/Standard/Pioneer/Legacy/Vintage/Pauper flags |
| **Misc** | Legendary, modal DFC, adventure, saga, full art, promo, EDHREC rank |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect card data

```bash
# Download recent sets (last 2 years, ~fast)
python main.py collect

# Or download ALL cards (~300 MB, slower)
python main.py collect --mode bulk

# Only sets since a given date
python main.py collect --since 2024-06-01
```

### 3. Train the model

```bash
python main.py train
```

This will:
- Engineer ~90 features from the raw data
- Train an XGBoost regressor with log-transformed prices
- Print evaluation metrics (MAE, RMSE, R²) and per-rarity breakdown
- Save the model to `models/`

### 4. Predict a card's price

```bash
# Single card
python main.py predict "Sheoldred, the Apocalypse"

# Specific printing
python main.py predict "Lightning Bolt" --set lea

# Multiple cards at once
python main.py batch "Ragavan, Nimble Pilferer" "The One Ring" "Orcish Bowmasters"

# Save batch results to JSON
python main.py batch "Force of Will" "Mana Crypt" -o results.json
```

### 5. Check status

```bash
python main.py info
```

---

## Project structure

```
CardPricePredictor/
├── main.py               # CLI entry point
├── config.py             # Paths, API URLs, hyperparameters
├── data_collector.py     # Scryfall API data fetching
├── feature_engineer.py   # Raw JSON → numeric feature matrix
├── model.py              # XGBoost training & evaluation
├── predictor.py          # Single-card inference
├── requirements.txt
├── data/                 # (auto-created) raw data + features CSV
└── models/               # (auto-created) trained model artifacts
```

---

## How it works

1. **Data collection**: The Scryfall API provides every Magic card ever printed, including Cardmarket EUR prices (`prices.eur`). We filter for English paper cards with a known price.

2. **Reprint enrichment**: For each card, we count how many other printings exist and compute the average/min/max price of older printings — a strong signal for predicting new print prices.

3. **Feature engineering**: Each card is converted to ~90 numeric features covering mana cost, type line, oracle text, keywords, rarity, legalities, and more.

4. **Model**: An XGBoost gradient-boosted tree regressor is trained on log-transformed prices (to handle the heavy right skew in MTG card prices). The model is evaluated with train/test split.

5. **Inference**: Given a card name, we fetch its data from Scryfall, engineer the same features, and run the model to predict the Cardmarket EUR price ~2 months after release.

---

## Notes

- **Data source**: Scryfall is free and includes Cardmarket EUR prices. No Cardmarket API key is needed.
- **Rate limiting**: Scryfall asks for 50–100 ms between requests. The collector respects this.
- **Price target**: The model predicts the current `prices.eur` value (Cardmarket trend price). For newly released cards, this approximates the ~2-month settled price if you train on cards that have been out for at least 2 months.
- **Accuracy**: Expect MAE in the range of €0.30–€2.00 depending on dataset size. Mythic rares and chase cards have higher error due to price volatility.
