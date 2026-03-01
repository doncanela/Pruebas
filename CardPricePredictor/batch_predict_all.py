"""Batch predict prices for ALL rare/mythic cards."""
import json, os, sys
import pandas as pd
import numpy as np
import joblib

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))
import config
from feature_engineer import _ensure_metagame_enriched, _card_to_row, get_feature_columns
from model import load_model

def main():
    # 1. Load raw cards, filter to rare/mythic with EUR price
    print("Loading raw cards...")
    with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
        all_cards = json.load(f)

    rm_cards = [
        c for c in all_cards
        if c.get("rarity") in ("rare", "mythic")
        and c.get("prices", {}).get("eur")
    ]
    print(f"Rare/Mythic with EUR price: {len(rm_cards):,}")

    # 2. Enrich with metagame data, then build feature rows + metadata together
    print("Enriching with metagame data...")
    rm_cards = _ensure_metagame_enriched(rm_cards)

    print("Building features...")
    rows = []
    metadata = []  # parallel list to track card identifiers
    is_reserved = []  # track which cards are on the Reserved List
    total = len(rm_cards)
    for i, c in enumerate(rm_cards):
        row = _card_to_row(c)
        if row.get("price_eur") is None or np.isnan(row.get("price_eur", float("nan"))):
            continue  # skip cards with missing target
        rows.append(row)
        metadata.append({
            "card_name": c.get("name", "?"),
            "set_code": c.get("set", "?"),
            "set_name": c.get("set_name", "?"),
            "rarity": c.get("rarity", "?"),
            "current_price_eur": float(c.get("prices", {}).get("eur", 0)),
        })
        is_reserved.append(bool(c.get("reserved", False)))
        if (i + 1) % 5000 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"  Engineering features... {pct:5.1f}% ({i+1:,}/{total:,})", flush=True)

    df = pd.DataFrame(rows)
    is_reserved = np.array(is_reserved)

    # Impute EDHREC rank
    known_mask = df["edhrec_rank"] > 0
    if known_mask.any():
        median_rank = df.loc[known_mask, "edhrec_rank"].median()
        df.loc[~known_mask, "edhrec_rank"] = median_rank
        print(f"EDHREC rank: imputed {(~known_mask).sum():,} missing -> median {median_rank:.0f}")

    # NOTE: No outlier filtering for inference — predict ALL cards
    n_rl = is_reserved.sum()
    n_non_rl = len(df) - n_rl
    print(f"Feature matrix: {len(df):,} rows x {len(df.columns)} cols")
    print(f"  Reserved List: {n_rl:,}  |  Non-RL: {n_non_rl:,}")

    # 3. Predict NON-Reserved-List cards with the main model
    preds_eur = np.zeros(len(df), dtype=np.float64)

    print("\nPredicting non-Reserved-List cards with main model...")
    model, scaler, feature_cols = load_model()
    non_rl_mask = ~is_reserved

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    if non_rl_mask.sum() > 0:
        X_non_rl = df.loc[non_rl_mask, feature_cols].copy()
        X_non_rl_sc = scaler.transform(X_non_rl)
        pred_log = model.predict(X_non_rl_sc)
        preds_non_rl = np.expm1(pred_log)
        preds_non_rl = np.clip(preds_non_rl, 0.01, 5000.0)

        # Expensive-card specialist blend for non-RL cards
        specialist_path = os.path.join(config.MODEL_DIR, "specialist_expensive.joblib")
        if os.path.exists(specialist_path):
            specialist = joblib.load(specialist_path)
            spec_log = specialist.predict(X_non_rl_sc)
            spec_preds = np.clip(np.expm1(spec_log), 0.01, 5000.0)
            exp_mask = preds_non_rl >= 20.0
            preds_non_rl[exp_mask] = 0.6 * spec_preds[exp_mask] + 0.4 * preds_non_rl[exp_mask]
            print(f"  Expensive specialist blended for {exp_mask.sum()} cards")

        preds_eur[non_rl_mask] = preds_non_rl
        print(f"  Main model: {non_rl_mask.sum():,} cards predicted")

    # 4. Predict Reserved List cards with the RL specialist model
    rl_mask = is_reserved
    if rl_mask.sum() > 0 and os.path.exists(config.RL_MODEL_PATH):
        print("\nPredicting Reserved List cards with RL specialist model...")
        from model import load_reserved_list_model
        rl_model, rl_scaler, rl_feature_cols = load_reserved_list_model()

        for col in rl_feature_cols:
            if col not in df.columns:
                df[col] = 0

        X_rl = df.loc[rl_mask, rl_feature_cols].copy()
        X_rl_sc = rl_scaler.transform(X_rl)
        rl_pred_log = rl_model.predict(X_rl_sc)
        preds_rl = np.expm1(rl_pred_log)
        preds_rl = np.clip(preds_rl, 0.01, 50000.0)
        preds_eur[rl_mask] = preds_rl
        print(f"  RL model: {rl_mask.sum():,} cards predicted")
    elif rl_mask.sum() > 0:
        print(f"\nWARNING: No RL model found at {config.RL_MODEL_PATH}")
        print("  Using main model for RL cards as fallback.")
        X_rl = df.loc[rl_mask, feature_cols].copy()
        X_rl_sc = scaler.transform(X_rl)
        rl_pred_log = model.predict(X_rl_sc)
        preds_rl = np.clip(np.expm1(rl_pred_log), 0.01, 50000.0)
        preds_eur[rl_mask] = preds_rl

    # 5. Build results table (metadata is already aligned 1:1 with df rows)
    results = []
    for i, meta in enumerate(metadata):
        results.append({
            **meta,
            "reserved_list": bool(is_reserved[i]),
            "predicted_price_eur": round(float(preds_eur[i]), 2),
            "difference_eur": round(float(preds_eur[i]) - meta["current_price_eur"], 2),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("predicted_price_eur", ascending=False)

    # 8. Save to CSV
    out_path = os.path.join(config.BASE_DIR, "output", "rare_mythic_predictions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df):,} predictions to {out_path}")

    # 9. Print summary
    n_rare = len(results_df[results_df["rarity"] == "rare"])
    n_mythic = len(results_df[results_df["rarity"] == "mythic"])
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total cards predicted: {len(results_df):,}")
    print(f"  Rare:   {n_rare:,}")
    print(f"  Mythic: {n_mythic:,}")
    print(f"\nAvg predicted price: EUR {results_df['predicted_price_eur'].mean():.2f}")
    print(f"Avg actual price:    EUR {results_df['current_price_eur'].mean():.2f}")
    print(f"Avg difference:      EUR {results_df['difference_eur'].mean():.2f}")

    print(f"\n--- Top 20 most valuable (predicted) ---")
    for _, r in results_df.head(20).iterrows():
        print(f"  EUR {r['predicted_price_eur']:7.2f} (actual {r['current_price_eur']:7.2f}) | {r['card_name']} [{r['set_code']}]")

    print(f"\n--- Top 20 biggest undervalued (predicted > actual) ---")
    under = results_df.sort_values("difference_eur", ascending=False).head(20)
    for _, r in under.iterrows():
        print(f"  +EUR {r['difference_eur']:7.2f} | pred {r['predicted_price_eur']:7.2f} vs actual {r['current_price_eur']:7.2f} | {r['card_name']} [{r['set_code']}]")

    print(f"\n--- Top 20 biggest overvalued (predicted < actual) ---")
    over = results_df.sort_values("difference_eur", ascending=True).head(20)
    for _, r in over.iterrows():
        print(f"  EUR {r['difference_eur']:7.2f} | pred {r['predicted_price_eur']:7.2f} vs actual {r['current_price_eur']:7.2f} | {r['card_name']} [{r['set_code']}]")

if __name__ == "__main__":
    main()
