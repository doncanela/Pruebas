"""
main.py — CLI entry point for the MTG Card Price Predictor.

Commands:
    collect   – Download card data from Scryfall (includes Cardmarket EUR prices)
    train     – Build features and train the XGBoost model
    predict   – Predict the price of a card
    batch     – Predict prices for a list of cards
    snapshot  – Save current prices for all cards (run periodically)
    history   – Show prediction log and price snapshots for a card
    accuracy  – Check past prediction accuracy vs current prices

Examples:
    python main.py collect --mode sets --since 2024-01-01
    python main.py train
    python main.py predict "Sheoldred, the Apocalypse"
    python main.py predict "Lightning Bolt" --set lea
    python main.py batch "Ragavan, Nimble Pilferer" "The One Ring" "Orcish Bowmasters"
    python main.py snapshot
    python main.py history "Sheoldred, the Apocalypse"
    python main.py accuracy
"""

import argparse
import json
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def cmd_collect(args):
    """Download card data."""
    from data_collector import download_bulk, download_sets

    print("=" * 60)
    print("  MTG Card Price Predictor — Data Collection")
    print("=" * 60)

    if args.mode == "bulk":
        cards = download_bulk(force=args.force)
    else:
        cards = download_sets(released_after=args.since, force=args.force)

    print(f"\nDone. {len(cards):,} cards stored in {config.RAW_DATA_PATH}")

    # Sync card metadata to Neon DB
    if args.sync_db:
        try:
            from db import init_db, upsert_cards_batch
            init_db()
            n = upsert_cards_batch(cards)
            print(f"  ✓ Synced {n:,} cards to Neon DB.")
        except Exception as e:
            print(f"  ⚠ DB sync failed: {e}")


def cmd_train(args):
    """Feature-engineer and train the model."""
    from model import train as train_model, train_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        print(f"  Expected: {config.RAW_DATA_PATH}")
        sys.exit(1)

    # Train main model (automatically excludes Reserved List cards)
    print("\n" + "─" * 60)
    print("  STAGE 1: Main model (non-Reserved List)")
    print("─" * 60)
    train_model(df=df)

    # Train Reserved List specialist model
    print("\n" + "─" * 60)
    print("  STAGE 2: Reserved List specialist model")
    print("─" * 60)
    train_reserved_list(df=df)


def cmd_train_lasso(args):
    """Feature-engineer and train the Lasso model."""
    from model_lasso import train_lasso, train_lasso_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Lasso Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        print(f"  Expected: {config.RAW_DATA_PATH}")
        sys.exit(1)

    # Train main Lasso model (excludes RL cards)
    print("\n" + "─" * 60)
    print("  STAGE 1: Lasso main model (non-Reserved List)")
    print("─" * 60)
    train_lasso(df=df)

    # Train Lasso Reserved List specialist
    print("\n" + "─" * 60)
    print("  STAGE 2: Lasso Reserved List specialist model")
    print("─" * 60)
    train_lasso_reserved_list(df=df)


def cmd_predict_lasso(args):
    """Predict price for a single card using the Lasso model."""
    from predictor import predict_card_lasso

    print("=" * 60)
    print("  MTG Card Price Predictor — Lasso Prediction")
    print("=" * 60)

    if not os.path.exists(config.LASSO_MODEL_PATH):
        print("ERROR: No trained Lasso model found. Run 'train-lasso' first.")
        sys.exit(1)

    predict_card_lasso(card_name=args.name, set_code=args.set)


def cmd_train_rf(args):
    """Feature-engineer and train the Random Forest model."""
    from model_rf import train_rf, train_rf_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Random Forest Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: Random Forest main model (non-Reserved List)")
    print("─" * 60)
    train_rf(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: Random Forest Reserved List specialist model")
    print("─" * 60)
    train_rf_reserved_list(df=df)


def cmd_predict_rf(args):
    """Predict price for a single card using Random Forest."""
    from predictor import predict_card_rf

    print("=" * 60)
    print("  MTG Card Price Predictor — Random Forest Prediction")
    print("=" * 60)

    if not os.path.exists(config.RF_MODEL_PATH):
        print("ERROR: No trained RF model found. Run 'train-rf' first.")
        sys.exit(1)

    predict_card_rf(card_name=args.name, set_code=args.set)


def cmd_train_tabnet(args):
    """Feature-engineer and train the TabNet model."""
    from model_tabnet import train_tabnet, train_tabnet_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — TabNet Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: TabNet main model (non-Reserved List)")
    print("─" * 60)
    train_tabnet(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: TabNet Reserved List specialist model")
    print("─" * 60)
    train_tabnet_reserved_list(df=df)


def cmd_predict_tabnet(args):
    """Predict price for a single card using TabNet."""
    from predictor import predict_card_tabnet

    print("=" * 60)
    print("  MTG Card Price Predictor — TabNet Prediction")
    print("=" * 60)

    if not os.path.exists(config.TABNET_MODEL_DIR):
        print("ERROR: No trained TabNet model found. Run 'train-tabnet' first.")
        sys.exit(1)

    predict_card_tabnet(card_name=args.name, set_code=args.set)


# ─── Elastic Net ─────────────────────────────────────────────────────────────

def cmd_train_elasticnet(args):
    """Feature-engineer and train the Elastic Net (TF-IDF) model."""
    from model_elasticnet import train_elasticnet, train_elasticnet_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Elastic Net Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: Elastic Net main model (non-Reserved List)")
    print("─" * 60)
    train_elasticnet(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: Elastic Net Reserved List specialist model")
    print("─" * 60)
    train_elasticnet_reserved_list(df=df)


def cmd_predict_elasticnet(args):
    """Predict price using Elastic Net."""
    from predictor import predict_card_elasticnet

    print("=" * 60)
    print("  MTG Card Price Predictor — Elastic Net Prediction")
    print("=" * 60)

    if not os.path.exists(config.ELASTICNET_MODEL_PATH):
        print("ERROR: No trained Elastic Net model. Run 'train-elasticnet' first.")
        sys.exit(1)

    predict_card_elasticnet(card_name=args.name, set_code=args.set)


# ─── LightGBM ───────────────────────────────────────────────────────────────

def cmd_train_lgbm(args):
    """Feature-engineer and train the LightGBM model."""
    from model_lgbm import train_lgbm, train_lgbm_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — LightGBM Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: LightGBM main model (non-Reserved List)")
    print("─" * 60)
    train_lgbm(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: LightGBM Reserved List specialist model")
    print("─" * 60)
    train_lgbm_reserved_list(df=df)


def cmd_predict_lgbm(args):
    """Predict price using LightGBM."""
    from predictor import predict_card_lgbm

    print("=" * 60)
    print("  MTG Card Price Predictor — LightGBM Prediction")
    print("=" * 60)

    if not os.path.exists(config.LGBM_MODEL_PATH):
        print("ERROR: No trained LightGBM model. Run 'train-lgbm' first.")
        sys.exit(1)

    predict_card_lgbm(card_name=args.name, set_code=args.set)


# ─── CatBoost ───────────────────────────────────────────────────────────────

def cmd_train_catboost(args):
    """Feature-engineer and train the CatBoost model."""
    from model_catboost import train_catboost, train_catboost_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — CatBoost Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: CatBoost main model (non-Reserved List)")
    print("─" * 60)
    train_catboost(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: CatBoost Reserved List specialist model")
    print("─" * 60)
    train_catboost_reserved_list(df=df)


def cmd_predict_catboost(args):
    """Predict price using CatBoost."""
    from predictor import predict_card_catboost

    print("=" * 60)
    print("  MTG Card Price Predictor — CatBoost Prediction")
    print("=" * 60)

    if not os.path.exists(config.CATBOOST_MODEL_PATH):
        print("ERROR: No trained CatBoost model. Run 'train-catboost' first.")
        sys.exit(1)

    predict_card_catboost(card_name=args.name, set_code=args.set)


# ─── Two-Stage ───────────────────────────────────────────────────────────────

def cmd_train_twostage(args):
    """Feature-engineer and train the Two-Stage bulk/non-bulk model."""
    from model_twostage import train_twostage, train_twostage_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Two-Stage Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: Two-Stage main model (non-Reserved List)")
    print("─" * 60)
    train_twostage(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: Two-Stage Reserved List specialist model")
    print("─" * 60)
    train_twostage_reserved_list(df=df)


def cmd_predict_twostage(args):
    """Predict price using the Two-Stage model."""
    from predictor import predict_card_twostage

    print("=" * 60)
    print("  MTG Card Price Predictor — Two-Stage Prediction")
    print("=" * 60)

    if not os.path.exists(config.TWOSTAGE_CLASSIFIER_PATH):
        print("ERROR: No trained Two-Stage model. Run 'train-twostage' first.")
        sys.exit(1)

    predict_card_twostage(card_name=args.name, set_code=args.set)


# ─── Quantile ────────────────────────────────────────────────────────────────

def cmd_train_quantile(args):
    """Feature-engineer and train the Quantile regression model (P10/P50/P90)."""
    from model_quantile import train_quantile, train_quantile_reserved_list
    import pandas as pd

    print("=" * 60)
    print("  MTG Card Price Predictor — Quantile Model Training")
    print("=" * 60)

    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        print(f"Loading pre-built features from {config.FEATURES_PATH}")
        df = pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.RAW_DATA_PATH):
        print("Building features from raw card data …")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        from feature_engineer import build_feature_dataframe
        df = build_feature_dataframe(cards)
    else:
        print("ERROR: No data found. Run 'collect' first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("  STAGE 1: Quantile main model (non-Reserved List)")
    print("─" * 60)
    train_quantile(df=df)

    print("\n" + "─" * 60)
    print("  STAGE 2: Quantile Reserved List specialist model")
    print("─" * 60)
    train_quantile_reserved_list(df=df)


def cmd_predict_quantile(args):
    """Predict price range using Quantile regression."""
    from predictor import predict_card_quantile

    print("=" * 60)
    print("  MTG Card Price Predictor — Quantile Prediction")
    print("=" * 60)

    if not os.path.exists(config.QUANTILE_MODEL_P50_PATH):
        print("ERROR: No trained Quantile model. Run 'train-quantile' first.")
        sys.exit(1)

    predict_card_quantile(card_name=args.name, set_code=args.set)


def cmd_predict(args):
    """Predict price for a single card using ALL available models."""
    from predictor import predict_card_all

    print("=" * 60)
    print("  MTG Card Price Predictor — All-Model Prediction")
    print("=" * 60)

    # Check that at least one model exists
    any_model = any(os.path.exists(p) for p in [
        config.MODEL_PATH, config.RF_MODEL_PATH,
        config.TABNET_MODEL_DIR, config.LASSO_MODEL_PATH,
        config.ELASTICNET_MODEL_PATH, config.LGBM_MODEL_PATH,
        config.CATBOOST_MODEL_PATH, config.TWOSTAGE_CLASSIFIER_PATH,
        config.QUANTILE_MODEL_P50_PATH,
    ])
    if not any_model:
        print("ERROR: No trained models found. Run a train command first.")
        sys.exit(1)

    predict_card_all(card_name=args.name, set_code=args.set)


def cmd_batch(args):
    """Predict prices for multiple cards."""
    from predictor import predict_batch

    print("=" * 60)
    print("  MTG Card Price Predictor — Batch Prediction")
    print("=" * 60)

    if not os.path.exists(config.MODEL_PATH):
        print("ERROR: No trained model found. Run 'train' first.")
        sys.exit(1)

    results = predict_batch(args.names, set_code=args.set)

    # Print summary table
    print(f"\n{'Card':<40s} {'Rarity':<10s} {'Current':>10s} {'Predicted':>10s}")
    print("─" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['card_name']:<40s} ERROR: {r['error']}")
            continue
        curr = f"€{r['current_price_eur']}" if r['current_price_eur'] else "N/A"
        pred = f"€{r['predicted_price_eur']}"
        print(f"{r['card_name']:<40s} {r['rarity']:<10s} {curr:>10s} {pred:>10s}")

    # Optionally save to JSON
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_snapshot(args):
    """Take a price snapshot for all cards in the database."""
    from price_history import take_snapshot, get_snapshot_dates

    print("=" * 60)
    print("  MTG Card Price Predictor — Price Snapshot")
    print("=" * 60)

    if not os.path.exists(config.RAW_DATA_PATH):
        print("ERROR: No card data found. Run 'collect' first.")
        sys.exit(1)

    n = take_snapshot()
    dates = get_snapshot_dates()
    print(f"\n  ✓ Snapshot saved: {n:,} card prices recorded.")
    print(f"  Total snapshots on file: {len(dates)} date(s)")
    if dates:
        print(f"  Date range: {dates[0]} → {dates[-1]}")


def cmd_history(args):
    """Show prediction log and price snapshots for a card."""
    from price_history import print_card_history

    print("=" * 60)
    print("  MTG Card Price Predictor — Card History")
    print("=" * 60)

    print_card_history(args.name, set_code=args.set)


def cmd_accuracy(args):
    """Check accuracy of past predictions against current prices."""
    from price_history import print_accuracy_report

    print("=" * 60)
    print("  MTG Card Price Predictor — Prediction Accuracy")
    print("=" * 60)

    print_accuracy_report()


def cmd_db_init(args):
    """Create Neon DB tables."""
    from db import init_db

    print("=" * 60)
    print("  MTG Card Price Predictor — Initialize Neon DB")
    print("=" * 60)

    init_db()


def cmd_db_stats(args):
    """Show Neon DB statistics."""
    from db import print_db_stats

    print("=" * 60)
    print("  MTG Card Price Predictor — Neon DB Stats")
    print("=" * 60)

    print_db_stats()


def cmd_sync_db(args):
    """Sync local card data + take a snapshot to Neon DB."""
    from db import init_db, upsert_cards_batch
    from price_history import take_snapshot

    print("=" * 60)
    print("  MTG Card Price Predictor — Sync to Neon DB")
    print("=" * 60)

    init_db()

    # Sync card metadata
    if os.path.exists(config.RAW_DATA_PATH):
        print("\n  Syncing card metadata…")
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            cards = json.load(f)
        n = upsert_cards_batch(cards)
        print(f"  ✓ {n:,} cards synced to Neon DB.\n")

        # Also take a price snapshot
        print("  Taking price snapshot…")
        snap_n = take_snapshot(cards)
        print(f"  ✓ {snap_n:,} prices snapshot saved.")
    else:
        print("  ERROR: No local card data found. Run 'collect' first.")
        sys.exit(1)


def cmd_info(args):
    """Show dataset and model info."""
    print("=" * 60)
    print("  MTG Card Price Predictor — Info")
    print("=" * 60)

    if os.path.exists(config.RAW_DATA_PATH):
        size_mb = os.path.getsize(config.RAW_DATA_PATH) / (1024 * 1024)
        with open(config.RAW_DATA_PATH, "r", encoding="utf-8") as f:
            n = len(json.load(f))
        print(f"  Raw data  : {n:,} cards  ({size_mb:.1f} MB)")
    else:
        print("  Raw data  : not collected yet")

    if os.path.exists(config.FEATURES_PATH):
        import pandas as pd
        df = pd.read_csv(config.FEATURES_PATH)
        print(f"  Features  : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Price range: €{df['price_eur'].min():.2f} – €{df['price_eur'].max():.2f}")
        print(f"  Median     : €{df['price_eur'].median():.2f}")
    else:
        print("  Features  : not built yet")

    if os.path.exists(config.MODEL_PATH):
        print(f"  Model     : {config.MODEL_PATH}")
    else:
        print("  Model     : not trained yet")

    # Neon DB stats
    try:
        from db import print_db_stats
        print_db_stats()
    except Exception as e:
        print(f"  Neon DB   : not available ({e})")

    # Metagame cache
    if os.path.exists(config.METAGAME_CACHE_PATH):
        with open(config.METAGAME_CACHE_PATH, "r", encoding="utf-8") as f:
            meta_cache = json.load(f)
        n_cards = len(meta_cache.get("data", {}))
        fetched = meta_cache.get("fetched_at", "unknown")
        print(f"  Metagame  : {n_cards} cards (fetched {fetched})")
    else:
        print("  Metagame  : not fetched yet (run 'metagame' command)")


def cmd_metagame(args):
    """Fetch or display metagame data from MTGGoldfish."""
    from metagame_collector import fetch_metagame_data, get_card_metagame
    from metagame_collector import METAGAME_FORMATS

    print("=" * 60)
    print("  MTG Card Price Predictor — Metagame Data (MTGGoldfish)")
    print("=" * 60)

    data = fetch_metagame_data(force=args.force)

    if args.card:
        card_data = get_card_metagame(args.card, data)
        if card_data:
            print(f"\nMetagame data for '{args.card}':")
            for fmt, info in sorted(card_data.items()):
                print(f"  {fmt:>10s}: {info['pct']:5.1f}% of decks, "
                      f"{info['copies']:.1f} avg copies")
        else:
            print(f"\n'{args.card}' not in the top-50 staples of any format.")
    else:
        print(f"\n{len(data):,} unique cards with tournament usage data:")
        for fmt in METAGAME_FORMATS:
            cards_in_fmt = {k: v[fmt] for k, v in data.items() if fmt in v}
            top = sorted(cards_in_fmt.items(),
                         key=lambda x: x[1]["pct"], reverse=True)[:5]
            print(f"\n  {fmt.upper()} (top 5):")
            for name, info in top:
                print(f"    {name:<35s} {info['pct']:5.1f}%  ({info['copies']:.1f} copies)")


# ─── Argument parser ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="mtg-price-predictor",
        description="Predict Magic: The Gathering card prices (Cardmarket EUR) "
                    "using machine learning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── collect ──
    p_collect = sub.add_parser("collect", help="Download card data from Scryfall")
    p_collect.add_argument(
        "--mode", choices=["bulk", "sets"], default="sets",
        help="'bulk' downloads every card (~300 MB). "
             "'sets' downloads only recent sets (faster, default).",
    )
    p_collect.add_argument("--force", action="store_true", help="Re-download even if cached")
    p_collect.add_argument(
        "--sync-db", action="store_true", default=True,
        help="Also sync card metadata to Neon DB (default: yes).",
    )
    p_collect.add_argument(
        "--no-sync-db", action="store_false", dest="sync_db",
        help="Skip syncing card metadata to Neon DB.",
    )
    p_collect.add_argument(
        "--since", type=str, default=None,
        help="ISO date (YYYY-MM-DD). Download sets released since this date. "
             "Default: 2 years ago.",
    )
    p_collect.set_defaults(func=cmd_collect)

    # ── train ──
    p_train = sub.add_parser("train", help="Build features & train the XGBoost model")
    p_train.add_argument(
        "--rebuild-features", action="store_true",
        help="Force re-engineering features even if features.csv exists.",
    )
    p_train.set_defaults(func=cmd_train)

    # ── train-lasso ──
    p_train_lasso = sub.add_parser("train-lasso", help="Build features & train the Lasso model")
    p_train_lasso.add_argument(
        "--rebuild-features", action="store_true",
        help="Force re-engineering features even if features.csv exists.",
    )
    p_train_lasso.set_defaults(func=cmd_train_lasso)

    # ── predict-lasso ──
    p_pred_lasso = sub.add_parser("predict-lasso", help="Predict a card's price using Lasso")
    p_pred_lasso.add_argument("name", type=str, help="Card name (English)")
    p_pred_lasso.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_lasso.set_defaults(func=cmd_predict_lasso)

    # ── train-rf ──
    p_train_rf = sub.add_parser("train-rf", help="Build features & train the Random Forest model")
    p_train_rf.add_argument("--rebuild-features", action="store_true")
    p_train_rf.set_defaults(func=cmd_train_rf)

    # ── predict-rf ──
    p_pred_rf = sub.add_parser("predict-rf", help="Predict a card's price using Random Forest")
    p_pred_rf.add_argument("name", type=str, help="Card name (English)")
    p_pred_rf.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_rf.set_defaults(func=cmd_predict_rf)

    # ── train-tabnet ──
    p_train_tabnet = sub.add_parser("train-tabnet", help="Build features & train the TabNet model")
    p_train_tabnet.add_argument("--rebuild-features", action="store_true")
    p_train_tabnet.set_defaults(func=cmd_train_tabnet)

    # ── predict-tabnet ──
    p_pred_tabnet = sub.add_parser("predict-tabnet", help="Predict a card's price using TabNet")
    p_pred_tabnet.add_argument("name", type=str, help="Card name (English)")
    p_pred_tabnet.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_tabnet.set_defaults(func=cmd_predict_tabnet)

    # ── train-elasticnet ──
    p_train_en = sub.add_parser("train-elasticnet", help="Build features & train the Elastic Net model")
    p_train_en.add_argument("--rebuild-features", action="store_true")
    p_train_en.set_defaults(func=cmd_train_elasticnet)

    # ── predict-elasticnet ──
    p_pred_en = sub.add_parser("predict-elasticnet", help="Predict a card's price using Elastic Net")
    p_pred_en.add_argument("name", type=str, help="Card name (English)")
    p_pred_en.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_en.set_defaults(func=cmd_predict_elasticnet)

    # ── train-lgbm ──
    p_train_lgbm = sub.add_parser("train-lgbm", help="Build features & train the LightGBM model")
    p_train_lgbm.add_argument("--rebuild-features", action="store_true")
    p_train_lgbm.set_defaults(func=cmd_train_lgbm)

    # ── predict-lgbm ──
    p_pred_lgbm = sub.add_parser("predict-lgbm", help="Predict a card's price using LightGBM")
    p_pred_lgbm.add_argument("name", type=str, help="Card name (English)")
    p_pred_lgbm.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_lgbm.set_defaults(func=cmd_predict_lgbm)

    # ── train-catboost ──
    p_train_cb = sub.add_parser("train-catboost", help="Build features & train the CatBoost model")
    p_train_cb.add_argument("--rebuild-features", action="store_true")
    p_train_cb.set_defaults(func=cmd_train_catboost)

    # ── predict-catboost ──
    p_pred_cb = sub.add_parser("predict-catboost", help="Predict a card's price using CatBoost")
    p_pred_cb.add_argument("name", type=str, help="Card name (English)")
    p_pred_cb.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_cb.set_defaults(func=cmd_predict_catboost)

    # ── train-twostage ──
    p_train_ts = sub.add_parser("train-twostage", help="Build features & train the Two-Stage model")
    p_train_ts.add_argument("--rebuild-features", action="store_true")
    p_train_ts.set_defaults(func=cmd_train_twostage)

    # ── predict-twostage ──
    p_pred_ts = sub.add_parser("predict-twostage", help="Predict a card's price using Two-Stage model")
    p_pred_ts.add_argument("name", type=str, help="Card name (English)")
    p_pred_ts.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_ts.set_defaults(func=cmd_predict_twostage)

    # ── train-quantile ──
    p_train_q = sub.add_parser("train-quantile", help="Build features & train the Quantile model")
    p_train_q.add_argument("--rebuild-features", action="store_true")
    p_train_q.set_defaults(func=cmd_train_quantile)

    # ── predict-quantile ──
    p_pred_q = sub.add_parser("predict-quantile", help="Predict price range using Quantile model")
    p_pred_q.add_argument("name", type=str, help="Card name (English)")
    p_pred_q.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_pred_q.set_defaults(func=cmd_predict_quantile)

    # ── predict ──
    p_predict = sub.add_parser("predict", help="Predict a card's price")
    p_predict.add_argument("name", type=str, help="Card name (English)")
    p_predict.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_predict.set_defaults(func=cmd_predict)

    # ── batch ──
    p_batch = sub.add_parser("batch", help="Predict prices for multiple cards")
    p_batch.add_argument("names", nargs="+", help="Card names (English)")
    p_batch.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_batch.add_argument("--output", "-o", type=str, default=None, help="Save results to JSON file")
    p_batch.set_defaults(func=cmd_batch)

    # ── info ──
    p_info = sub.add_parser("info", help="Show dataset and model status")
    p_info.set_defaults(func=cmd_info)

    # ── snapshot ──
    p_snap = sub.add_parser(
        "snapshot",
        help="Take a snapshot of current prices for all cards in the database",
    )
    p_snap.set_defaults(func=cmd_snapshot)

    # ── history ──
    p_hist = sub.add_parser("history", help="Show prediction + price history for a card")
    p_hist.add_argument("name", type=str, help="Card name (English)")
    p_hist.add_argument("--set", type=str, default=None, help="3-letter set code")
    p_hist.set_defaults(func=cmd_history)

    # ── accuracy ──
    p_acc = sub.add_parser(
        "accuracy",
        help="Check accuracy of past predictions vs current Cardmarket prices",
    )
    p_acc.set_defaults(func=cmd_accuracy)

    # ── db-init ──
    p_dbinit = sub.add_parser("db-init", help="Initialize Neon DB tables")
    p_dbinit.set_defaults(func=cmd_db_init)

    # ── db-stats ──
    p_dbstats = sub.add_parser("db-stats", help="Show Neon DB statistics")
    p_dbstats.set_defaults(func=cmd_db_stats)

    # ── sync-db ──
    p_syncdb = sub.add_parser(
        "sync-db",
        help="Sync local card data to Neon DB (cards + snapshot)",
    )
    p_syncdb.set_defaults(func=cmd_sync_db)

    # ── metagame ──
    p_meta = sub.add_parser(
        "metagame",
        help="Fetch/refresh metagame data from MTGGoldfish",
    )
    p_meta.add_argument(
        "--force", action="store_true",
        help="Ignore cache, re-fetch from MTGGoldfish.",
    )
    p_meta.add_argument(
        "--card", type=str, default=None,
        help="Look up metagame data for a specific card.",
    )
    p_meta.set_defaults(func=cmd_metagame)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
