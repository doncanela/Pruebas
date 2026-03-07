"""
retrain_all_v4.py — Retrain ALL models with V4 weight capping.

Run sequentially to avoid memory issues.  Skip feature rebuild (already done).
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import config

print("Loading raw cards …")
cards = json.load(open(config.RAW_DATA_PATH, encoding="utf-8"))
print(f"  {len(cards):,} cards\n")

MODELS = [
    ("XGBoost",        "model",            "train"),
    ("Random Forest",  "model_rf",         "train_rf"),
    ("LightGBM",       "model_lgbm",       "train_lgbm"),
    ("Lasso",          "model_lasso",       "train_lasso"),
    ("Quantile",       "model_quantile",    "train_quantile"),
    ("Two-Stage",      "model_twostage",    "train_twostage"),
    ("CatBoost",       "model_catboost",    "train_catboost"),
    ("Elastic Net",    "model_elasticnet",  "train_elasticnet"),
    ("TabNet",         "model_tabnet",      "train_tabnet"),
    ("High-End",       "model_highend",     "train_highend"),
]

t0 = time.time()
results = {}
for name, module_name, func_name in MODELS:
    print("=" * 70)
    print(f"  TRAINING: {name}")
    print("=" * 70)
    t1 = time.time()
    try:
        mod = __import__(module_name, fromlist=[func_name])
        train_fn = getattr(mod, func_name)
        train_fn(cards=cards)
        elapsed = time.time() - t1
        results[name] = f"OK ({elapsed:.0f}s)"
        print(f"\n  ✓ {name} done in {elapsed:.0f}s\n")
    except Exception as e:
        elapsed = time.time() - t1
        results[name] = f"FAILED ({elapsed:.0f}s): {e}"
        print(f"\n  ✗ {name} FAILED after {elapsed:.0f}s: {e}\n")
        import traceback
        traceback.print_exc()

total = time.time() - t0
print("\n" + "=" * 70)
print("  RETRAINING SUMMARY (V4)")
print("=" * 70)
for name, status in results.items():
    print(f"  {name:15s}  {status}")
print(f"\n  Total time: {total/60:.1f} minutes")
