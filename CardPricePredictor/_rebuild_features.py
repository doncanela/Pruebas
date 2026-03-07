"""Quick rebuild of features.csv to include _oracle_id (V4)."""
import json, config
from feature_engineer import build_feature_dataframe

print("Loading raw cards …")
cards = json.load(open(config.RAW_DATA_PATH, encoding="utf-8"))
print(f"  {len(cards):,} cards")

print("Building features …")
df = build_feature_dataframe(cards)
print(f"  Shape: {df.shape}")

oracle_cols = [c for c in df.columns if "oracle" in c.lower()]
print(f"  Oracle columns: {oracle_cols}")
print(f"  _oracle_id sample: {df['_oracle_id'].iloc[:3].tolist()}")

print("Saving …")
df.to_csv(config.FEATURES_PATH, index=False)
print(f"  Saved to {config.FEATURES_PATH}")
print("Done.")
