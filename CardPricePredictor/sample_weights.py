"""
sample_weights.py — Centralised sample-weight computation (V4).

Replaces the duplicated `_compute_sample_weights` in every model file.
Supports:
  - Rarity-based weights (from config.RARITY_WEIGHTS)
  - Price-based multiplier (>PRICE_WEIGHT_THRESHOLD × PRICE_WEIGHT_FACTOR)
  - Very expensive boost (>€20 × 2.0)
  - **V4**: Configurable hard cap (WEIGHT_CAP) and optional log scaling
"""

import numpy as np
import pandas as pd
import config


def compute_sample_weights(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute per-sample weights that up-weight rare/mythic and expensive cards.

    Weight scheme (applied sequentially):
      1. Rarity weight (config.RARITY_WEIGHTS)
      2. Price > PRICE_WEIGHT_THRESHOLD  →  ×PRICE_WEIGHT_FACTOR
      3. Price > €20                     →  ×2.0
      4. Hard cap at WEIGHT_CAP (default 25)
      5. Optional log scaling: w → 1 + log(w) if WEIGHT_LOG_SCALE is True

    Returns
    -------
    pd.Series of weights aligned to df.index
    """
    weights = pd.Series(1.0, index=df.index)

    # 1. Rarity-based
    for r, w in config.RARITY_WEIGHTS.items():
        col = f"rarity_{r}"
        if col in df.columns:
            mask = df[col] == 1
            weights.loc[mask] = w

    # 2. Price-based boost
    expensive_mask = y > config.PRICE_WEIGHT_THRESHOLD
    weights.loc[expensive_mask] *= config.PRICE_WEIGHT_FACTOR

    # 3. Very expensive extra boost (€20+)
    very_expensive = y > 20.0
    weights.loc[very_expensive] *= 2.0

    # 4. Hard cap  (V4)
    cap = getattr(config, "WEIGHT_CAP", None)
    if cap is not None:
        weights = weights.clip(upper=cap)

    # 5. Optional log scaling  (V4)
    if getattr(config, "WEIGHT_LOG_SCALE", False):
        weights = 1.0 + np.log(weights.clip(lower=1.0))

    return weights
