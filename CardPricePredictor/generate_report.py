"""
generate_report.py — Sample 10,000 cards, compare predicted vs actual price,
                     and produce a full PDF report with charts and statistics.

Usage:
    python generate_report.py
    python generate_report.py --n 5000 --output my_report.pdf
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config
from feature_engineer import get_feature_columns

# ─── Constants ────────────────────────────────────────────────────────────────

REPORT_DIR = os.path.join(config.BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_model():
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    feature_cols = joblib.load(config.FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def _sample_and_predict(n: int) -> pd.DataFrame:
    """Load features, sample n cards, predict, return comparison DataFrame."""
    print(f"Loading feature matrix from {config.FEATURES_PATH} …")
    df = pd.read_csv(config.FEATURES_PATH)
    model, scaler, feature_cols = _load_model()

    # Sample
    if n >= len(df):
        print(f"Requested {n:,} but only {len(df):,} cards available — using all.")
        sample = df.copy()
    else:
        sample = df.sample(n=n, random_state=config.RANDOM_STATE).copy()

    print(f"Sampled {len(sample):,} cards.")

    # Predict
    X = sample[feature_cols].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    X_sc = scaler.transform(X)
    pred_log = model.predict(X_sc)
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0.01)

    sample = sample.reset_index(drop=True)
    sample["predicted_eur"] = np.round(pred, 2)
    sample["actual_eur"] = sample["price_eur"]
    sample["error"] = sample["predicted_eur"] - sample["actual_eur"]
    sample["abs_error"] = sample["error"].abs()
    sample["pct_error"] = (
        (sample["error"] / sample["actual_eur"]).replace([np.inf, -np.inf], np.nan) * 100
    )

    # Assign rarity label for grouping
    rarity_map = {
        "rarity_common": "Common",
        "rarity_uncommon": "Uncommon",
        "rarity_rare": "Rare",
        "rarity_mythic": "Mythic",
    }
    sample["rarity"] = "Unknown"
    for col, label in rarity_map.items():
        if col in sample.columns:
            sample.loc[sample[col] == 1, "rarity"] = label

    return sample


# ─── PDF Report ───────────────────────────────────────────────────────────────

def generate_pdf(sample: pd.DataFrame, output_path: str):
    """Build a multi-page PDF report."""
    n = len(sample)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_pages = 10

    def _progress(page: int, desc: str):
        pct = page / total_pages * 100
        print(f"  Generating PDF… {pct:5.1f}% (page {page}/{total_pages}: {desc})", flush=True)

    with PdfPages(output_path) as pdf:

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 1 — Title & Global Statistics
        # ═══════════════════════════════════════════════════════════════════
        _progress(1, "Global Statistics")
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        mae = mean_absolute_error(sample["actual_eur"], sample["predicted_eur"])
        medae = median_absolute_error(sample["actual_eur"], sample["predicted_eur"])
        rmse = np.sqrt(mean_squared_error(sample["actual_eur"], sample["predicted_eur"]))
        r2 = r2_score(sample["actual_eur"], sample["predicted_eur"])
        mape = sample["pct_error"].dropna().abs().mean()
        median_pct = sample["pct_error"].dropna().abs().median()

        within_1 = (sample["abs_error"] <= 1.0).sum()
        within_2 = (sample["abs_error"] <= 2.0).sum()
        within_5 = (sample["abs_error"] <= 5.0).sum()

        title_text = "MTG Card Price Predictor — Validation Report"
        ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="center", va="top")
        ax.text(0.5, 0.90, f"Generated: {now}   |   Cards tested: {n:,}",
                transform=ax.transAxes, fontsize=11, ha="center", va="top", color="gray")

        stats_text = (
            f"═══════════════════════════════════════════════\n"
            f"  GLOBAL METRICS\n"
            f"═══════════════════════════════════════════════\n\n"
            f"  Mean Absolute Error (MAE)    :   €{mae:.4f}\n"
            f"  Median Absolute Error (MedAE):   €{medae:.4f}\n"
            f"  Root Mean Squared Error (RMSE):  €{rmse:.4f}\n"
            f"  R² Score                     :   {r2:.4f}\n\n"
            f"  Mean Abs % Error (MAPE)      :   {mape:.1f}%\n"
            f"  Median Abs % Error           :   {median_pct:.1f}%\n\n"
            f"═══════════════════════════════════════════════\n"
            f"  ACCURACY BUCKETS\n"
            f"═══════════════════════════════════════════════\n\n"
            f"  Within €1 of actual :  {within_1:,}  ({100*within_1/n:.1f}%)\n"
            f"  Within €2 of actual :  {within_2:,}  ({100*within_2/n:.1f}%)\n"
            f"  Within €5 of actual :  {within_5:,}  ({100*within_5/n:.1f}%)\n\n"
            f"═══════════════════════════════════════════════\n"
            f"  PRICE DISTRIBUTION\n"
            f"═══════════════════════════════════════════════\n\n"
            f"  Actual    — mean: €{sample['actual_eur'].mean():.2f}  "
            f"median: €{sample['actual_eur'].median():.2f}  "
            f"max: €{sample['actual_eur'].max():.2f}\n"
            f"  Predicted — mean: €{sample['predicted_eur'].mean():.2f}  "
            f"median: €{sample['predicted_eur'].median():.2f}  "
            f"max: €{sample['predicted_eur'].max():.2f}\n"
        )
        ax.text(0.08, 0.82, stats_text, transform=ax.transAxes,
                fontsize=10, va="top", family="monospace")

        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 2 — Predicted vs Actual Scatter
        # ═══════════════════════════════════════════════════════════════════
        _progress(2, "Scatter Plots")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Full range scatter
        ax1 = axes[0]
        ax1.scatter(sample["actual_eur"], sample["predicted_eur"],
                    alpha=0.15, s=8, color="#2196F3", edgecolors="none")
        max_val = max(sample["actual_eur"].max(), sample["predicted_eur"].max())
        ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
        ax1.set_xlabel("Actual Price (€)")
        ax1.set_ylabel("Predicted Price (€)")
        ax1.set_title("Predicted vs Actual — Full Range")
        ax1.legend()
        ax1.set_xlim(0, min(max_val * 1.05, 500))
        ax1.set_ylim(0, min(max_val * 1.05, 500))

        # Zoomed scatter (0–20€)
        ax2 = axes[1]
        mask = (sample["actual_eur"] <= 20) & (sample["predicted_eur"] <= 20)
        ax2.scatter(sample.loc[mask, "actual_eur"], sample.loc[mask, "predicted_eur"],
                    alpha=0.2, s=8, color="#4CAF50", edgecolors="none")
        ax2.plot([0, 20], [0, 20], "r--", linewidth=1, label="Perfect prediction")
        ax2.set_xlabel("Actual Price (€)")
        ax2.set_ylabel("Predicted Price (€)")
        ax2.set_title("Predicted vs Actual — Zoomed (€0–€20)")
        ax2.legend()
        ax2.set_xlim(0, 20)
        ax2.set_ylim(0, 20)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 3 — Error Distribution
        # ═══════════════════════════════════════════════════════════════════
        _progress(3, "Error Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Absolute error histogram
        ax1 = axes[0]
        clipped_err = sample["abs_error"].clip(upper=20)
        ax1.hist(clipped_err, bins=80, color="#FF9800", edgecolor="white", alpha=0.85)
        ax1.axvline(mae, color="red", linestyle="--", label=f"MAE = €{mae:.2f}")
        ax1.axvline(medae, color="blue", linestyle="--", label=f"MedAE = €{medae:.2f}")
        ax1.set_xlabel("Absolute Error (€)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Absolute Error (clipped at €20)")
        ax1.legend()

        # Percentage error histogram
        ax2 = axes[1]
        pct_clipped = sample["pct_error"].dropna().clip(-200, 200)
        ax2.hist(pct_clipped, bins=80, color="#9C27B0", edgecolor="white", alpha=0.85)
        ax2.axvline(0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Percentage Error (%)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Percentage Error (clipped ±200%)")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 4 — Per-Rarity Breakdown
        # ═══════════════════════════════════════════════════════════════════
        _progress(4, "Per-Rarity Breakdown")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        rarity_order = ["Common", "Uncommon", "Rare", "Mythic"]
        colors = {"Common": "#9E9E9E", "Uncommon": "#C0C0C0", "Rare": "#FFD700", "Mythic": "#FF5722"}

        rarity_stats = []
        for r in rarity_order:
            sub = sample[sample["rarity"] == r]
            if len(sub) == 0:
                continue
            rarity_stats.append({
                "Rarity": r,
                "Count": len(sub),
                "MAE": mean_absolute_error(sub["actual_eur"], sub["predicted_eur"]),
                "MedAE": median_absolute_error(sub["actual_eur"], sub["predicted_eur"]),
                "R²": r2_score(sub["actual_eur"], sub["predicted_eur"]) if len(sub) > 1 else 0,
                "Avg Price": sub["actual_eur"].mean(),
            })

        rarity_df = pd.DataFrame(rarity_stats)

        # Bar chart — MAE per rarity
        ax = axes[0][0]
        bars = ax.bar(rarity_df["Rarity"], rarity_df["MAE"],
                      color=[colors.get(r, "gray") for r in rarity_df["Rarity"]],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rarity_df["MAE"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"€{val:.2f}", ha="center", fontsize=9)
        ax.set_ylabel("MAE (€)")
        ax.set_title("Mean Absolute Error by Rarity")

        # Bar chart — R² per rarity
        ax = axes[0][1]
        bars = ax.bar(rarity_df["Rarity"], rarity_df["R²"],
                      color=[colors.get(r, "gray") for r in rarity_df["Rarity"]],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rarity_df["R²"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.set_ylabel("R²")
        ax.set_title("R² Score by Rarity")

        # Bar chart — count per rarity
        ax = axes[1][0]
        bars = ax.bar(rarity_df["Rarity"], rarity_df["Count"],
                      color=[colors.get(r, "gray") for r in rarity_df["Rarity"]],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rarity_df["Count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{val:,}", ha="center", fontsize=9)
        ax.set_ylabel("# Cards")
        ax.set_title("Cards Tested by Rarity")

        # Bar chart — avg actual price per rarity
        ax = axes[1][1]
        bars = ax.bar(rarity_df["Rarity"], rarity_df["Avg Price"],
                      color=[colors.get(r, "gray") for r in rarity_df["Rarity"]],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rarity_df["Avg Price"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"€{val:.2f}", ha="center", fontsize=9)
        ax.set_ylabel("Avg Actual Price (€)")
        ax.set_title("Average Card Price by Rarity")

        fig.suptitle("Per-Rarity Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 5 — Error by Price Bracket
        # ═══════════════════════════════════════════════════════════════════
        _progress(5, "Price Brackets")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        brackets = [
            ("€0–€0.50", 0, 0.50),
            ("€0.50–€1", 0.50, 1),
            ("€1–€2", 1, 2),
            ("€2–€5", 2, 5),
            ("€5–€10", 5, 10),
            ("€10–€20", 10, 20),
            ("€20–€50", 20, 50),
            ("€50+", 50, 9999),
        ]

        bracket_stats = []
        for label, lo, hi in brackets:
            sub = sample[(sample["actual_eur"] >= lo) & (sample["actual_eur"] < hi)]
            if len(sub) == 0:
                continue
            bracket_stats.append({
                "Bracket": label,
                "Count": len(sub),
                "MAE": mean_absolute_error(sub["actual_eur"], sub["predicted_eur"]),
                "MedAE": median_absolute_error(sub["actual_eur"], sub["predicted_eur"]),
            })

        bdf = pd.DataFrame(bracket_stats)

        ax = axes[0]
        x = range(len(bdf))
        bars = ax.bar(x, bdf["MAE"], color="#2196F3", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(bdf["Bracket"], rotation=35, ha="right", fontsize=8)
        for bar, val in zip(bars, bdf["MAE"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"€{val:.2f}", ha="center", fontsize=8)
        ax.set_ylabel("MAE (€)")
        ax.set_title("MAE by Price Bracket")

        ax = axes[1]
        bars = ax.bar(x, bdf["Count"], color="#4CAF50", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(bdf["Bracket"], rotation=35, ha="right", fontsize=8)
        for bar, val in zip(bars, bdf["Count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{val:,}", ha="center", fontsize=8)
        ax.set_ylabel("# Cards")
        ax.set_title("Cards per Price Bracket")

        fig.suptitle("Error Analysis by Price Bracket", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 6 — Residual plot + QQ-like
        # ═══════════════════════════════════════════════════════════════════
        _progress(6, "Residual Plots")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(sample["actual_eur"], sample["error"],
                   alpha=0.15, s=6, color="#E91E63", edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Actual Price (€)")
        ax.set_ylabel("Error (Predicted − Actual) (€)")
        ax.set_title("Residuals vs Actual Price")
        ax.set_xlim(0, min(sample["actual_eur"].quantile(0.99) * 1.2, 200))
        ax.set_ylim(-30, 30)

        ax = axes[1]
        log_actual = np.log1p(sample["actual_eur"])
        log_pred = np.log1p(sample["predicted_eur"])
        ax.scatter(log_actual, log_pred,
                   alpha=0.15, s=6, color="#3F51B5", edgecolors="none")
        max_log = max(log_actual.max(), log_pred.max())
        ax.plot([0, max_log], [0, max_log], "r--", linewidth=1)
        ax.set_xlabel("log(1 + Actual Price)")
        ax.set_ylabel("log(1 + Predicted Price)")
        ax.set_title("Log-Scale Predicted vs Actual")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 7 — Top 25 Worst Predictions
        # ═══════════════════════════════════════════════════════════════════
        _progress(7, "Worst Predictions")
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        worst = sample.nlargest(25, "abs_error")
        header = f"{'#':<4s} {'Actual':>9s} {'Predicted':>10s} {'Error':>9s}  {'Rarity':<10s}\n"
        header += "─" * 56 + "\n"
        rows_text = ""
        for i, (_, row) in enumerate(worst.iterrows(), 1):
            rows_text += (
                f"{i:<4d} €{row['actual_eur']:>7.2f}  €{row['predicted_eur']:>8.2f}  "
                f"€{row['error']:>+7.2f}   {row['rarity']:<10s}\n"
            )

        ax.text(0.5, 0.96, "Top 25 Worst Predictions (by Absolute Error)",
                transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")
        ax.text(0.1, 0.88, header + rows_text, transform=ax.transAxes,
                fontsize=9, va="top", family="monospace")

        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 8 — Top 25 Best Predictions
        # ═══════════════════════════════════════════════════════════════════
        _progress(8, "Best Predictions")
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        # Best among cards that cost at least €1 (trivial cheap cards don't count)
        interesting = sample[sample["actual_eur"] >= 1.0]
        best = interesting.nsmallest(25, "abs_error")
        header = f"{'#':<4s} {'Actual':>9s} {'Predicted':>10s} {'Error':>9s}  {'Rarity':<10s}\n"
        header += "─" * 56 + "\n"
        rows_text = ""
        for i, (_, row) in enumerate(best.iterrows(), 1):
            rows_text += (
                f"{i:<4d} €{row['actual_eur']:>7.2f}  €{row['predicted_eur']:>8.2f}  "
                f"€{row['error']:>+7.2f}   {row['rarity']:<10s}\n"
            )

        ax.text(0.5, 0.96, "Top 25 Best Predictions (cards ≥ €1, by Absolute Error)",
                transform=ax.transAxes, fontsize=14, fontweight="bold", ha="center", va="top")
        ax.text(0.1, 0.88, header + rows_text, transform=ax.transAxes,
                fontsize=9, va="top", family="monospace")

        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 9 — Cumulative Error Curve
        # ═══════════════════════════════════════════════════════════════════
        _progress(9, "Cumulative Error Curve")
        fig, ax = plt.subplots(figsize=(11, 6))
        sorted_err = np.sort(sample["abs_error"].values)
        y = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        ax.plot(sorted_err, y, color="#009688", linewidth=1.5)
        ax.set_xlabel("Absolute Error Threshold (€)")
        ax.set_ylabel("% of Cards Within Threshold")
        ax.set_title("Cumulative Error Distribution")
        ax.set_xlim(0, 20)
        ax.axhline(90, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(95, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(99, color="gray", linestyle=":", linewidth=0.8)
        ax.text(19, 90.5, "90%", fontsize=8, color="gray")
        ax.text(19, 95.5, "95%", fontsize=8, color="gray")
        ax.text(19, 99.3, "99%", fontsize=8, color="gray")
        ax.grid(True, alpha=0.3)

        # Mark key thresholds
        for threshold in [0.5, 1.0, 2.0, 5.0]:
            pct = (sample["abs_error"] <= threshold).mean() * 100
            ax.axvline(threshold, color="red", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.text(threshold + 0.1, 50, f"€{threshold}\n{pct:.0f}%",
                    fontsize=7, color="red")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ═══════════════════════════════════════════════════════════════════
        # PAGE 10 — Feature Importance (from model)
        # ═══════════════════════════════════════════════════════════════════
        _progress(10, "Feature Importance")
        model, _, feature_cols = _load_model()
        importances = model.feature_importances_
        top_n = 30
        top_idx = np.argsort(importances)[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=(11, 8))
        names = [feature_cols[i] for i in top_idx]
        values = [importances[i] for i in top_idx]
        y_pos = range(len(names))
        ax.barh(y_pos, values, color="#673AB7", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"Top {top_n} Most Important Features")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n✓ PDF report saved to: {output_path}")
    print(f"  Pages: 10  |  Cards: {n:,}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a PDF validation report")
    parser.add_argument("--n", type=int, default=10000, help="Number of cards to sample")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output PDF path")
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_PATH):
        print("ERROR: No trained model found. Run 'python main.py train' first.")
        sys.exit(1)
    if not os.path.exists(config.FEATURES_PATH):
        print("ERROR: No feature matrix found. Run 'python main.py train --rebuild-features' first.")
        sys.exit(1)

    output_path = args.output or os.path.join(
        REPORT_DIR, f"validation_report_{args.n}cards.pdf"
    )

    sample = _sample_and_predict(args.n)
    generate_pdf(sample, output_path)
