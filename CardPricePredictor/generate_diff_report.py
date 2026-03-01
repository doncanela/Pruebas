"""
generate_diff_report.py — Visual report of biggest price differences
from the rare/mythic batch predictions CSV.

Produces a multi-page PDF + individual PNGs in output/report/.
"""

import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ── Settings ────────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)
REPORT_DIR = os.path.join(BASE, "output", "report")
CSV_PATH = os.path.join(BASE, "output", "rare_mythic_predictions.csv")
TOP_N = 30

# Colour palette
C_UNDER  = "#2ecc71"
C_OVER   = "#e74c3c"
C_RARE   = "#3498db"
C_MYTHIC = "#e67e22"
C_GRID   = "#cccccc"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": C_GRID,
})


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["abs_diff"] = df["difference_eur"].abs()
    df["pct_diff"] = (
        (df["difference_eur"] / df["current_price_eur"].clip(lower=0.01)) * 100
    )
    df["label"] = df["card_name"] + "  [" + df["set_code"] + "]"
    return df


def _wrap(labels, width=38):
    return [textwrap.shorten(l, width=width, placeholder="...") for l in labels]


# ═══════════════════════════════════════════════════════════════════════════
# CHART 0 — Summary statistics text panel
# ═══════════════════════════════════════════════════════════════════════════

def chart_summary(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    total = len(df)
    n_rare = (df["rarity"] == "rare").sum()
    n_mythic = (df["rarity"] == "mythic").sum()
    pool = df[df["current_price_eur"] >= 0.50]

    within_10 = (pool["pct_diff"].abs() <= 10).sum()
    within_25 = (pool["pct_diff"].abs() <= 25).sum()
    within_50 = (pool["pct_diff"].abs() <= 50).sum()
    underval = (df["difference_eur"] > 0).sum()
    overval  = (df["difference_eur"] < 0).sum()

    mae = df["abs_diff"].mean()
    medae = df["abs_diff"].median()

    train_range = df[df["current_price_eur"] <= 500]
    mae_train = train_range["abs_diff"].mean()
    medae_train = train_range["abs_diff"].median()

    within_1 = (df["abs_diff"] <= 1.0).sum()
    within_2 = (df["abs_diff"] <= 2.0).sum()
    within_5 = (df["abs_diff"] <= 5.0).sum()

    y = 0.94
    def _title(text, ypos):
        ax.text(0.5, ypos, text, ha="center", fontsize=18, fontweight="bold",
                transform=ax.transAxes, color="#2c3e50")
    def _section(text, ypos):
        ax.text(0.08, ypos, text, ha="left", fontsize=12, fontweight="bold",
                transform=ax.transAxes, color="#34495e")
        return ypos - 0.035
    def _line(label, value, ypos, indent=0.08):
        ax.text(indent, ypos, label, ha="left", fontsize=10,
                transform=ax.transAxes, color="#2c3e50", fontweight="bold")
        ax.text(0.52, ypos, str(value), ha="left", fontsize=10,
                transform=ax.transAxes, color="#333")
        return ypos - 0.033

    _title("RARE & MYTHIC PRICE DIFFERENCE REPORT", y)
    y -= 0.07
    y = _section("DATASET", y); y -= 0.005
    y = _line("Total cards predicted", f"{total:,}", y)
    y = _line("Rare", f"{n_rare:,}", y)
    y = _line("Mythic", f"{n_mythic:,}", y)
    y -= 0.015
    y = _section("PRICES", y); y -= 0.005
    y = _line("Mean actual price", f"\u20ac{df['current_price_eur'].mean():.2f}", y)
    y = _line("Mean predicted price", f"\u20ac{df['predicted_price_eur'].mean():.2f}", y)
    y = _line("Median actual price", f"\u20ac{df['current_price_eur'].median():.2f}", y)
    y = _line("Median predicted price", f"\u20ac{df['predicted_price_eur'].median():.2f}", y)
    y -= 0.015
    y = _section("ERROR METRICS", y); y -= 0.005
    y = _line("Mean Absolute Error (all)", f"\u20ac{mae:.2f}", y)
    y = _line("Median Absolute Error (all)", f"\u20ac{medae:.2f}", y)
    y = _line(f"MAE (cards \u2264 \u20ac500)", f"\u20ac{mae_train:.2f}", y)
    y = _line(f"MedAE (cards \u2264 \u20ac500)", f"\u20ac{medae_train:.2f}", y)
    y -= 0.015
    y = _section("ACCURACY BUCKETS", y); y -= 0.005
    y = _line(f"Within \u20ac1 of actual", f"{within_1:,}  ({100*within_1/total:.1f}%)", y)
    y = _line(f"Within \u20ac2 of actual", f"{within_2:,}  ({100*within_2/total:.1f}%)", y)
    y = _line(f"Within \u20ac5 of actual", f"{within_5:,}  ({100*within_5/total:.1f}%)", y)
    y -= 0.015
    y = _section(f"% ACCURACY (cards \u2265 \u20ac0.50)", y); y -= 0.005
    y = _line(f"Within \u00b110%", f"{within_10:,} / {len(pool):,}  ({100*within_10/len(pool):.1f}%)", y)
    y = _line(f"Within \u00b125%", f"{within_25:,} / {len(pool):,}  ({100*within_25/len(pool):.1f}%)", y)
    y = _line(f"Within \u00b150%", f"{within_50:,} / {len(pool):,}  ({100*within_50/len(pool):.1f}%)", y)
    y -= 0.015
    y = _section("DIRECTION", y); y -= 0.005
    y = _line("Undervalued (pred > actual)", f"{underval:,}  ({100*underval/total:.1f}%)", y)
    y = _line("Overvalued  (pred < actual)", f"{overval:,}  ({100*overval/total:.1f}%)", y)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Top N Undervalued (EUR)
# ═══════════════════════════════════════════════════════════════════════════

def chart_undervalued(df: pd.DataFrame):
    sub = df[df["difference_eur"] > 0].nlargest(TOP_N, "difference_eur").iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 10))
    y = range(len(sub))
    bars = ax.barh(y, sub["difference_eur"], color=C_UNDER, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(sub["label"]), fontsize=8)
    ax.set_xlabel("Predicted \u2212 Actual (\u20ac)")
    ax.set_title(f"Top {TOP_N} Undervalued Cards  (model predicts HIGHER than market)",
                 fontsize=13, fontweight="bold", pad=12)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+\u20ac{row['difference_eur']:.2f}  "
                f"(\u20ac{row['current_price_eur']:.2f} \u2192 \u20ac{row['predicted_price_eur']:.2f})",
                va="center", fontsize=7, color="#333")
    ax.margins(x=0.28)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Top N Overvalued (EUR)
# ═══════════════════════════════════════════════════════════════════════════

def chart_overvalued(df: pd.DataFrame):
    sub = df[df["difference_eur"] < 0].nsmallest(TOP_N, "difference_eur").iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 10))
    y = range(len(sub))
    bars = ax.barh(y, sub["difference_eur"].abs(), color=C_OVER, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(sub["label"]), fontsize=8)
    ax.set_xlabel("|Actual \u2212 Predicted| (\u20ac)")
    ax.set_title(f"Top {TOP_N} Overvalued Cards  (model predicts LOWER than market)",
                 fontsize=13, fontweight="bold", pad=12)

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"\u20ac{row['current_price_eur']:,.0f} \u2192 \u20ac{row['predicted_price_eur']:.0f}",
                va="center", fontsize=7, color="#333")
    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — Top N Undervalued %  (cards >= EUR 1)
# ═══════════════════════════════════════════════════════════════════════════

def chart_undervalued_pct(df: pd.DataFrame):
    pool = df[(df["difference_eur"] > 0) & (df["current_price_eur"] >= 1.0)]
    sub = pool.nlargest(TOP_N, "pct_diff").iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 10))
    y = range(len(sub))
    bars = ax.barh(y, sub["pct_diff"], color="#27ae60", edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(sub["label"]), fontsize=8)
    ax.set_xlabel("% Undervalued  ((pred \u2212 actual) / actual \u00d7 100)")
    ax.set_title(f"Top {TOP_N} Undervalued by %  (cards \u2265 \u20ac1 actual price)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"+{row['pct_diff']:.0f}%  "
                f"(\u20ac{row['current_price_eur']:.2f} \u2192 \u20ac{row['predicted_price_eur']:.2f})",
                va="center", fontsize=7, color="#333")
    ax.margins(x=0.30)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — Top N Overvalued %  (cards >= EUR 1)
# ═══════════════════════════════════════════════════════════════════════════

def chart_overvalued_pct(df: pd.DataFrame):
    pool = df[(df["difference_eur"] < 0) & (df["current_price_eur"] >= 1.0)]
    pool = pool.copy()
    pool["neg_pct"] = pool["pct_diff"].abs()
    sub = pool.nlargest(TOP_N, "neg_pct").iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 10))
    y = range(len(sub))
    bars = ax.barh(y, sub["neg_pct"], color="#c0392b", edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(sub["label"]), fontsize=8)
    ax.set_xlabel("% Overvalued  (|pred \u2212 actual| / actual \u00d7 100)")
    ax.set_title(f"Top {TOP_N} Overvalued by %  (cards \u2265 \u20ac1 actual price)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())

    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"\u2212{row['neg_pct']:.0f}%  "
                f"(\u20ac{row['current_price_eur']:.2f} \u2192 \u20ac{row['predicted_price_eur']:.2f})",
                va="center", fontsize=7, color="#333")
    ax.margins(x=0.30)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — Predicted vs Actual scatter
# ═══════════════════════════════════════════════════════════════════════════

def chart_scatter(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: full range (log)
    ax = axes[0]
    rare = df[df["rarity"] == "rare"]
    mythic = df[df["rarity"] == "mythic"]
    ax.scatter(rare["current_price_eur"], rare["predicted_price_eur"],
               s=3, alpha=0.12, color=C_RARE, label=f"Rare ({len(rare):,})")
    ax.scatter(mythic["current_price_eur"], mythic["predicted_price_eur"],
               s=5, alpha=0.20, color=C_MYTHIC, label=f"Mythic ({len(mythic):,})")
    lim = max(df["current_price_eur"].max(), df["predicted_price_eur"].max()) * 1.1
    ax.plot([0.01, lim], [0.01, lim], "k--", lw=0.8, alpha=0.5, label="Perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.01, lim); ax.set_ylim(0.01, lim)
    ax.set_xlabel("Actual Price (\u20ac, log)")
    ax.set_ylabel("Predicted Price (\u20ac, log)")
    ax.set_title("Predicted vs Actual \u2014 Full Range (log scale)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Right: zoomed <= EUR 50
    ax = axes[1]
    zoom = df[df["current_price_eur"] <= 50]
    rare_z = zoom[zoom["rarity"] == "rare"]
    mythic_z = zoom[zoom["rarity"] == "mythic"]
    ax.scatter(rare_z["current_price_eur"], rare_z["predicted_price_eur"],
               s=3, alpha=0.12, color=C_RARE, label="Rare")
    ax.scatter(mythic_z["current_price_eur"], mythic_z["predicted_price_eur"],
               s=5, alpha=0.20, color=C_MYTHIC, label="Mythic")
    ax.plot([0, 55], [0, 55], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-1, 55); ax.set_ylim(-1, 55)
    ax.set_xlabel("Actual Price (\u20ac)")
    ax.set_ylabel("Predicted Price (\u20ac)")
    ax.set_title("Predicted vs Actual \u2014 Zoomed \u2264 \u20ac50", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Cross-Sectional Price Model \u2014 32,244 Rare & Mythic Cards",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 6 — Difference distribution histogram
# ═══════════════════════════════════════════════════════════════════════════

def chart_difference_dist(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: absolute EUR
    ax = axes[0]
    clipped = df["difference_eur"].clip(-50, 50)
    ax.hist(clipped, bins=150, color="#8e44ad", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="black", lw=1, ls="--")
    med = df["difference_eur"].median()
    ax.axvline(med, color=C_UNDER, lw=1.5, label=f"Median: \u20ac{med:.2f}")
    ax.set_xlabel("Predicted \u2212 Actual (\u20ac)")
    ax.set_ylabel("Number of Cards")
    ax.set_title("EUR Difference Distribution (clipped \u00b1\u20ac50)", fontweight="bold")
    ax.legend(fontsize=9)

    # Right: percentage (cards >= EUR 0.50)
    ax = axes[1]
    pool = df[df["current_price_eur"] >= 0.50]
    pct_clip = pool["pct_diff"].clip(-200, 200)
    ax.hist(pct_clip, bins=150, color="#2980b9", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="black", lw=1, ls="--")
    med_pct = pool["pct_diff"].median()
    ax.axvline(med_pct, color=C_UNDER, lw=1.5, label=f"Median: {med_pct:.1f}%")
    ax.set_xlabel("% Difference")
    ax.set_ylabel("Number of Cards")
    ax.set_title("% Difference Distribution (cards \u2265 \u20ac0.50, clipped \u00b1200%)", fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 7 — Biggest divergence by set (top 15)
# ═══════════════════════════════════════════════════════════════════════════

def chart_by_set(df: pd.DataFrame):
    set_stats = (
        df.groupby("set_name")
        .agg(
            count=("abs_diff", "size"),
            mean_abs_diff=("abs_diff", "mean"),
            mean_actual=("current_price_eur", "mean"),
            mean_pred=("predicted_price_eur", "mean"),
        )
        .query("count >= 20")
        .nlargest(15, "mean_abs_diff")
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    y = range(len(set_stats))
    bars = ax.barh(y, set_stats["mean_abs_diff"], color="#e67e22", edgecolor="white", height=0.72)
    labels = [textwrap.shorten(s, width=45, placeholder="...") for s in set_stats.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |Predicted \u2212 Actual| (\u20ac)")
    ax.set_title("Top 15 Sets with Largest Avg Price Divergence  (\u2265 20 rare/mythic cards)",
                 fontsize=12, fontweight="bold", pad=12)

    for i, (_, row) in enumerate(set_stats.iterrows()):
        ax.text(row["mean_abs_diff"] + 0.3, i,
                f"\u20ac{row['mean_abs_diff']:.1f}  (n={int(row['count'])}, "
                f"avg actual \u20ac{row['mean_actual']:.1f})",
                va="center", fontsize=7, color="#333")
    ax.margins(x=0.25)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 8 — Rarity breakdown (box plots)
# ═══════════════════════════════════════════════════════════════════════════

def chart_rarity_box(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: actual vs predicted
    ax = axes[0]
    ra = df.loc[df["rarity"] == "rare", "current_price_eur"].clip(upper=100)
    rp = df.loc[df["rarity"] == "rare", "predicted_price_eur"].clip(upper=100)
    ma = df.loc[df["rarity"] == "mythic", "current_price_eur"].clip(upper=100)
    mp = df.loc[df["rarity"] == "mythic", "predicted_price_eur"].clip(upper=100)

    bp = ax.boxplot([ra, rp, ma, mp],
                    labels=["Rare\nActual", "Rare\nPred", "Mythic\nActual", "Mythic\nPred"],
                    patch_artist=True, widths=0.6, showfliers=False)
    colors = [C_RARE, "#85c1e9", C_MYTHIC, "#f0b27a"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Price \u20ac (capped at \u20ac100)")
    ax.set_title("Price Distribution by Rarity", fontweight="bold")

    # Right: difference
    ax = axes[1]
    rd = df.loc[df["rarity"] == "rare", "difference_eur"].clip(-20, 20)
    md = df.loc[df["rarity"] == "mythic", "difference_eur"].clip(-20, 20)
    bp2 = ax.boxplot([rd, md], labels=["Rare", "Mythic"],
                     patch_artist=True, widths=0.5, showfliers=False)
    bp2["boxes"][0].set_facecolor(C_RARE);   bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_facecolor(C_MYTHIC); bp2["boxes"][1].set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Predicted \u2212 Actual (\u20ac, clipped \u00b1\u20ac20)")
    ax.set_title("Prediction Error by Rarity", fontweight="bold")

    fig.suptitle("Rarity Breakdown", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 9 — Cumulative error curve
# ═══════════════════════════════════════════════════════════════════════════

def chart_cumulative(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    sorted_err = np.sort(df["abs_diff"].values)
    y = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, y, color="#009688", linewidth=1.8)
    ax.set_xlabel("Absolute Error Threshold (\u20ac)")
    ax.set_ylabel("% of Cards Within Threshold")
    ax.set_title("Cumulative Error Distribution", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 30)

    for pct in [80, 90, 95, 99]:
        ax.axhline(pct, color="#bbb", ls=":", lw=0.7)
        ax.text(29, pct + 0.4, f"{pct}%", fontsize=8, color="#888", ha="right")

    for thr in [0.5, 1.0, 2.0, 5.0, 10.0]:
        pct_within = (df["abs_diff"] <= thr).mean() * 100
        ax.axvline(thr, color="#e74c3c", ls="--", lw=0.5, alpha=0.5)
        ax.text(thr + 0.15, 40, f"\u20ac{thr}\n{pct_within:.0f}%", fontsize=7, color="#c0392b")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 10 — Error by price bracket
# ═══════════════════════════════════════════════════════════════════════════

def chart_brackets(df: pd.DataFrame):
    brackets = [
        ("\u20ac0\u2013\u20ac0.50", 0, 0.50),
        ("\u20ac0.50\u2013\u20ac1", 0.50, 1),
        ("\u20ac1\u2013\u20ac2", 1, 2),
        ("\u20ac2\u2013\u20ac5", 2, 5),
        ("\u20ac5\u2013\u20ac10", 5, 10),
        ("\u20ac10\u2013\u20ac20", 10, 20),
        ("\u20ac20\u2013\u20ac50", 20, 50),
        ("\u20ac50\u2013\u20ac200", 50, 200),
        ("\u20ac200+", 200, 999999),
    ]

    rows = []
    for label, lo, hi in brackets:
        sub = df[(df["current_price_eur"] >= lo) & (df["current_price_eur"] < hi)]
        if len(sub) == 0:
            continue
        rows.append({
            "Bracket": label, "Count": len(sub),
            "MAE": sub["abs_diff"].mean(),
            "MedAE": sub["abs_diff"].median(),
            "Avg Actual": sub["current_price_eur"].mean(),
        })
    bdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    x = range(len(bdf))
    bars = ax.bar(x, bdf["MAE"], color="#2196F3", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(bdf["Bracket"], rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, bdf["MAE"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"\u20ac{val:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("MAE (\u20ac)")
    ax.set_title("Mean Absolute Error by Price Bracket", fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, bdf["Count"], color="#4CAF50", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(bdf["Bracket"], rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, bdf["Count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontsize=8)
    ax.set_ylabel("# Cards")
    ax.set_title("Cards per Price Bracket", fontweight="bold")

    fig.suptitle("Error Analysis by Price Bracket", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} cards loaded.\n")

    charts = [
        ("00_summary",            chart_summary),
        ("01_undervalued_eur",    chart_undervalued),
        ("02_overvalued_eur",     chart_overvalued),
        ("03_undervalued_pct",    chart_undervalued_pct),
        ("04_overvalued_pct",     chart_overvalued_pct),
        ("05_scatter",            chart_scatter),
        ("06_difference_dist",    chart_difference_dist),
        ("07_sets_divergence",    chart_by_set),
        ("08_rarity_breakdown",   chart_rarity_box),
        ("09_cumulative_error",   chart_cumulative),
        ("10_price_brackets",     chart_brackets),
    ]

    pdf_path = os.path.join(REPORT_DIR, "price_difference_report.pdf")

    with PdfPages(pdf_path) as pdf:
        for i, (name, func) in enumerate(charts, 1):
            pct = i / len(charts) * 100
            print(f"  [{i:2d}/{len(charts)}] {pct:5.1f}%  Generating {name}...", flush=True)
            fig = func(df)
            fig.savefig(os.path.join(REPORT_DIR, f"{name}.png"), bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nDone! Report saved to:")
    print(f"  PDF:  {pdf_path}")
    print(f"  PNGs: {len(charts)} individual charts in {REPORT_DIR}")


if __name__ == "__main__":
    main()
