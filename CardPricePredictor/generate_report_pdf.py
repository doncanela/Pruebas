"""
generate_report_pdf.py — Generate a detailed visual PDF report of model evaluation.

Reads evaluation_report.csv and produces a multi-page PDF with charts and
a conclusion section identifying best and worst models.

Usage:
    python generate_report_pdf.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═════════════════════════════════════════════════════════════════════════════
#   CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

PANTONE_343  = "#006747"
DARK_GREEN   = "#004D35"
LIGHT_GREEN  = "#4CAF50"
ACCENT_RED   = "#D32F2F"
ACCENT_AMBER = "#FF8F00"
ACCENT_BLUE  = "#1565C0"
BG_LIGHT     = "#F5F5F5"
TXT_DARK     = "#212121"
TXT_MID      = "#616161"

MODEL_ORDER = [
    "Random Forest", "Quantile", "XGBoost", "LightGBM",
    "CatBoost", "Two-Stage", "TabNet", "Elastic Net", "Lasso",
]

TIER_COLORS = {
    "Random Forest": PANTONE_343, "Quantile": PANTONE_343, "XGBoost": PANTONE_343,
    "LightGBM": ACCENT_BLUE, "CatBoost": ACCENT_BLUE, "Two-Stage": ACCENT_BLUE,
    "TabNet": ACCENT_AMBER, "Elastic Net": ACCENT_RED, "Lasso": ACCENT_RED,
}

BRACKET_NAMES = [
    "€0–€0.50", "€0.50–€1", "€1–€2", "€2–€5",
    "€5–€10", "€10–€20", "€20–€50", "€50+",
]


def _bar_colors(models):
    return [TIER_COLORS.get(m, TXT_MID) for m in models]


def _add_title_page(pdf, df):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.72, "MTG Card Price Predictor", ha="center", va="center",
            fontsize=36, fontweight="bold", color=PANTONE_343, family="sans-serif")
    ax.text(0.5, 0.62, "Comprehensive Model Evaluation Report", ha="center",
            fontsize=22, color=TXT_DARK, family="sans-serif")
    ax.add_patch(plt.Rectangle((0.2, 0.57), 0.6, 0.003, color=PANTONE_343))

    ax.text(0.5, 0.48, f"Test Set: {int(df['N'].iloc[0]):,} cards  ·  9 Models  ·  8 Price Brackets",
            ha="center", fontsize=14, color=TXT_MID)
    ax.text(0.5, 0.42, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            ha="center", fontsize=12, color=TXT_MID)

    # Legend
    patches = [
        mpatches.Patch(color=PANTONE_343, label="Top Tier (MAE < €1.10)"),
        mpatches.Patch(color=ACCENT_BLUE, label="Mid Tier (MAE €1.10–1.20)"),
        mpatches.Patch(color=ACCENT_AMBER, label="Weak (MAE €1.20–1.60)"),
        mpatches.Patch(color=ACCENT_RED, label="Poor (MAE > €2.00)"),
    ]
    ax.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.28),
              fontsize=11, frameon=True, edgecolor="#CCC", ncol=2)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_global_metrics_table(pdf, df):
    """Page 2: Table of global metrics."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title("Global Metrics Summary", fontsize=20, fontweight="bold",
                 color=PANTONE_343, pad=20, loc="left")

    cols_display = ["Model", "MAE (€)", "MedAE (€)", "RMSE (€)", "R²",
                    "MAPE (%)", "SMAPE (%)", "Within ±25%", "Within ±50%", "Mean Bias (€)"]
    tbl_data = []
    for _, r in df.iterrows():
        tbl_data.append([
            r["Model"],
            f"{r['MAE (€)']:.3f}",
            f"{r['MedAE (€)']:.3f}",
            f"{r['RMSE (€)']:.3f}",
            f"{r['R²']:.4f}",
            f"{r['MAPE (%)']:.1f}%",
            f"{r['SMAPE (%)']:.1f}%",
            f"{r['Within ±25%']:.1f}%",
            f"{r['Within ±50%']:.1f}%",
            f"{r['Mean Bias (€)']:+.3f}",
        ])

    table = ax.table(cellText=tbl_data, colLabels=cols_display, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(cols_display)):
        cell = table[0, j]
        cell.set_facecolor(PANTONE_343)
        cell.set_text_props(color="white", fontweight="bold")

    # Color rows by tier
    for i, (_, r) in enumerate(df.iterrows()):
        color = TIER_COLORS.get(r["Model"], "#EEE")
        for j in range(len(cols_display)):
            cell = table[i + 1, j]
            cell.set_facecolor(color + "18")  # light tint
            cell.set_edgecolor("#CCC")

    # Highlight best values
    for j, col in enumerate(cols_display[1:], 1):
        vals = [float(tbl_data[i][j].replace("%", "").replace("+", "").replace("€", ""))
                for i in range(len(tbl_data))]
        if col in ["R²", "Within ±25%", "Within ±50%"]:
            best_i = np.argmax(vals)
        else:
            best_i = np.argmin(vals)
        table[best_i + 1, j].set_text_props(fontweight="bold", color=PANTONE_343)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_mae_chart(pdf, df):
    """Page 3: Horizontal bar chart — MAE."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    models = df["Model"].values
    maes = df["MAE (€)"].values
    colors = _bar_colors(models)

    bars = ax.barh(range(len(models)), maes, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (€)", fontsize=13)
    ax.set_title("Mean Absolute Error by Model", fontsize=18, fontweight="bold",
                 color=PANTONE_343, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, maes):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"€{val:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, max(maes) * 1.2)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_r2_chart(pdf, df):
    """Page 4: R² bar chart."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    models = df["Model"].values
    r2s = df["R²"].values
    colors = _bar_colors(models)

    bars = ax.barh(range(len(models)), r2s, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("R²", fontsize=13)
    ax.set_title("R² (Coefficient of Determination) by Model", fontsize=18,
                 fontweight="bold", color=PANTONE_343, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, r2s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 0.75)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_smape_mape(pdf, df):
    """Page 5: SMAPE and MAPE side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Percentage Error Metrics", fontsize=18, fontweight="bold",
                 color=PANTONE_343, y=0.96)

    models = df["Model"].values
    colors = _bar_colors(models)

    # SMAPE
    bars1 = ax1.barh(range(len(models)), df["SMAPE (%)"].values, color=colors,
                     edgecolor="white", height=0.65)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("SMAPE (%)")
    ax1.set_title("SMAPE (Symmetric)", fontsize=14, fontweight="bold", color=TXT_DARK)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for bar, val in zip(bars1, df["SMAPE (%)"].values):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    # MAPE
    bars2 = ax2.barh(range(len(models)), df["MAPE (%)"].values, color=colors,
                     edgecolor="white", height=0.65)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("MAPE (%)")
    ax2.set_title("MAPE (Mean Abs %)", fontsize=14, fontweight="bold", color=TXT_DARK)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for bar, val in zip(bars2, df["MAPE (%)"].values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_accuracy_chart(pdf, df):
    """Page 6: Within ±25% and ±50% grouped bars."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    models = df["Model"].values
    x = np.arange(len(models))
    w = 0.35

    bars1 = ax.bar(x - w/2, df["Within ±25%"].values, w, label="Within ±25%",
                   color=PANTONE_343, edgecolor="white")
    bars2 = ax.bar(x + w/2, df["Within ±50%"].values, w, label="Within ±50%",
                   color=ACCENT_BLUE, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% of predictions", fontsize=12)
    ax.set_title("Prediction Accuracy: % Within Tolerance of Actual Price",
                 fontsize=16, fontweight="bold", color=PANTONE_343, pad=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 80)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_bias_chart(pdf, df):
    """Page 7: Mean Bias — diverging bar chart."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    models = df["Model"].values
    bias = df["Mean Bias (€)"].values
    colors = [ACCENT_RED if b > 0 else PANTONE_343 for b in bias]

    bars = ax.barh(range(len(models)), bias, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.invert_yaxis()
    ax.axvline(0, color=TXT_DARK, linewidth=1)
    ax.set_xlabel("Mean Bias (€)", fontsize=13)
    ax.set_title("Prediction Bias (negative = underpredicts, positive = overpredicts)",
                 fontsize=15, fontweight="bold", color=PANTONE_343, pad=15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, bias):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"€{val:+.3f}", va="center", ha=ha, fontsize=11, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_bracket_heatmap(pdf, df):
    """Page 8: Per-bracket MAE heatmap."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    models = df["Model"].values
    data = np.zeros((len(models), len(BRACKET_NAMES)))
    for i, (_, r) in enumerate(df.iterrows()):
        for j, bn in enumerate(BRACKET_NAMES):
            data[i, j] = r.get(f"MAE {bn}", np.nan)

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=20)

    ax.set_xticks(range(len(BRACKET_NAMES)))
    ax.set_xticklabels(BRACKET_NAMES, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(BRACKET_NAMES)):
            val = data[i, j]
            txt_color = "white" if val > 10 else "black"
            ax.text(j, i, f"€{val:.2f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=txt_color)

    ax.set_title("Per-Bracket MAE Heatmap (€)",
                 fontsize=18, fontweight="bold", color=PANTONE_343, pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="MAE (€)")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_bracket_line_chart(pdf, df):
    """Page 9: Per-bracket MAE line chart — how errors scale with price."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    markers = ["o", "s", "^", "D", "v", "P", "X", "h", "*"]
    for idx, (_, r) in enumerate(df.iterrows()):
        model = r["Model"]
        maes = [r.get(f"MAE {bn}", np.nan) for bn in BRACKET_NAMES]
        ax.plot(range(len(BRACKET_NAMES)), maes, marker=markers[idx % len(markers)],
                label=model, color=TIER_COLORS.get(model, TXT_MID),
                linewidth=2, markersize=7, alpha=0.85)

    ax.set_xticks(range(len(BRACKET_NAMES)))
    ax.set_xticklabels(BRACKET_NAMES, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("MAE (€)", fontsize=13)
    ax.set_title("MAE by Price Bracket — Error Scaling", fontsize=18,
                 fontweight="bold", color=PANTONE_343, pad=15)
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_rmse_maxerr(pdf, df):
    """Page 10: RMSE and Max Error side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Error Magnitude Metrics", fontsize=18, fontweight="bold",
                 color=PANTONE_343, y=0.96)

    models = df["Model"].values
    colors = _bar_colors(models)

    # RMSE
    bars1 = ax1.barh(range(len(models)), df["RMSE (€)"].values, color=colors,
                     edgecolor="white", height=0.65)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE (€)")
    ax1.set_title("RMSE", fontsize=14, fontweight="bold", color=TXT_DARK)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    for bar, val in zip(bars1, df["RMSE (€)"].values):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"€{val:.3f}", va="center", fontsize=9, fontweight="bold")

    # Max Error
    bars2 = ax2.barh(range(len(models)), df["Max Error (€)"].values, color=colors,
                     edgecolor="white", height=0.65)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Max Error (€)")
    ax2.set_title("Maximum Single-Card Error", fontsize=14, fontweight="bold", color=TXT_DARK)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for bar, val in zip(bars2, df["Max Error (€)"].values):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"€{val:.1f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_radar_chart(pdf, df):
    """Page 11: Radar/spider chart comparing top 5 models across key metrics."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    # Normalize metrics to 0-1 (1 = best)
    metrics = ["MAE (€)", "RMSE (€)", "SMAPE (%)", "R²", "Within ±25%", "Within ±50%"]
    labels = ["MAE", "RMSE", "SMAPE", "R²", "±25% Acc", "±50% Acc"]
    top5 = df.head(5)

    norm = pd.DataFrame()
    for m in metrics:
        vals = top5[m].values
        if m in ["R²", "Within ±25%", "Within ±50%"]:
            # Higher is better
            norm[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        else:
            # Lower is better — invert
            norm[m] = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=TXT_MID)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    for idx, (_, r) in enumerate(top5.iterrows()):
        values = norm.iloc[idx].values.tolist()
        values += values[:1]
        color = TIER_COLORS.get(r["Model"], TXT_MID)
        ax.plot(angles, values, 'o-', linewidth=2, label=r["Model"], color=color, markersize=5)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_title("Top 5 Models — Radar Comparison (normalized, 1 = best)",
                 fontsize=16, fontweight="bold", color=PANTONE_343, pad=25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=10)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_bracket_counts(pdf, df):
    """Page 12: Stacked bar chart of test-set price distribution."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    # All models have the same counts, take first row
    counts = [int(df.iloc[0].get(f"N {bn}", 0)) for bn in BRACKET_NAMES]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    bracket_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BRACKET_NAMES)))
    bars = ax.bar(range(len(BRACKET_NAMES)), counts, color=bracket_colors, edgecolor="white")

    ax.set_xticks(range(len(BRACKET_NAMES)))
    ax.set_xticklabels(BRACKET_NAMES, fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("Number of cards", fontsize=13)
    ax.set_title("Test Set Price Distribution (15,426 cards)",
                 fontsize=18, fontweight="bold", color=PANTONE_343, pad=15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _add_conclusion(pdf, df):
    """Final page: Conclusion with best/worst analysis."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.95
    def _text(txt, size=11, bold=False, color=TXT_DARK, indent=0):
        nonlocal y
        weight = "bold" if bold else "normal"
        ax.text(0.02 + indent, y, txt, fontsize=size, color=color,
                fontweight=weight, family="sans-serif", va="top",
                transform=ax.transAxes, wrap=True)
        y -= size * 0.0022 + 0.015

    def _line():
        nonlocal y
        ax.add_patch(plt.Rectangle((0.02, y + 0.005), 0.96, 0.002, color=PANTONE_343,
                                   transform=ax.transAxes))
        y -= 0.02

    best = df.iloc[0]
    worst = df.iloc[-1]
    second = df.iloc[1]

    _text("Conclusion & Analysis", size=22, bold=True, color=PANTONE_343)
    _line()
    y -= 0.01

    _text("BEST MODEL: Random Forest", size=16, bold=True, color=PANTONE_343)
    _text(f"Random Forest achieves the best overall performance across all key metrics:", size=11, indent=0.02)
    _text(f"• MAE: €{best['MAE (€)']:.3f} (best) — on average, predictions are off by €1.05", size=10, indent=0.04)
    _text(f"• R²: {best['R²']:.4f} (best) — explains 56.2% of price variance", size=10, indent=0.04)
    _text(f"• SMAPE: {best['SMAPE (%)']:.1f}% (best) — lowest symmetric percentage error", size=10, indent=0.04)
    _text(f"• MedAE: €{best['MedAE (€)']:.3f} — for a typical card, the error is just 13 cents", size=10, indent=0.04)
    _text(f"• Bias: €{best['Mean Bias (€)']:+.3f} — slight conservative underprediction", size=10, indent=0.04)
    y -= 0.01

    _text(f"Quantile (P50) is a very close second (MAE €{second['MAE (€)']:.3f}), and wins on", size=11, indent=0.02)
    _text(f"  ±25% accuracy ({second['Within ±25%']:.1f}% vs {best['Within ±25%']:.1f}%) and ±50% accuracy ({second['Within ±50%']:.1f}% vs {best['Within ±50%']:.1f}%).", size=11, indent=0.02)
    _text(f"  It also provides prediction intervals (P10–P90 coverage: 72.8%).", size=11, indent=0.02)
    y -= 0.01

    _line()
    y -= 0.005

    _text("WORST MODELS: Lasso & Elastic Net", size=16, bold=True, color=ACCENT_RED)
    _text("The two linear models are clearly the weakest performers:", size=11, indent=0.02)
    _text(f"• Lasso — MAE €{worst['MAE (€)']:.3f}, R² {worst['R²']:.4f}, SMAPE {worst['SMAPE (%)']:.1f}%, ±25% accuracy just {worst['Within ±25%']:.1f}%", size=10, indent=0.04)

    en = df[df["Model"] == "Elastic Net"].iloc[0]
    _text(f"• Elastic Net — MAE €{en['MAE (€)']:.3f}, R² {en['R²']:.4f}, SMAPE {en['SMAPE (%)']:.1f}%, ±25% accuracy {en['Within ±25%']:.1f}%", size=10, indent=0.04)
    _text("Both models have MAPE > 300%, meaning their errors are 3x the actual price on average.", size=10, indent=0.04)
    _text("The linear assumption fails: card prices depend on complex, nonlinear interactions", size=10, indent=0.04)
    _text("(rarity × format demand × oracle text synergy) that linear models cannot capture.", size=10, indent=0.04)
    y -= 0.01

    _line()
    y -= 0.005

    _text("TIER SUMMARY", size=16, bold=True, color=PANTONE_343)
    _text("Top Tier (MAE < €1.10):  Random Forest, Quantile, XGBoost", size=11, bold=True, indent=0.02, color=PANTONE_343)
    _text("  → Tree-ensemble models that best capture nonlinear price dynamics.", size=10, indent=0.04)
    _text("Mid Tier (MAE €1.10–1.20):  LightGBM, CatBoost, Two-Stage", size=11, bold=True, indent=0.02, color=ACCENT_BLUE)
    _text("  → Competitive but slightly behind; Two-Stage's bulk classifier adds overhead.", size=10, indent=0.04)
    _text("Weak:  TabNet (MAE €1.53)", size=11, bold=True, indent=0.02, color=ACCENT_AMBER)
    _text("  → Deep learning model underfits; needs more data or tuning.", size=10, indent=0.04)
    _text("Poor:  Elastic Net (€2.19), Lasso (€2.21)", size=11, bold=True, indent=0.02, color=ACCENT_RED)
    _text("  → Linear models are fundamentally unsuitable for this task.", size=10, indent=0.04)

    y -= 0.01
    _line()
    _text("All models struggle with expensive cards (€50+), where MAE ranges €46–€55.", size=10, indent=0.02)
    _text("58.4% of the test set is in the bulk bracket (€0–€0.50), where all tree models excel (MAE ~€0.13).", size=10, indent=0.02)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "evaluation_report.csv")
    if not os.path.exists(csv_path):
        print("ERROR: evaluation_report.csv not found. Run evaluate_all_models.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Sort by MAE (best first)
    df = df.sort_values("MAE (€)").reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), "Model_Evaluation_Report.pdf")
    print(f"Generating PDF report → {out_path}")

    with PdfPages(out_path) as pdf:
        _add_title_page(pdf, df)
        print("  ✔ Title page")
        _add_global_metrics_table(pdf, df)
        print("  ✔ Global metrics table")
        _add_mae_chart(pdf, df)
        print("  ✔ MAE chart")
        _add_r2_chart(pdf, df)
        print("  ✔ R² chart")
        _add_smape_mape(pdf, df)
        print("  ✔ SMAPE / MAPE charts")
        _add_accuracy_chart(pdf, df)
        print("  ✔ Accuracy (±25% / ±50%) chart")
        _add_bias_chart(pdf, df)
        print("  ✔ Bias chart")
        _add_bracket_heatmap(pdf, df)
        print("  ✔ Per-bracket MAE heatmap")
        _add_bracket_line_chart(pdf, df)
        print("  ✔ Per-bracket MAE line chart")
        _add_rmse_maxerr(pdf, df)
        print("  ✔ RMSE / Max Error charts")
        _add_radar_chart(pdf, df)
        print("  ✔ Radar comparison (top 5)")
        _add_bracket_counts(pdf, df)
        print("  ✔ Test set price distribution")
        _add_conclusion(pdf, df)
        print("  ✔ Conclusion page")

    print(f"\n  ✔ Report saved: {out_path} (13 pages)")


if __name__ == "__main__":
    main()
