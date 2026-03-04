"""
gui.py — Interactive MTG Card Price Predictor GUI.

Launch with:
    python -m streamlit run gui.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MTG Card Price Predictor",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Header area */
    .block-container { padding-top: 2rem; }

    /* Card image styling */
    .card-img-container img {
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #555;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #1a7a3a;
    }

    /* Dataframe polish */
    .stDataFrame { font-size: 0.92rem; }

    /* Hide Streamlit menu & footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Scryfall API helpers ────────────────────────────────────────────────────

SCRYFALL_DELAY = 0.08  # polite rate-limiting


@st.cache_data(ttl=300, show_spinner=False)
def scryfall_autocomplete(query: str) -> list[str]:
    """Fetch up to 20 autocomplete suggestions from Scryfall."""
    if len(query) < 2:
        return []
    try:
        time.sleep(SCRYFALL_DELAY)
        resp = requests.get(
            "https://api.scryfall.com/cards/autocomplete",
            params={"q": query},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=600, show_spinner=False)
def scryfall_search_printings(card_name: str) -> list[dict]:
    """Get all printings of a card, ordered by release date."""
    try:
        time.sleep(SCRYFALL_DELAY)
        all_data = []
        url = "https://api.scryfall.com/cards/search"
        params = {
            "q": f'!"{card_name}"',
            "unique": "prints",
            "order": "released",
        }
        while url:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            body = resp.json()
            all_data.extend(body.get("data", []))
            if body.get("has_more"):
                url = body.get("next_page")
                params = {}  # next_page is a full URL
                time.sleep(SCRYFALL_DELAY)
            else:
                break
        # Keep only cards that have at least some image
        return [
            c for c in all_data
            if c.get("image_uris") or (c.get("card_faces") and c["card_faces"][0].get("image_uris"))
        ]
    except Exception:
        return []


def get_card_image_url(card: dict) -> str:
    """Extract the normal-size image URL from a Scryfall card object."""
    if "image_uris" in card:
        return card["image_uris"].get("normal", card["image_uris"].get("large", ""))
    faces = card.get("card_faces", [])
    if faces and "image_uris" in faces[0]:
        return faces[0]["image_uris"].get("normal", "")
    return ""


# ─── Prediction engine ───────────────────────────────────────────────────────

BAR_COLOR = "#006747"  # PANTONE 343 C — uniform dark green for all bars

# (label, predictor function name, artifact existence check path)
_REGISTRY = [
    ("XGBoost",       "predict_card",           config.MODEL_PATH),
    ("Random Forest", "predict_card_rf",        config.RF_MODEL_PATH),
    ("TabNet",        "predict_card_tabnet",    config.TABNET_MODEL_DIR),
    ("Lasso",         "predict_card_lasso",     config.LASSO_MODEL_PATH),
    ("Elastic Net",   "predict_card_elasticnet", config.ELASTICNET_MODEL_PATH),
    ("LightGBM",     "predict_card_lgbm",      config.LGBM_MODEL_PATH),
    ("CatBoost",      "predict_card_catboost",  config.CATBOOST_MODEL_PATH),
    ("Two-Stage",     "predict_card_twostage",  config.TWOSTAGE_CLASSIFIER_PATH),
    ("Quantile",      "predict_card_quantile",  config.QUANTILE_MODEL_P50_PATH),
]


def _get_predictor_funcs() -> dict:
    """Lazy-import all predictor functions once."""
    from predictor import (
        predict_card, predict_card_rf, predict_card_tabnet,
        predict_card_lasso, predict_card_elasticnet,
        predict_card_lgbm, predict_card_catboost,
        predict_card_twostage, predict_card_quantile,
    )
    return {
        "predict_card":           predict_card,
        "predict_card_rf":        predict_card_rf,
        "predict_card_tabnet":    predict_card_tabnet,
        "predict_card_lasso":     predict_card_lasso,
        "predict_card_elasticnet": predict_card_elasticnet,
        "predict_card_lgbm":      predict_card_lgbm,
        "predict_card_catboost":  predict_card_catboost,
        "predict_card_twostage":  predict_card_twostage,
        "predict_card_quantile":  predict_card_quantile,
    }


def run_predictions(card_dict: dict) -> list[dict]:
    """Run every trained model on a single card and return results."""
    from predictor import _enrich_single_card

    # Enrich once, then prevent re-enrichment in each predictor
    enriched = _enrich_single_card(dict(card_dict))
    enriched.pop("prints_search_uri", None)

    funcs = _get_predictor_funcs()
    results = []

    for label, fn_name, artifact_path in _REGISTRY:
        if not os.path.exists(artifact_path):
            continue
        try:
            fn = funcs[fn_name]
            r = fn(card_dict=dict(enriched), verbose=False)
            results.append({
                "Model": label,
                "Price": r.get("predicted_price_eur", 0),
                "P10": r.get("predicted_p10"),
                "P90": r.get("predicted_p90"),
            })
        except Exception as e:
            results.append({
                "Model": label,
                "Price": None,
                "Error": str(e),
            })

    return results


# ─── Chart builder ────────────────────────────────────────────────────────────

def build_price_chart(valid: pd.DataFrame, current_price: float | None) -> go.Figure:
    """Build a horizontal bar chart with model predictions."""
    bar_colors = BAR_COLOR

    fig = go.Figure()

    # Main bars
    fig.add_trace(go.Bar(
        y=valid["Model"],
        x=valid["Price"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"€{p:.2f}" for p in valid["Price"]],
        textposition="outside",
        textfont=dict(size=13, weight="bold"),
        name="Predicted Price",
        hovertemplate="%{y}: <b>€%{x:.2f}</b><extra></extra>",
    ))

    # Quantile P10–P90 range line
    q_row = valid[valid["Model"] == "Quantile"]
    if not q_row.empty:
        p10 = q_row.iloc[0].get("P10")
        p90 = q_row.iloc[0].get("P90")
        if p10 is not None and p90 is not None:
            fig.add_trace(go.Scatter(
                x=[p10, p90],
                y=["Quantile", "Quantile"],
                mode="markers+lines",
                marker=dict(size=11, color="#004D35", symbol="diamond"),
                line=dict(color="#004D35", width=3),
                name=f"P10–P90  (€{p10:.2f} – €{p90:.2f})",
                hovertemplate="P10: €%{x:.2f}<extra></extra>",
            ))

    # Current price reference line
    if current_price:
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="#e53935",
            line_width=2,
            annotation_text=f"  Current: €{current_price:.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#e53935", size=13, weight="bold"),
        )

    # Auto-range: ensure bars + text labels fit
    max_price = valid["Price"].max()
    q_p90 = q_row.iloc[0].get("P90") if not q_row.empty else None
    x_max = max(filter(None, [max_price, q_p90, current_price or 0]))
    x_upper = x_max * 1.25  # 25 % headroom for text labels

    fig.update_layout(
        title=dict(text="Predicted Prices by Model", font=dict(size=20)),
        xaxis_title="Predicted Price (€)",
        xaxis=dict(gridcolor="#eee", zeroline=True, range=[0, x_upper]),
        yaxis=dict(title=""),
        height=max(380, len(valid) * 52),
        margin=dict(l=10, r=30, t=55, b=40),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=12),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.30,
    )

    return fig


# ─── Summary table builder ───────────────────────────────────────────────────

def build_summary_table(valid: pd.DataFrame, current_price: float | None) -> pd.DataFrame:
    """Build a sorted summary DataFrame for display."""
    df = valid[["Model", "Price"]].copy()
    df = df.sort_values("Price", ascending=False).reset_index(drop=True)

    if current_price:
        df["Diff (€)"] = df["Price"].apply(
            lambda x: f"{'+'if x - current_price >= 0 else ''}{x - current_price:.2f}"
        )
        df["Diff (%)"] = df["Price"].apply(
            lambda x: f"{'+'if x - current_price >= 0 else ''}{(x / current_price - 1) * 100:.1f}%"
        )

    # Quantile range
    q = valid[valid["Model"] == "Quantile"]
    if not q.empty and q.iloc[0].get("P10") is not None:
        p10, p90 = q.iloc[0]["P10"], q.iloc[0]["P90"]
        df.loc[df["Model"] == "Quantile", "Range (P10–P90)"] = f"€{p10:.2f} – €{p90:.2f}"

    df["Price"] = df["Price"].apply(lambda x: f"€{x:.2f}")
    df = df.rename(columns={"Price": "Predicted"})
    return df


# ═════════════════════════════════════════════════════════════════════════════
#   MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("# 🃏 MTG Card Price Predictor")
st.caption("Search for any Magic: The Gathering card and compare Cardmarket price predictions across 9 models.")

# ─── Search bar (real-time autocomplete) ──────────────────────────────────────

def _search_scryfall(query: str) -> list[str]:
    """Callback for st_searchbox — returns suggestions for each keystroke."""
    if len(query) < 2:
        return []
    return scryfall_autocomplete(query)

selected_name = st_searchbox(
    _search_scryfall,
    placeholder="Type a card name (e.g. Sheoldred, The One Ring, Ragavan…)",
    label="🔍 Search for a card",
    clear_on_submit=False,
    key="card_searchbox",
)

if not selected_name:
    # Show welcome / placeholder
    st.markdown("---")
    st.markdown(
        "#### How to use\n"
        "1. **Type** at least 2 characters above to search for a card.  \n"
        "2. **Select** the card from the autocomplete list.  \n"
        "3. **Pick** an edition — the card image, current price, and all model predictions will appear.  \n"
    )
    st.stop()

# ─── Edition selector ────────────────────────────────────────────────────────

with st.spinner(f"Fetching editions for **{selected_name}** …"):
    printings = scryfall_search_printings(selected_name)

if not printings:
    st.error("Could not find printings for this card.")
    st.stop()

set_options: list[str] = []
for p in printings:
    sname = p.get("set_name", "Unknown")
    scode = p.get("set", "???").upper()
    price = p.get("prices", {}).get("eur")
    tag = f"  — €{price}" if price else ""
    set_options.append(f"{sname} ({scode}){tag}")

selected_idx = st.selectbox(
    "📦 Select edition",
    options=range(len(set_options)),
    format_func=lambda i: set_options[i],
)

card = printings[selected_idx]

st.divider()

# ─── Two-column layout: image + details  |  chart + table ────────────────────

col_img, col_chart = st.columns([1, 2.5], gap="large")

# ── Left column: Card image and details ──
with col_img:
    img_url = get_card_image_url(card)
    if img_url:
        st.image(img_url, use_container_width=True)

    # Card details
    st.markdown(f"### {card.get('name', 'Unknown')}")
    st.markdown(
        f"*{card.get('set_name', '')}* "
        f"(**{card.get('set', '').upper()}**) · "
        f"{card.get('rarity', 'unknown').title()}"
    )
    st.markdown(f"**Type:** {card.get('type_line', '—')}")
    st.markdown(f"**Mana:** {card.get('mana_cost', '—') or '—'}")

    current_raw = card.get("prices", {}).get("eur")
    current_price = float(current_raw) if current_raw else None
    if current_price:
        st.metric("Current Cardmarket Price", f"€{current_price:.2f}")
    else:
        st.metric("Current Cardmarket Price", "N/A")

    # Oracle text
    oracle = card.get("oracle_text") or (
        card.get("card_faces", [{}])[0].get("oracle_text", "")
    )
    if oracle:
        with st.expander("📜 Oracle text", expanded=False):
            st.text(oracle)


# ── Right column: Predictions ──
with col_chart:
    with st.spinner("Running predictions across all models…"):
        results = run_predictions(card)

    if not results:
        st.error("No trained models found. Run training commands first.")
        st.stop()

    df_results = pd.DataFrame(results)
    valid = df_results[df_results["Price"].notna()].copy()

    if valid.empty:
        st.error("All models failed. Check training state.")
        st.stop()

    # Sort ascending so highest bar appears at the top of the chart
    valid = valid.sort_values("Price", ascending=True).reset_index(drop=True)

    # Chart
    fig = build_price_chart(valid, current_price)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("### 📊 Prediction Summary")
    summary = build_summary_table(valid, current_price)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    # Errors
    errors = df_results[df_results["Price"].isna()]
    if not errors.empty:
        with st.expander("⚠️ Model errors"):
            for _, row in errors.iterrows():
                st.warning(f"**{row['Model']}**: {row.get('Error', 'Unknown error')}")

    # Overall stats
    prices = valid["Price"].values
    if len(prices) >= 2:
        mean_pred = np.mean(prices)
        median_pred = np.median(prices)
        std_pred = np.std(prices)
        st.markdown(
            f"**Ensemble stats:** "
            f"Mean = €{mean_pred:.2f} · "
            f"Median = €{median_pred:.2f} · "
            f"Std = €{std_pred:.2f}"
        )
        if current_price:
            consensus = np.median(prices)
            diff = consensus - current_price
            direction = "overvalued" if diff < 0 else "undervalued"
            st.info(
                f"📈 **Median consensus: €{consensus:.2f}** "
                f"({'+'if diff >= 0 else ''}{diff:.2f}, "
                f"{abs(diff) / current_price * 100:.0f}% {direction})"
            )
