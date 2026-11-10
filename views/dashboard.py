"""PAGE 1 - Executive Dashboard."""
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, prediction, ui, visualization as viz


def render() -> None:
    art = app_data.artefacts()
    audit, ev = app_data.eda(), art["evaluation"]
    mt = app_data.metrics_table()
    best = art["best"]
    brow = mt[mt["key"] == best].iloc[0]
    tp = app_data.test_predictions()
    thr = art["pre"]["thresholds"][best]
    avg_risk = tp[f"proba_{best}"].mean()
    n0, n1 = audit["class_counts"]["0"], audit["class_counts"]["1"]

    ui.page_header("Executive Dashboard", "Portfolio-level view of the dataset, the best-performing model and the overall financial risk picture.",
                   "Overview")
    ui.kpi_row([("Companies analysed", f"{audit['n_rows']:,}", f"{audit['n_features']} raw / {len(art['pre']['features'])} modelled", cfg.PALETTE["blue"]),
                ("Bankrupt", f"{n1:,}", "label = 1", cfg.PALETTE["red"]),
                ("Non-bankrupt", f"{n0:,}", "label = 0", cfg.PALETTE["green"]),
                ("Bankruptcy rate", f"{audit['bankruptcy_rate_pct']:.2f}%", f"imbalance {audit['imbalance_ratio']:.0f} : 1", cfg.PALETTE["amber"]),
                ("Best model", brow["Short"], f"CV PR-AUC {brow['CV PR-AUC']:.3f}", cfg.PALETTE["purple"]),
                ("Avg. risk score", f"{avg_risk * 100:.1f}%", "mean probability, held-out test set", cfg.PALETTE["cyan"])])

    ui.section("Distribution and model performance")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        with ui.card("donut", "Bankruptcy distribution", "Heavily imbalanced target"):
            ui.plot(viz.class_donut(n0, n1), "dash_donut")
    with c2:
        with ui.card("perf", "Model performance comparison", "Held-out test set; decision threshold tuned on training data only"):
            ui.plot(viz.model_bars(mt, ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]), "dash_bars")

    ui.section(f"Financial risk overview - {cfg.MODELS[best]['name']} on the held-out test set")
    y = tp["y_true"].to_numpy(); p = tp[f"proba_{best}"].to_numpy()
    cats = pd.Series([prediction.risk_level(v, thr) for v in p])
    c3, c4 = st.columns([1, 1.3])
    with c3:
        ui.plot(viz.risk_category_bar(cats.value_counts(), "Companies per risk category"), "dash_risk")
    with c4:
        d = pd.DataFrame({"Risk category": cfg.RISK_LEVELS})
        d["Probability band"] = [f"{a * 100:.0f}% - {b * 100:.0f}%" for a, b in (prediction.risk_bands(thr)[k] for k in cfg.RISK_LEVELS)]
        d["Companies"] = [int((cats == k).sum()) for k in cfg.RISK_LEVELS]
        d["Actually bankrupt"] = [int(((cats == k).to_numpy() & (y == 1)).sum()) for k in cfg.RISK_LEVELS]
        d["Observed bankruptcy rate"] = [ (a / c if c else 0.0) for a, c in zip(d["Actually bankrupt"], d["Companies"])]
        d["Risk category"] = [ui.badge(k, cfg.RISK_COLORS[k]) for k in cfg.RISK_LEVELS]
        ui.html_table(d, formats={"Observed bankruptcy rate": lambda v: f"{v * 100:.1f}%", "Companies": lambda v: f"{int(v):,}", "Actually bankrupt": lambda v: f"{int(v):,}"})
        st.caption(f"Bands are anchored to the model's decision threshold ({thr:.2f}); they are a project convention, not an industry standard. "
                   "Observed rates come from real test labels.")

    ui.section("Recent prediction activity")
    hist = prediction.read_history(limit=8)
    if hist.empty:
        ui.empty_state("No predictions have been made yet. Use the <b>Prediction Studio</b> to score a company - real activity will appear here.")
    else:
        hist = hist.assign(Probability=hist["Probability"].map(lambda v: f"{v * 100:.1f}%"),
                           Risk=hist["Risk"].map(lambda k: ui.badge(k, cfg.RISK_COLORS[k])))
        ui.html_table(hist)
    ui.disclaimer()
