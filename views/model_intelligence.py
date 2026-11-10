"""PAGE 3 - Model Intelligence (comparison, curves, confusion matrices, tuning, monitoring)."""
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, prediction, ui, visualization as viz

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]


def _why_best(mt: pd.DataFrame, art: dict) -> str:
    best = art["best"]
    s = mt.sort_values("CV PR-AUC", ascending=False).reset_index(drop=True)
    b, second = s.iloc[0], s.iloc[1]
    gap = b["CV PR-AUC"] - second["CV PR-AUC"]
    near = gap < b["CV PR-AUC std"]
    test_best = mt.sort_values("PR-AUC", ascending=False).iloc[0]
    txt = (f"<b>{b['Model']}</b> was selected because it achieved the highest mean 5-fold cross-validated PR-AUC on the <i>training</i> data "
           f"({b['CV PR-AUC']:.3f} +/- {b['CV PR-AUC std']:.3f}). PR-AUC is used because only ~3% of companies are bankrupt, so accuracy and ROC-AUC look flattering "
           f"while PR-AUC reflects how well the rare class is found. ")
    if near:
        txt += (f"Its lead over {second['Model']} ({gap:.3f}) is smaller than the cross-validation standard deviation, so the top models are statistically very close; "
                "the choice is a principled tie-break, not a decisive win. ")
    if test_best["key"] != best:
        txt += (f"On the held-out test set {test_best['Model']} scored a higher PR-AUC ({test_best['PR-AUC']:.3f} vs {mt.loc[mt['key'] == best, 'PR-AUC'].iloc[0]:.3f}). "
                "We report this honestly rather than re-selecting on the test set, which would leak test information into model choice. "
                "With only 44 bankrupt companies in the test set, such differences are noisy. ")
    return txt


def render() -> None:
    art = app_data.artefacts()
    res = art["evaluation"]["results"]
    mt = app_data.metrics_table()
    best = art["best"]
    audit = app_data.eda()
    ui.page_header("Model Intelligence", "Side-by-side comparison of Support Vector Machine, Extra Trees and CatBoost on identical stratified splits.", "Models")
    tabs = st.tabs(["Comparison", "ROC / PR / Calibration", "Confusion matrices", "Tuning & training", "Monitoring"])

    with tabs[0]:
        cols = st.columns(3)
        for c, (_, r) in zip(cols, mt.iterrows()):
            tag = ui.badge("BEST MODEL", cfg.PALETTE["green"]) if r["key"] == best else "&nbsp;"
            rows = "".join(f'<tr><td style="color:var(--muted)">{m}</td><td style="text-align:right;font-weight:700">{r[m]:.3f}</td></tr>' for m in METRICS)
            c.markdown(f"""<div class="kpi" style="--accent:{cfg.MODEL_COLORS[r['key']]}"><div style="min-height:26px">{tag}</div>
                <div class="k-value" style="font-size:1.3rem">{r['Model']}</div>
                <table style="width:100%;margin-top:10px;font-size:.88rem;border-collapse:collapse">{rows}
                <tr><td style="color:var(--muted)">Training time</td><td style="text-align:right;font-weight:700">{r['Train time (s)']:.1f}s</td></tr></table></div>""", unsafe_allow_html=True)
        st.caption("Test-set metrics at each model's decision threshold, chosen by out-of-fold cross-validation on the training set "
                   "(F1-optimal). Default 0.5-threshold metrics are on the *Confusion matrices* tab.")
        ui.section("Interactive comparison")
        c1, c2 = st.columns(2)
        with c1:
            sel = st.multiselect("Metrics to plot", METRICS, default=["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"])
            ui.plot(viz.model_bars(mt, sel or METRICS[:3]), "mi_bars")
        with c2:
            st.markdown("**Radar view**")
            ui.plot(viz.radar(mt, ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]), "mi_radar")
        c3, c4 = st.columns(2)
        with c3:
            ui.plot(viz.single_bar(mt["Short"], mt["Train time (s)"], [cfg.MODEL_COLORS[k] for k in mt["key"]], "Final training time (seconds, full training set)", "{:.1f}s"), "mi_time")
        with c4:
            ui.plot(viz.single_bar(mt["Short"], mt["CV PR-AUC"], [cfg.MODEL_COLORS[k] for k in mt["key"]], "Cross-validated PR-AUC (model-selection metric)", "{:.3f}"), "mi_cv")
        ui.section("Which algorithm performs best, and why?")
        ui.callout(_why_best(mt, art), "ok")
        ui.html_table(mt[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "CV PR-AUC", "Train time (s)"]].round(4),
                      formats={"Train time (s)": lambda v: f"{v:.1f}"})
        ui.callout("<b>Reading the numbers:</b> accuracy is high for every model (~95-96%) because 97% of companies are healthy - a model predicting "
                   "'never bankrupt' would score 96.8%. Recall (share of bankruptcies caught) and precision (share of alerts that are correct) are the meaningful measures here.", "info")

    with tabs[1]:
        prev = audit["split"]["test_bankrupt"] / audit["split"]["test_rows"]
        c1, c2 = st.columns(2)
        with c1:
            ui.plot(viz.roc_fig(res), "mi_roc")
        with c2:
            ui.plot(viz.pr_fig(res, prev), "mi_pr")
        ui.plot(viz.calibration_fig(res), "mi_cal")
        st.caption("Class-weighted models over-estimate absolute probabilities (they are trained as if bankruptcies were common), which is why "
                   "the calibration curve sits below the diagonal. Rankings remain valid; treat probabilities as risk scores, not literal frequencies.")

    with tabs[2]:
        which = st.radio("Decision threshold", ["Tuned (cross-validated)", "Default 0.5"], horizontal=True)
        field = "tuned" if which.startswith("Tuned") else "default_0_5"
        cols = st.columns(3)
        for c, k in zip(cols, res):
            m = res[k]["evaluation"][field]
            with c:
                ui.plot(viz.confusion_fig(m["confusion_matrix"], f"{cfg.MODELS[k]['short']} (threshold {m['threshold']:.2f})"), f"cm_{k}_{field}")
        ui.section("Classification reports")
        rep = st.columns(3)
        for c, k in zip(rep, res):
            cr = pd.DataFrame(res[k]["evaluation"][field]["classification_report"]).T.round(3)
            c.markdown(f"**{cfg.MODELS[k]['name']}**")
            c.dataframe(cr, width="stretch")

    with tabs[3]:
        ui.section("Training and tuning summary")
        rows = []
        for k, r in res.items():
            rows.append({"Model": cfg.MODELS[k]["name"], "Imbalance handling": {"svm": "class_weight='balanced'", "extra_trees": "class_weight (balanced_subsample/balanced)", "catboost": "auto_class_weights='Balanced'"}[k],
                         "Search": f"RandomizedSearchCV, {r['n_iter']} candidates x {cfg.CV_FOLDS}-fold stratified CV, scoring = PR-AUC",
                         "Best hyper-parameters": ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in r["best_params"].items()),
                         "Tuning time": f"{r['tuning_time_sec']:.0f}s", "Final fit time": f"{r['fit_time_sec']:.1f}s", "Threshold": f"{r['threshold']:.3f}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        sp = audit["split"]
        ui.callout(f"All models share one stratified split (random_state={sp['random_state']}): {sp['train_rows']:,} training rows ({sp['train_bankrupt']} bankrupt) and "
                   f"{sp['test_rows']:,} test rows ({sp['test_bankrupt']} bankrupt). Tuning, thresholds and model selection use training data only. "
                   f"Run date: {art['evaluation']['created']}.", "info")

    with tabs[4]:
        ui.section("Threshold monitoring (held-out test set)")
        tp = app_data.test_predictions()
        mk = st.selectbox("Model", list(cfg.MODELS), index=list(cfg.MODELS).index(best), format_func=lambda k: cfg.MODELS[k]["name"], key="mon_model")
        default_t = float(art["pre"]["thresholds"][mk])
        t = st.slider("Decision threshold", 0.02, 0.98, default_t, 0.01)
        y, p = tp["y_true"].to_numpy(), tp[f"proba_{mk}"].to_numpy()
        fig, _ = viz.threshold_sweep(y, p, t)
        c1, c2 = st.columns([1.5, 1])
        with c1:
            ui.plot(fig, "mi_sweep")
        with c2:
            pred = p >= t
            tp_, fp_, fn_, tn_ = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
            prec = tp_ / (tp_ + fp_) if tp_ + fp_ else 0; rec = tp_ / (tp_ + fn_) if tp_ + fn_ else 0
            ui.kpi_row([("Recall", f"{rec * 100:.1f}%", f"{tp_} of {tp_ + fn_} bankruptcies caught", cfg.PALETTE["green"]),
                        ("Precision", f"{prec * 100:.1f}%", f"{fp_} false alarms", cfg.PALETTE["cyan"])], cols=2)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            ui.plot(viz.confusion_fig([[tn_, fp_], [fn_, tp_]], "Live confusion matrix"), "mi_live_cm")
        st.caption("Lowering the threshold catches more bankruptcies (higher recall) at the cost of more false alarms (lower precision). The default marker is the cross-validated F1-optimal value.")
        ui.section("Application usage monitor")
        h = prediction.read_history()
        if h.empty:
            ui.empty_state("No predictions logged yet. Predictions made in the Prediction Studio are stored locally and monitored here.")
        else:
            ui.kpi_row([("Predictions logged", f"{len(h):,}", "since history was created", cfg.PALETTE["blue"]),
                        ("Mean probability", f"{h['Probability'].mean() * 100:.1f}%", "all logged predictions", cfg.PALETTE["cyan"]),
                        ("Flagged bankrupt", f"{(h['Prediction'] == 'Bankrupt').mean() * 100:.1f}%", "share of logged predictions", cfg.PALETTE["red"])])
            c1, c2 = st.columns(2)
            with c1:
                ui.plot(viz.risk_category_bar(h["Risk"].value_counts(), "Logged predictions by risk category"), "mon_risk")
            with c2:
                mc = h["Model"].value_counts()
                ui.plot(viz.single_bar(mc.index, mc.values, [cfg.PALETTE["blue"]] * len(mc), "Logged predictions by model", "{:.0f}"), "mon_models")
            if st.button("Clear prediction history"):
                prediction.clear_history(); st.rerun()
    ui.disclaimer()
