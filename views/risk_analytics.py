"""PAGE 4 - Financial Risk Analytics."""
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, prediction, ui, visualization as viz

RATIO_GROUPS = {
    "Profitability": ["ROA(A) before interest and % after tax", "Net Income to Total Assets", "Operating Gross Margin", "Net Income to Stockholder's Equity",
                      "Persistent EPS in the Last Four Seasons", "Retained Earnings to Total Assets"],
    "Leverage": ["Debt ratio %", "Borrowing dependency", "Liability to Equity", "Total debt/Total net worth", "Current Liability to Assets", "Interest Coverage Ratio (Interest expense to EBIT)"],
    "Liquidity": ["Current Ratio", "Quick Ratio", "Cash/Current Liability", "Working Capital to Total Assets", "Cash/Total Assets", "Current Liability to Current Assets"],
    "Cash flow": ["CFO to Assets", "Cash Flow to Liability", "Cash Flow to Total Assets", "Cash Flow to Equity", "Cash Flow Per Share"],
    "Efficiency": ["Total Asset Turnover", "Inventory Turnover Rate (times)", "Accounts Receivable Turnover", "Fixed Assets Turnover Frequency", "Total expense/Assets"],
}


def render() -> None:
    art = app_data.artefacts()
    audit, df, imp = app_data.eda(), app_data.dataset(), app_data.importance()
    feats = art["pre"]["features"]
    ui.page_header("Financial Risk Analytics", "Explore which financial indicators drive bankruptcy risk, how the models use them, and how the raw data behaves.", "Analytics")
    tabs = st.tabs(["Correlation", "Feature importance", "Top related features", "Ratio analysis", "Risk distribution", "Dataset explorer", "EDA gallery"])

    with tabs[0]:
        n = st.slider("Number of features (ranked by single-feature predictive power)", 8, 40, 20)
        names = list(audit["single_feature_auc_all"])
        top = [f for f in names if f in feats][:n]
        ui.plot(viz.corr_heatmap(app_data.correlation(tuple(top))), "corr")
        ui.callout(f"<b>{audit['pairs_corr_ge_095']} feature pairs</b> have |r| >= 0.95, so many indicators carry overlapping information (e.g. the three ROA variants). "
                   f"{len(audit['duplicate_feature_pairs'])} perfectly redundant columns (|r| >= {cfg.DUPLICATE_FEATURE_CORR}) were removed before modelling.", "info")

    with tabs[1]:
        c = st.columns([1, 1, 1])
        mk = c[0].selectbox("Model", list(cfg.MODELS), index=list(cfg.MODELS).index(art["best"]), format_func=lambda k: cfg.MODELS[k]["name"], key="imp_model")
        methods = imp.loc[imp["model"] == mk, "method"].unique().tolist()
        method = c[1].selectbox("Method", methods)
        topn = c[2].slider("Top N", 5, 30, 15)
        d = imp[(imp["model"] == mk) & (imp["method"] == method)].nlargest(topn, "importance")
        ui.plot(viz.importance_bar(d, cfg.MODEL_COLORS[mk], f"{cfg.MODELS[mk]['name']} - {method}"), "imp")
        if mk == "svm":
            ui.callout("An RBF-kernel SVM has no native feature importance (it has no coefficients or tree splits). Importance here is measured by "
                       "<b>permutation importance</b>: each feature is shuffled in the test set and the drop in ROC-AUC is recorded. Per-company explanations use occlusion "
                       "(replacing one feature by the training median and measuring the change in probability).", "info")
        else:
            ui.callout("<b>SHAP</b> (SHapley Additive exPlanations) values attribute each prediction to features; Mean |SHAP| averages that over a stratified test sample "
                       "(all test bankruptcies + random healthy companies). Native tree importance is shown for comparison and can differ because it measures split usage rather than impact on the output.", "info")
            img = cfg.FIGURES_DIR / f"shap_summary_{mk}.png"
            if img.exists():
                with st.expander("SHAP summary plot (each dot is one company; colour = feature value)", expanded=True):
                    st.image(str(img), width="stretch")

    with tabs[2]:
        auc = pd.Series(audit["single_feature_auc_all"]).head(15)
        stats = art["pre"]["stats"]
        rows = pd.DataFrame({"Feature": auc.index, "Single-feature AUC": auc.values})
        rows["Non-bankrupt median"] = [stats[f]["healthy_median"] if f in stats else np.nan for f in rows["Feature"]]
        rows["Bankrupt median"] = [stats[f]["bankrupt_median"] if f in stats else np.nan for f in rows["Feature"]]
        c1, c2 = st.columns([1.1, 1])
        with c1:
            ui.plot(viz.single_bar(rows["Feature"].str.slice(0, 40)[::-1], rows["Single-feature AUC"][::-1], [cfg.PALETTE["purple"]] * len(rows), "ROC-AUC of each feature used alone", "{:.3f}", 520, True), "auc_bar")
        with c2:
            st.dataframe(rows.round(4), width="stretch", hide_index=True, height=520)
        ui.callout(f"The strongest single indicator reaches AUC {audit['max_single_feature_auc']:.3f}. No feature exceeds the {cfg.LEAKAGE_AUC_FLAG} leakage-suspicion cut-off, "
                   "so no column looks like a hidden copy of the target. Profitability and retained-earnings ratios dominate: companies that earn little relative to their assets are the ones that fail.", "ok")

    with tabs[3]:
        c = st.columns(2)
        grp = c[0].selectbox("Ratio family", list(RATIO_GROUPS))
        opts = [f for f in RATIO_GROUPS[grp] if f in feats]
        f = c[1].selectbox("Indicator", opts)
        lo, hi = df[f].quantile(.005), df[f].quantile(.995)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            ui.plot(viz.class_box(df, f, (lo, hi)), "ratio_violin")
        with c2:
            g = df.groupby(cfg.TARGET)[f].describe(percentiles=[.25, .5, .75])[["mean", "25%", "50%", "75%"]].rename(index={0: "Non-Bankrupt", 1: "Bankrupt"})
            ui.html_table(g.reset_index().rename(columns={cfg.TARGET: "Class", "mean": "Mean", "25%": "Q1", "50%": "Median", "75%": "Q3"}),
                          formats={c_: (lambda v: f"{v:.4g}") for c_ in ["Mean", "Q1", "Median", "Q3"]})
            st.caption("Violin plots are clipped at the 0.5th-99.5th percentile for readability; statistics use all data.")
            st.markdown("<br>", unsafe_allow_html=True)
            h, b = g.loc["Non-Bankrupt", "50%"], g.loc["Bankrupt", "50%"]
            direction = "lower" if b < h else "higher"
            ui.callout(f"Median <b>{f}</b> is <b>{direction}</b> for bankrupt companies ({b:.4g}) than for healthy ones ({h:.4g}).", "info")

    with tabs[4]:
        mk = st.selectbox("Model", list(cfg.MODELS), index=list(cfg.MODELS).index(art["best"]), format_func=lambda k: cfg.MODELS[k]["name"], key="rd_model")
        tp = app_data.test_predictions()
        thr = art["pre"]["thresholds"][mk]
        p, y = tp[f"proba_{mk}"].to_numpy(), tp["y_true"].to_numpy()
        c1, c2 = st.columns([1.4, 1])
        with c1:
            ui.plot(viz.risk_hist(p, y, thr), "rd_hist")
        with c2:
            cats = pd.Series([prediction.risk_level(v, thr) for v in p])
            ui.plot(viz.risk_category_bar(cats.value_counts(), "Test companies by risk category"), "rd_cat")
        bands = prediction.risk_bands(thr)
        tbl = pd.DataFrame({"Category": cfg.RISK_LEVELS, "Probability band": [f"{bands[k][0] * 100:.0f}% - {bands[k][1] * 100:.0f}%" for k in cfg.RISK_LEVELS],
                            "Companies": [int((cats == k).sum()) for k in cfg.RISK_LEVELS],
                            "Bankrupt": [int(((cats == k).to_numpy() & (y == 1)).sum()) for k in cfg.RISK_LEVELS]})
        tbl["Observed bankruptcy rate"] = tbl["Bankrupt"] / tbl["Companies"].clip(lower=1)
        tbl["Category"] = [ui.badge(k, cfg.RISK_COLORS[k]) for k in cfg.RISK_LEVELS]
        ui.html_table(tbl, formats={"Observed bankruptcy rate": lambda v: f"{v * 100:.1f}%"})
        st.caption(f"Categories are defined relative to the model's decision threshold t = {thr:.2f}: Low < t/2, Moderate < t, High < (1+t)/2, Critical above. This is a project-specific convention.")

    with tabs[5]:
        k = st.columns(4)
        k[0].markdown(ui.kpi_html("Rows", f"{audit['n_rows']:,}", "companies"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi_html("Columns", f"{audit['n_cols']}", "incl. target", cfg.PALETTE["purple"]), unsafe_allow_html=True)
        k[2].markdown(ui.kpi_html("Missing values", f"{audit['missing_total']}", "none found", cfg.PALETTE["green"]), unsafe_allow_html=True)
        k[3].markdown(ui.kpi_html("Duplicate rows", f"{audit['duplicate_rows']}", "none found", cfg.PALETTE["green"]), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c = st.columns([1, 2])
        cls = c[0].radio("Class filter", ["All", "Bankrupt", "Non-Bankrupt"], horizontal=True)
        cols = c[1].multiselect("Columns", list(df.columns), default=[cfg.TARGET] + feats[:5])
        view = df if cls == "All" else df[df[cfg.TARGET] == (1 if cls == "Bankrupt" else 0)]
        st.dataframe(view[cols] if cols else view, width="stretch", height=360)
        st.caption(f"Showing {len(view):,} rows.")
        with st.expander("Descriptive statistics (selected columns)"):
            st.dataframe((view[cols] if cols else view).describe().T, width="stretch")
        with st.expander("Data-quality findings"):
            st.markdown(f"- **Constant feature removed:** `{', '.join(audit['constant_features'])}` has a single value for all companies, so it carries no information.\n"
                        + "".join(f"- **Redundant feature removed:** `{b}` duplicates `{a}` (|r| = {r:.4f}).\n" for a, b, r in audit["duplicate_feature_pairs"])
                        + f"- **Outliers:** on average {audit['mean_outlier_share_pct']:.1f}% of each feature lies outside the 1.5xIQR fences; {len(audit['features_with_extreme_values_1e8'])} features contain values >= 1e8 "
                          "(sentinel-like magnitudes). Rows were kept; SVM clips at the 1st/99th percentile, tree models are unaffected.\n"
                        + f"- **Leakage check:** max single-feature AUC = {audit['max_single_feature_auc']:.3f}; suspects: {audit['leakage_suspects'] or 'none'}.")

    with tabs[6]:
        gal = [("class_distribution.png", "Class distribution"), ("correlation_heatmap.png", "Correlation heatmap (Seaborn)"), ("feature_distributions.png", "Class-conditional distributions"),
               ("outlier_boxplots.png", "Outlier analysis (robust z-scores)"), ("roc_pr_curves.png", "ROC and PR curves (Matplotlib)"), ("confusion_matrices.png", "Confusion matrices")]
        cols = st.columns(2)
        for i, (fn, cap) in enumerate(gal):
            p = cfg.FIGURES_DIR / fn
            if p.exists():
                with cols[i % 2]:
                    st.markdown(f"**{cap}**")
                    st.image(str(p), width="stretch")
    ui.disclaimer()
