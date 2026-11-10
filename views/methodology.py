"""PAGE 6 - About & Methodology."""
import streamlit as st

import config as cfg
from src import app_data, ui


def render() -> None:
    art = app_data.artefacts()
    a, ev = app_data.eda(), art["evaluation"]
    sp = a["split"]
    best = cfg.MODELS[art["best"]]["name"]
    ui.page_header("About & Methodology", "The objective, data, machine-learning workflow and the reasoning behind each decision.", "Documentation")

    ui.section("1. Project objective")
    st.markdown("Predict whether a company is **Bankrupt** or **Non-Bankrupt** from its financial indicators, estimate a bankruptcy probability, "
                "compare three machine-learning algorithms, and explain which indicators drive the risk. The application is an educational prototype of a financial risk-analytics product.")

    ui.section("2. Dataset")
    st.markdown(f"- **Source:** the supplied `data.csv` (company financial ratios, {a['n_rows']:,} companies, {a['n_cols']} columns incl. target `{cfg.TARGET}`).\n"
                f"- **Target balance:** {a['class_counts']['0']:,} non-bankrupt vs {a['class_counts']['1']:,} bankrupt ({a['bankruptcy_rate_pct']:.2f}%) - a {a['imbalance_ratio']:.0f}:1 imbalance.\n"
                f"- **Quality:** {a['missing_total']} missing values, {a['duplicate_rows']} duplicate rows. All {a['n_features']} predictors are numeric financial ratios (profitability, leverage, liquidity, cash flow, efficiency).")

    ui.section("3. Machine-learning workflow")
    steps = ["Audit the data (types, missing, duplicates, constant / redundant features, outliers, leakage screen).",
             f"Clean: drop constant and perfectly redundant features ({sp['train_rows'] + sp['test_rows']:,} rows kept - no row deletions).",
             f"Stratified {int((1 - sp['test_size']) * 100)}/{int(sp['test_size'] * 100)} train-test split (random_state={sp['random_state']}) shared by all models.",
             "Fit every preprocessing step (imputer, winsoriser, scaler) on training folds only.",
             f"Tune each model with RandomizedSearchCV, {cfg.CV_FOLDS}-fold stratified CV, scoring PR-AUC.",
             "Choose each model's decision threshold from out-of-fold training predictions (F1-optimal).",
             "Evaluate once on the untouched test set; select the best model by cross-validated PR-AUC.",
             "Explain with SHAP (trees) and permutation importance (SVM); serve predictions from the saved models."]
    st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)))

    ui.section("4. The three algorithms")
    c = st.columns(3)
    info = [("Support Vector Machine", cfg.MODEL_COLORS["svm"], "Finds the maximum-margin boundary between classes in a kernel-transformed space (RBF kernel). Needs scaled inputs and is sensitive to outliers, so it uses winsorising + StandardScaler. Class imbalance: <code>class_weight='balanced'</code>. Probabilities via Platt scaling."),
            ("Extra Trees Classifier", cfg.MODEL_COLORS["extra_trees"], "An ensemble of many decision trees with randomised split thresholds, which lowers variance. Insensitive to scale and outliers. Imbalance: balanced class weights per tree. Gives native importance and exact SHAP values."),
            ("CatBoost Classifier", cfg.MODEL_COLORS["catboost"], "Gradient boosting with ordered boosting and symmetric trees; builds trees sequentially, each correcting the previous one. Strong on tabular data. Imbalance: <code>auto_class_weights='Balanced'</code>. Native SHAP support.")]
    for col, (n, color, t) in zip(c, info):
        col.markdown(f'<div class="kpi" style="--accent:{color}"><div class="k-value" style="font-size:1.15rem">{n}</div><div class="k-sub" style="margin-top:10px;line-height:1.6">{t}</div></div>', unsafe_allow_html=True)

    ui.section("5. Preprocessing decisions and justification")
    st.markdown(f"- **`{cfg.CONSTANT_FLAG}` is constant** (one unique value), so it cannot help prediction and was removed.\n"
                f"- **{len(a['duplicate_feature_pairs'])} perfectly redundant columns** (|r| >= {cfg.DUPLICATE_FEATURE_CORR}) were removed, keeping one of each pair - they add no information and would split importance scores. "
                f"{a['pairs_corr_ge_095']} other pairs are strongly (not perfectly) correlated and were kept because tree and kernel models tolerate collinearity.\n"
                f"- **Leakage:** the best single feature reaches AUC {a['max_single_feature_auc']:.3f}; nothing is suspiciously predictive. Scaling/clipping statistics come from training data only, and the test set was never used for tuning or selection.\n"
                f"- **Outliers:** {len(a['features_with_extreme_values_1e8'])} features contain values around 1e9-1e10. Rows were *not* deleted (only {sp['train_bankrupt'] + sp['test_bankrupt']} bankrupt examples exist); SVM clips at the 1st/99th percentile, trees ignore magnitude.\n"
                "- **Class imbalance:** handled with class weights. SMOTE was deliberately not used: with ~176 positive training rows in 90 dimensions, synthetic interpolation risks inventing unrealistic companies, and it must be applied inside CV folds to avoid leakage. Class weights give the same re-balancing effect with no synthetic data.")

    ui.section("6. Model evaluation")
    st.markdown("Because 96.8% of companies are healthy, accuracy is misleading. We therefore report Precision, Recall, F1, ROC-AUC and **PR-AUC**, plus confusion matrices, ROC and PR curves. "
                f"The best model is **{best}**, chosen on cross-validated PR-AUC from training data only (see Model Intelligence for the full discussion, including how close the top models are).")
    ui.section("7. Risk categories")
    st.markdown("Probabilities are mapped to Low / Moderate / High / Critical relative to each model's decision threshold *t*: Low < t/2, Moderate < t, High < (1+t)/2, Critical otherwise. "
                "This is a transparent convention of this project, **not** an accepted industry or regulatory standard.")
    ui.section("8. Limitations")
    st.markdown("- Single historical dataset; no guarantee of performance on other countries, industries or years.\n- Only 220 bankrupt examples (44 in the test set), so metrics have wide uncertainty.\n"
                "- Moderate precision: many flagged companies will not go bankrupt.\n- Ratios only - no qualitative, macro-economic or time-series information.\n- Class-weighted probabilities are not calibrated frequencies.\n- Explanations show association, not causation.")
    ui.section("9. Future scope")
    st.markdown("- Probability calibration (isotonic / Platt) and cost-sensitive thresholds.\n- Time-aware validation using multi-year panels; macro-economic features.\n"
                "- Stacking / blending of the three models; additional algorithms (LightGBM, TabNet).\n- Drift monitoring on live data; user authentication and a database back-end.\n- Sector-specific models.")
    ui.disclaimer()
