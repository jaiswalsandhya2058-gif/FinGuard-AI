# FinGuard AI - Bankruptcy Prediction & Financial Risk Analytics Platform

A university Machine Learning project built like a fintech SaaS product: it predicts whether a company is **Bankrupt** or **Non-Bankrupt**
from 90 financial indicators, produces a probability and a risk category, compares **SVM, Extra Trees and CatBoost**, and explains the
drivers of each prediction through a Streamlit dashboard.

> FinGuard AI is an educational ML project. Predictions are statistical estimates, **not** financial or investment advice.

## 1. Problem statement
Company failure is costly for lenders, investors and suppliers. Financial ratios contain early warning signals, but bankruptcies are rare
(3.2% of the data), which makes naive models look accurate while missing almost every failing firm. The task is to detect bankruptcies from
financial ratios *honestly* - using imbalance-aware training and metrics suited to rare events.

## 2. Objectives
- Audit and clean the supplied dataset without leakage.
- Train and tune three required algorithms (SVM, Extra Trees, CatBoost) on identical stratified splits.
- Evaluate with Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC/PR curves.
- Explain predictions (SHAP for trees, permutation importance / occlusion for SVM).
- Deliver a working, professional web application with batch prediction and downloadable reports.

## 3. Technologies
Pandas, NumPy, SciPy, Scikit-learn, CatBoost, SHAP, Matplotlib, Seaborn, Plotly, Joblib, Streamlit (custom CSS + HTML components), ReportLab (PDF), SQLite (prediction history).

## 4. Dataset
`data/data.csv` - 6,819 companies x 96 columns; target `Bankrupt?` (0 = 6,599, 1 = 220). 95 numeric financial ratios (profitability, leverage,
liquidity, cash flow, efficiency). No missing values, no duplicate rows. Findings from the audit (`reports/eda_summary.json`):
- `Net Income Flag` is **constant** -> removed.
- 4 perfectly redundant columns (|r| >= 0.9999; e.g. `Debt ratio %` vs `Net worth/Assets`) -> second of each pair removed. **90 features remain.**
- Max single-feature ROC-AUC = 0.889 -> no leakage suspects.
- Many ratios contain values around 1e9-1e10 -> outliers handled per model (see below), no rows deleted.

## 5. Machine-learning workflow
1. Audit -> 2. clean -> 3. stratified 80/20 split (random_state 42) -> 4. tune each model with `RandomizedSearchCV` (5-fold stratified CV, scoring PR-AUC, training data only)
-> 5. choose each model's decision threshold from out-of-fold training predictions (F1-optimal) -> 6. evaluate once on the untouched test set -> 7. select best model by
cross-validated PR-AUC (never by test score) -> 8. SHAP / permutation importance -> 9. save artefacts.

| Model | Preprocessing | Imbalance handling |
|---|---|---|
| SVM (RBF, probability on) | median imputer -> winsorise 1%/99% -> StandardScaler | `class_weight='balanced'` |
| Extra Trees | median imputer (trees are scale/outlier invariant) | `class_weight` balanced / balanced_subsample |
| CatBoost | median imputer (no scaling needed) | `auto_class_weights='Balanced'` |

**Why not SMOTE?** Only 176 positive training rows in 90 dimensions: synthetic interpolation can create unrealistic companies, and it must be applied inside
each CV fold to avoid leakage. Class weights achieve the same re-balancing with no synthetic data. SMOTE was never applied before the split.

## 6. Results (held-out test set: 1,364 companies, 44 bankrupt)
Produced by `python train.py` and stored in `reports/evaluation_results.json`. Threshold = per-model F1-optimal value learned from training CV.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV PR-AUC |
|---|---|---|---|---|---|---|---|
| SVM | 0.951 | 0.338 | 0.545 | 0.417 | 0.946 | 0.366 | 0.396 |
| Extra Trees | 0.962 | 0.435 | 0.614 | 0.509 | 0.948 | 0.524 | 0.438 |
| **CatBoost (selected)** | 0.952 | 0.370 | 0.682 | 0.480 | 0.952 | 0.518 | 0.441 |

CatBoost and Extra Trees are statistically very close (CV difference 0.003 < CV std); CatBoost is chosen by the pre-declared rule (highest CV PR-AUC).
Accuracy is high for all models only because 96.8% of companies are healthy. Precision around 0.35-0.45 means most alerts are false alarms: treat flags as
"review candidates". Numbers will vary slightly if you retrain on another machine/library version.

## 7. Project structure
```
FinGuard_AI/
├── app.py                 # Streamlit entry point (navigation, theme, sidebar)
├── train.py               # one-command training pipeline
├── config.py              # paths, constants, palette
├── requirements.txt
├── data/                  # data.csv, sample_input.csv, input_template.csv
├── models/                # svm_model.pkl, extra_trees_model.pkl, catboost_model.cbm (+catboost_pipeline.pkl),
│                          # best_model.pkl, preprocessing.pkl
├── src/                   # data_preprocessing, model_training, model_evaluation, prediction,
│                          # visualization, explainability, eda_figures, report_generator, ui, app_data
├── views/                 # the 6 application pages (see note)
├── assets/custom.css      # premium dark/light theme
├── reports/               # evaluation_results.json, eda_summary.json, feature_importance.csv, figures/
└── screenshots/
```
*Note:* the spec's `pages/` folder is named `views/` because Streamlit treats a folder called `pages/` specially and would auto-register
it, overriding the custom sidebar navigation.

## 8. Application pages
1. **Executive Dashboard** - KPIs, class distribution, model comparison, risk overview, real recent activity.
2. **Prediction Studio** - manual form (top-12 indicators + all 78 others, validated against dataset ranges, loadable from real held-out companies) and batch CSV; model selector; CSV download; template/sample downloads.
3. **Model Intelligence** - metrics, radar, training time, ROC/PR/calibration, confusion matrices, tuning summary, threshold monitor and usage monitor.
4. **Financial Risk Analytics** - correlation heatmap, importance (SHAP / permutation), top related features, ratio analysis, risk distribution, dataset explorer, EDA gallery.
5. **Company Risk Report** - per-company explanation, observations, limitations; PDF and CSV download.
6. **About & Methodology** - objective, dataset, workflow, algorithms, preprocessing, evaluation, limitations and future scope.

Risk categories are relative to each model's decision threshold *t*: Low < t/2, Moderate < t, High < (1+t)/2, Critical above. This is a project convention, not an industry standard.

## 9. Installation (Windows PowerShell)
```powershell
cd FinGuard_AI
python -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py                      # ~5 minutes; retrains and overwrites models/ and reports/
streamlit run app.py
```
Open http://localhost:8501 . Trained models are already included, so `python train.py` is optional - but run it if you see a pickle/version error.

## 10. Limitations
Single historical dataset; only 220 bankruptcies (44 in test), so metrics are uncertain; moderate precision; ratios only (no macro, sector or time information);
class-weighted probabilities are risk scores rather than calibrated frequencies; explanations show association, not causation.

## 11. Future improvements
Probability calibration; cost-sensitive thresholds; time-aware validation; model stacking; drift monitoring; authentication and a production database.

## Screenshots
See the `screenshots/` folder (dashboard, prediction studio, batch results, model intelligence, analytics, company report, light mode).
