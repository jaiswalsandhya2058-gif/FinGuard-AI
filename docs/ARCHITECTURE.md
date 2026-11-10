# FinGuard AI - System Architecture

## 1. High-level overview
```mermaid
flowchart LR
    A[data/data.csv<br/>6,819 companies] --> B[Data audit + cleaning<br/>src/data_preprocessing.py]
    B --> C[Stratified 80/20 split]
    C --> D[Train + tune 3 models<br/>src/model_training.py]
    D --> E[Evaluate on test set<br/>src/model_evaluation.py]
    D --> F[Explainability<br/>src/explainability.py]
    E --> G[(models/ + reports/)]
    F --> G
    G --> H[Streamlit app<br/>app.py + views/]
    U[User: form / CSV] --> H
    H --> I[src/prediction.py]
    I --> G
    I --> J[(SQLite history)]
```

## 2. Training pipeline (`python train.py`)
```mermaid
flowchart TD
    S1[Load + strip column names] --> S2[Audit: missing, duplicates, constant,<br/>redundant pairs, outliers, leakage AUC]
    S2 --> S3[Drop constant + redundant features -> 90 features]
    S3 --> S4[Stratified split, random_state=42]
    S4 --> M1[SVM<br/>Imputer - Winsorizer - Scaler - SVC]
    S4 --> M2[Extra Trees<br/>Imputer - ExtraTrees]
    S4 --> M3[CatBoost<br/>Imputer - CatBoost]
    M1 & M2 & M3 --> T[RandomizedSearchCV<br/>5-fold CV, scoring PR-AUC, train only]
    T --> O[Out-of-fold predictions on train<br/>-> F1-optimal threshold]
    O --> V[Single evaluation on untouched test set]
    V --> B1[Best model = highest CV PR-AUC]
    V --> X[SHAP / permutation importance]
    B1 --> SAVE[Save models, preprocessing,<br/>evaluation JSON, figures]
    X --> SAVE
```

## 3. Application layers
```mermaid
flowchart TB
    subgraph UI[Presentation - Streamlit]
      APP[app.py: navigation, theme, sidebar]
      V1[views/dashboard.py]
      V2[views/prediction_studio.py]
      V3[views/model_intelligence.py]
      V4[views/risk_analytics.py]
      V5[views/company_report.py]
      V6[views/methodology.py]
      CSS[assets/custom.css]
    end
    subgraph SRC[Logic - src/]
      PRED[prediction.py]
      EXPL[explainability.py]
      VIZ[visualization.py]
      REP[report_generator.py]
      UIH[ui.py / app_data.py]
    end
    subgraph ART[Artefacts]
      MOD[models/*.pkl, *.cbm]
      RPT[reports/*.json, *.csv, figures]
      DB[(SQLite history)]
    end
    APP --> V1 & V2 & V3 & V4 & V5 & V6
    V1 & V2 & V3 & V4 & V5 & V6 --> UIH
    V2 & V5 --> PRED
    V2 & V5 --> EXPL
    V1 & V3 & V4 --> VIZ
    V5 --> REP
    PRED --> MOD
    PRED --> DB
    UIH --> RPT
```

## 4. Prediction flow (one company)
```mermaid
sequenceDiagram
    participant U as User
    participant S as Prediction Studio
    participant P as prediction.py
    participant M as Trained pipeline
    participant E as explainability.py
    U->>S: Enter 90 ratios / upload CSV
    S->>P: predict(df, model)
    P->>P: align columns, coerce numeric, flag missing
    P->>M: predict_proba
    M-->>P: probability
    P->>P: threshold -> class, risk band
    S->>E: local explanation (SHAP / occlusion)
    E-->>S: top contributing features
    S-->>U: gauge, class, risk, drivers, report
    P->>P: save to SQLite history
```

## 5. Design decisions
| Decision | Reason |
|---|---|
| Class weights, no SMOTE | 176 positives in 90 dimensions; synthetic data risky; SMOTE must be applied inside CV folds |
| PR-AUC for tuning/selection | 3.2% positive class; accuracy and ROC-AUC are optimistic |
| Threshold from out-of-fold training predictions | Test set never influences any choice |
| Winsorising for SVM only | Kernel models are outlier-sensitive; trees are invariant |
| `views/` instead of `pages/` | Streamlit auto-registers `pages/` and overrides custom navigation |
| Risk bands relative to model threshold | Probabilities are class-weighted; bands are a project convention, not an industry standard |
