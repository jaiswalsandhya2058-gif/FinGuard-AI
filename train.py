"""End-to-end training pipeline.  Run:  python train.py

audit -> clean -> stratified split -> tune/train SVM, Extra Trees, CatBoost -> evaluate on the held-out test set
-> explainability -> save every artefact into models/ and reports/.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src import data_preprocessing as dp
from src import eda_figures, explainability, model_evaluation as me, model_training as mt

warnings.filterwarnings("ignore")


def main() -> None:
    T0 = time.time()
    cfg.REPORTS_DIR.mkdir(exist_ok=True); cfg.MODELS_DIR.mkdir(exist_ok=True)

    print("[1/6] Loading and auditing dataset ...")
    df = dp.load_raw()
    audit = dp.audit_dataset(df)
    print(f"  shape={df.shape}  bankrupt={audit['class_counts']['1']} ({audit['bankruptcy_rate_pct']:.2f}%)  "
          f"missing={audit['missing_total']}  duplicates={audit['duplicate_rows']}")
    print(f"  constant={audit['constant_features']}  redundant pairs={[(a, b) for a, b, _ in audit['duplicate_feature_pairs']]}")
    print(f"  max single-feature AUC={audit['max_single_feature_auc']:.3f}  leakage suspects={audit['leakage_suspects']}")

    print("[2/6] Cleaning + stratified split ...")
    X, y, clean_log = dp.clean_dataset(df, audit)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)
    print(f"  features={X.shape[1]}  train={len(X_train)} (bankrupt {y_train.sum()})  test={len(X_test)} (bankrupt {y_test.sum()})")
    stats = dp.feature_statistics(X_train, y_train)
    audit["cleaning"] = clean_log
    audit["split"] = {"train_rows": len(X_train), "test_rows": len(X_test), "train_bankrupt": int(y_train.sum()),
                      "test_bankrupt": int(y_test.sum()), "test_size": cfg.TEST_SIZE, "random_state": cfg.RANDOM_STATE}
    with open(cfg.EDA_FILE, "w") as f:
        json.dump(audit, f, indent=2)
    eda_figures.make_eda_figures(df, audit)

    print("[3/6] Training models (tuning with 5-fold stratified CV on the training set) ...")
    trained = {}
    for key in cfg.MODELS:
        print(f" -> {cfg.MODELS[key]['name']}")
        trained[key] = mt.train_model(key, X_train, y_train)
        mt.save_model(key, trained[key]["pipeline"])

    print("[4/6] Evaluating on the untouched test set ...")
    results = {}
    test_pred = pd.DataFrame({"row_id": X_test.index, "y_true": y_test.values})
    for key, t in trained.items():
        proba = t["pipeline"].predict_proba(X_test)[:, 1]
        test_pred[f"proba_{key}"] = proba
        ev = me.evaluate_model(y_test.values, proba, t["threshold"])
        results[key] = {"name": cfg.MODELS[key]["name"], "best_params": t["best_params"], "cv_pr_auc": t["cv_pr_auc"],
                        "cv_pr_auc_std": t["cv_pr_auc_std"], "threshold": t["threshold"], "oof_f1": t["oof_f1"],
                        "fit_time_sec": t["fit_time_sec"], "tuning_time_sec": t["tuning_time_sec"],
                        "n_iter": t["n_iter"], "evaluation": ev}
        m = ev["tuned"]
        print(f"  {key:12s} thr={t['threshold']:.3f} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} f1={m['f1']:.3f} roc={ev['roc_auc']:.3f} pr={ev['pr_auc']:.3f}")
    test_pred.to_csv(cfg.TEST_PRED_FILE, index=False)
    best, why = me.select_best_model(results)
    print(f"  BEST MODEL (by CV PR-AUC): {best}")

    print("[5/6] Explainability (SHAP / permutation importance) ...")
    imps, extras = [], {}
    for key, t in trained.items():
        d, ex = explainability.global_importance(key, t["pipeline"], X_test, y_test)
        imps.append(d); extras[key] = ex
        print(f"  {key}: done")
    pd.concat(imps).to_csv(cfg.IMPORTANCE_FILE, index=False)
    eda_figures.make_model_figures(results, extras)

    print("[6/6] Saving artefacts ...")
    joblib.dump({"features": list(X.columns), "stats": stats, "dropped_constant": clean_log["dropped_constant"],
                 "dropped_redundant": clean_log["dropped_redundant"], "target": cfg.TARGET,
                 "thresholds": {k: t["threshold"] for k, t in trained.items()},
                 "svm_occlusion_candidates": list(imps[0].query("method.str.startswith('Perm')").nlargest(30, "importance")["feature"]),
                 "random_state": cfg.RANDOM_STATE, "test_size": cfg.TEST_SIZE}, cfg.PREPROCESSING_FILE, compress=3)
    joblib.dump({"key": best, "name": cfg.MODELS[best]["name"], "pipeline": trained[best]["pipeline"],
                 "threshold": trained[best]["threshold"]}, cfg.BEST_MODEL_FILE, compress=3)

    # Sample inputs (real HELD-OUT test companies, label removed) + blank template
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    pos = X_test[y_test == 1].sample(10, random_state=cfg.RANDOM_STATE)
    neg = X_test[y_test == 0].sample(15, random_state=cfg.RANDOM_STATE)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=cfg.RANDOM_STATE)
    sample.insert(0, "Company ID", [f"TEST-{i:04d}" for i in sample.index])
    sample.to_csv(cfg.SAMPLE_CSV, index=False)
    pd.DataFrame([{**{"Company ID": "COMPANY-001"}, **{c: stats[c]["median"] for c in X.columns}}]).to_csv(cfg.TEMPLATE_CSV, index=False)

    with open(cfg.EVALUATION_FILE, "w") as f:
        json.dump({"results": results, "best_model": best, "selection_reason": why,
                   "split": audit["split"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "total_runtime_sec": time.time() - T0}, f, indent=2, default=float)
    print(f"Done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
