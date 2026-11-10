"""Pipelines, hyper-parameter search, threshold selection and model persistence."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as cfg
from src.data_preprocessing import Winsorizer

N_JOBS = -1


def build_pipeline(key: str, probability: bool = True) -> Pipeline:
    """Each model gets the preprocessing it actually needs (all fitted inside CV folds)."""
    imputer = ("imputer", SimpleImputer(strategy="median"))
    if key == "svm":
        # Outlier clipping + standardisation are essential for a distance/kernel based model.
        return Pipeline([imputer, ("winsor", Winsorizer()), ("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", class_weight="balanced", probability=probability,
                                     cache_size=500, random_state=cfg.RANDOM_STATE))])
    if key == "extra_trees":
        # Trees are scale/outlier invariant: median imputation only.
        return Pipeline([imputer, ("clf", ExtraTreesClassifier(
            n_jobs=N_JOBS, random_state=cfg.RANDOM_STATE))])
    if key == "catboost":
        # CatBoost needs no scaling; 'Balanced' reweights the minority class.
        return Pipeline([imputer, ("clf", CatBoostClassifier(
            auto_class_weights="Balanced", random_seed=cfg.RANDOM_STATE, verbose=0,
            allow_writing_files=False, thread_count=4))])
    raise KeyError(key)


SEARCH_SPACES = {
    "svm": {"clf__C": loguniform(0.3, 300), "clf__gamma": ["scale", 0.0005, 0.001, 0.003, 0.01]},
    "extra_trees": {"clf__n_estimators": [200, 300, 400], "clf__max_depth": [None, 8, 14],
                    "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2", 0.3],
                    "clf__class_weight": ["balanced", "balanced_subsample"]},
    "catboost": {"clf__iterations": [200, 350, 500], "clf__learning_rate": [0.03, 0.06, 0.1],
                 "clf__depth": [4, 6, 8], "clf__l2_leaf_reg": [1, 3, 9]},
}
N_ITER = {"svm": 10, "extra_trees": 10, "catboost": 8}


def train_model(key: str, X_train: pd.DataFrame, y_train: pd.Series, log=print) -> dict:
    """Tune on the training set only (stratified CV, PR-AUC), refit, pick a decision threshold."""
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    t0 = time.time()
    # For SVM, tune without Platt scaling (decision_function is enough for PR-AUC) => much faster.
    search_pipe = build_pipeline(key, probability=False) if key == "svm" else build_pipeline(key)
    search = RandomizedSearchCV(search_pipe, SEARCH_SPACES[key], n_iter=N_ITER[key], scoring="average_precision",
                                cv=cv, random_state=cfg.RANDOM_STATE, n_jobs=1 if key == "catboost" else N_JOBS,
                                refit=False, verbose=0)
    search.fit(X_train, y_train)
    tuning_time = time.time() - t0
    best_params = search.best_params_
    cv_pr_auc = float(search.best_score_)
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
    log(f"  [{key}] best CV PR-AUC={cv_pr_auc:.4f} (+/-{cv_std:.4f}) params={best_params} tuning={tuning_time:.0f}s")

    # Final model with probability estimates, timed on the full training set.
    final = build_pipeline(key, probability=True).set_params(**best_params)
    t0 = time.time()
    final.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Out-of-fold probabilities on TRAIN only -> choose the F1-optimal threshold without touching the test set.
    oof = cross_val_predict(clone(final), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    p, r, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    threshold = float(thr[int(np.argmax(f1))])
    log(f"  [{key}] fit={fit_time:.1f}s  OOF F1-optimal threshold={threshold:.3f} (OOF F1={f1.max():.3f})")
    return {"pipeline": final, "best_params": {k.replace("clf__", ""): (v if not isinstance(v, np.generic) else v.item())
                                                for k, v in best_params.items()},
            "cv_pr_auc": cv_pr_auc, "cv_pr_auc_std": cv_std, "threshold": threshold, "oof_f1": float(f1.max()),
            "fit_time_sec": fit_time, "tuning_time_sec": tuning_time, "n_iter": N_ITER[key]}


def save_model(key: str, pipeline: Pipeline) -> None:
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    path = cfg.MODELS_DIR / cfg.MODELS[key]["file"]
    if key == "catboost":
        pipeline.named_steps["clf"].save_model(str(path))              # native CatBoost format
        joblib.dump(pipeline, cfg.MODELS_DIR / "catboost_pipeline.pkl", compress=3)  # imputer + model
    else:
        joblib.dump(pipeline, path, compress=3)
