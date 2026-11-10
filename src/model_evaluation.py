"""Metric computation, curves and best-model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve, matthews_corrcoef)


def _thin(a, n=250):
    a = np.asarray(a)
    if len(a) <= n:
        return a.tolist()
    return a[np.linspace(0, len(a) - 1, n).astype(int)].tolist()


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {"threshold": float(threshold),
            "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "mcc": matthews_corrcoef(y_true, pred),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_true, pred, target_names=["Non-Bankrupt", "Bankrupt"],
                                                           output_dict=True, zero_division=0)}


def evaluate_model(y_true, proba, threshold: float) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    pr, rc, _ = precision_recall_curve(y_true, proba)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "tuned": metrics_at(y_true, proba, threshold), "default_0_5": metrics_at(y_true, proba, 0.5),
            "roc_curve": {"fpr": _thin(fpr), "tpr": _thin(tpr)},
            "pr_curve": {"precision": _thin(pr), "recall": _thin(rc)},
            "calibration": {"observed": frac_pos.tolist(), "predicted": mean_pred.tolist()}}


def select_best_model(results: dict) -> tuple[str, str]:
    """Pick by mean CROSS-VALIDATED PR-AUC on the training set (never the test set)."""
    best = max(results, key=lambda k: results[k]["cv_pr_auc"])
    why = ("Selected on the highest mean 5-fold cross-validated PR-AUC (average precision) measured on the training "
           "data only. PR-AUC is preferred over accuracy/ROC-AUC because only ~3% of companies are bankrupt, so it "
           "directly reflects the precision/recall trade-off for the minority class. The untouched test set is used "
           "only to report final performance.")
    return best, why
