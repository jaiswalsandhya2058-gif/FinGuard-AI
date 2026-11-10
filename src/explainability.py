"""Global + local explanations. SHAP for tree models, permutation importance / occlusion for SVM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from catboost import Pool
from sklearn.inspection import permutation_importance

import config as cfg

_EXPLAINERS: dict = {}


def _imputed(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    Xi = pipeline.named_steps["imputer"].transform(X)
    return pd.DataFrame(Xi, columns=X.columns, index=X.index)


def tree_shap_values(key: str, pipeline, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds for CatBoost, probability units for Extra Trees) for the bankrupt class."""
    Xi = _imputed(pipeline, X)
    clf = pipeline.named_steps["clf"]
    if key == "catboost":
        return clf.get_feature_importance(Pool(Xi), type="ShapValues")[:, :-1]
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(clf)
    sv = _EXPLAINERS[key].shap_values(Xi, check_additivity=False)
    sv = np.array(sv[1]) if isinstance(sv, list) else np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def global_importance(key: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_shap: int = 300) -> tuple[pd.DataFrame, dict]:
    """Returns long-format importance table + the SHAP sample (for summary plots)."""
    feats = list(X_test.columns)
    rows, extra = [], {}
    clf = pipeline.named_steps["clf"]
    if key == "svm":
        pi = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                    random_state=cfg.RANDOM_STATE, n_jobs=-1)
        vals, method = np.clip(pi.importances_mean, 0, None), "Permutation importance (drop in ROC-AUC)"
        rows += [(key, method, f, v) for f, v in zip(feats, vals)]
    else:
        native = clf.feature_importances_ if key == "extra_trees" else clf.get_feature_importance()
        rows += [(key, "Native importance (" + ("Gini/MDI" if key == "extra_trees" else "PredictionValuesChange") + ")", f, v)
                 for f, v in zip(feats, native)]
        # stratified SHAP sample: every positive + random negatives
        pos = X_test[y_test == 1]
        neg = X_test[y_test == 0].sample(max(n_shap - len(pos), 50), random_state=cfg.RANDOM_STATE)
        sample = pd.concat([pos, neg])
        sv = tree_shap_values(key, pipeline, sample)
        rows += [(key, "Mean |SHAP|", f, v) for f, v in zip(feats, np.abs(sv).mean(axis=0))]
        extra = {"shap_values": sv, "sample": _imputed(pipeline, sample)}
    df = pd.DataFrame(rows, columns=["model", "method", "feature", "importance"])
    df["importance"] = df["importance"] / df.groupby(["model", "method"])["importance"].transform("sum").clip(lower=1e-12)
    return df, extra


def local_explanation(key: str, pipeline, x_row: pd.DataFrame, stats: dict, svm_candidates: list[str] | None = None,
                      top_k: int = 10) -> pd.DataFrame:
    """Per-company contribution of each feature to the bankruptcy score (+ = raises risk)."""
    feats = list(x_row.columns)
    xi = _imputed(pipeline, x_row)
    if key in ("catboost", "extra_trees"):
        contrib = tree_shap_values(key, pipeline, x_row)[0]
        unit = "SHAP (log-odds)" if key == "catboost" else "SHAP (probability)"
    else:
        # Occlusion: replace one feature at a time by the training median and measure the probability drop.
        cands = svm_candidates or feats
        base = pipeline.predict_proba(xi)[0, 1]
        variants = []
        for f in cands:
            v = xi.copy()
            v[f] = stats[f]["median"]
            variants.append(v)
        probs = pipeline.predict_proba(pd.concat(variants))[:, 1]
        contrib = np.zeros(len(feats))
        for f, p in zip(cands, probs):
            contrib[feats.index(f)] = base - p
        unit = "Occlusion (probability change)"
    out = pd.DataFrame({"feature": feats, "value": xi.iloc[0].values, "contribution": contrib,
                        "healthy_median": [stats[f]["healthy_median"] for f in feats],
                        "bankrupt_median": [stats[f]["bankrupt_median"] for f in feats]})
    out["unit"] = unit
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_k)
    return out.reset_index(drop=True)
