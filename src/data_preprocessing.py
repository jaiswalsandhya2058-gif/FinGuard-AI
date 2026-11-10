"""Data loading, auditing, cleaning and splitting. All fitting happens on TRAIN data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config as cfg


def load_raw(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Read the CSV and strip the stray leading/trailing spaces found in the column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every feature to the [low, high] quantiles learned on the training fold.

    Several ratios in this dataset contain values around 1e9-1e10, which would dominate an RBF
    kernel after StandardScaler. Clipping (instead of deleting rows) keeps all 220 scarce
    bankrupt companies. Tree models are invariant to monotone clipping, so only SVM uses it.
    """

    def __init__(self, low: float = cfg.WINSOR_LOW, high: float = cfg.WINSOR_HIGH):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.low, axis=0)
        self.upper_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def set_output(self, *, transform=None):  # keep sklearn pipelines happy with numpy output
        return self


def audit_dataset(df: pd.DataFrame) -> dict:
    """Full data audit used by the EDA report and the Dataset Explorer."""
    y = df[cfg.TARGET]
    X = df.drop(columns=cfg.TARGET)
    nun = X.nunique()
    constant = nun[nun <= 1].index.tolist()
    Xv = X.drop(columns=constant)

    corr = Xv.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    dup_pairs = [(a, b, float(v)) for (a, b), v in upper[upper >= cfg.DUPLICATE_FEATURE_CORR].items()]
    high_pairs = int((upper >= 0.95).sum())

    auc = {c: roc_auc_score(y, Xv[c]) for c in Xv.columns}
    auc = pd.Series({k: max(v, 1 - v) for k, v in auc.items()}).sort_values(ascending=False)

    q1, q3 = Xv.quantile(0.25), Xv.quantile(0.75)
    iqr = q3 - q1
    outlier_share = (((Xv < q1 - 1.5 * iqr) | (Xv > q3 + 1.5 * iqr)).mean() * 100).sort_values(ascending=False)
    extreme = (Xv.max() >= 1e8)

    return {
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "n_features": int(X.shape[1]),
        "dtypes": {k: int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "bankruptcy_rate_pct": float(y.mean() * 100),
        "imbalance_ratio": float((y == 0).sum() / max((y == 1).sum(), 1)),
        "constant_features": constant,
        "duplicate_feature_pairs": dup_pairs,
        "pairs_corr_ge_095": high_pairs,
        "top_single_feature_auc": {k: float(v) for k, v in auc.head(15).items()},
        "max_single_feature_auc": float(auc.iloc[0]),
        "leakage_suspects": [k for k, v in auc.items() if v >= cfg.LEAKAGE_AUC_FLAG],
        "outlier_share_top": {k: float(v) for k, v in outlier_share.head(15).items()},
        "mean_outlier_share_pct": float(outlier_share.mean()),
        "features_with_extreme_values_1e8": extreme[extreme].index.tolist(),
        "single_feature_auc_all": {k: float(v) for k, v in auc.items()},
    }


def clean_dataset(df: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Drop duplicate rows, constant features and perfectly redundant features (with a log)."""
    log = {"rows_before": int(len(df))}
    df = df.drop_duplicates().reset_index(drop=True)
    log["rows_after_dedup"] = int(len(df))
    drop_const = list(audit["constant_features"])
    drop_dup = [b for _, b, _ in audit["duplicate_feature_pairs"]]  # keep first of each pair
    X = df.drop(columns=[cfg.TARGET] + drop_const + drop_dup)
    log.update({"dropped_constant": drop_const,
                "dropped_redundant": [{"kept": a, "dropped": b, "abs_corr": r}
                                      for a, b, r in audit["duplicate_feature_pairs"]],
                "n_features_final": int(X.shape[1])})
    return X, df[cfg.TARGET].astype(int), log


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified split. SAME split is reused by every model for a fair comparison."""
    return train_test_split(X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)


def feature_statistics(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train-only statistics powering input defaults, validation and report observations."""
    stats = {}
    healthy, bankrupt = X_train[y_train == 0], X_train[y_train == 1]
    for c in X_train.columns:
        s = X_train[c]
        stats[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                    "p01": float(s.quantile(.01)), "p99": float(s.quantile(.99)),
                    "healthy_median": float(healthy[c].median()), "bankrupt_median": float(bankrupt[c].median())}
    return stats
