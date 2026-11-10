"""Loading trained artefacts, scoring companies, risk bands and prediction-history storage."""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src import explainability


class ArtefactsMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_artefacts() -> dict:
    """Load preprocessing metadata, all three trained pipelines and the evaluation report."""
    need = [cfg.PREPROCESSING_FILE, cfg.EVALUATION_FILE, cfg.MODELS_DIR / "svm_model.pkl",
            cfg.MODELS_DIR / "extra_trees_model.pkl", cfg.MODELS_DIR / "catboost_pipeline.pkl"]
    if any(not p.exists() for p in need):
        raise ArtefactsMissing("Trained models were not found. Run `python train.py` first.")
    pre = joblib.load(cfg.PREPROCESSING_FILE)
    models = {"svm": joblib.load(cfg.MODELS_DIR / "svm_model.pkl"),
              "extra_trees": joblib.load(cfg.MODELS_DIR / "extra_trees_model.pkl"),
              "catboost": joblib.load(cfg.MODELS_DIR / "catboost_pipeline.pkl")}
    with open(cfg.EVALUATION_FILE) as f:
        evaluation = json.load(f)
    return {"pre": pre, "models": models, "evaluation": evaluation, "best": evaluation["best_model"]}


def risk_level(p: float, t: float) -> str:
    """Bands anchored to the model's decision threshold t:
    Low: p < t/2 | Moderate: t/2 <= p < t | High: t <= p < (1+t)/2 | Critical: p >= (1+t)/2."""
    if p < t / 2:
        return "Low"
    if p < t:
        return "Moderate"
    if p < (1 + t) / 2:
        return "High"
    return "Critical"


def risk_bands(t: float) -> dict:
    return {"Low": (0.0, t / 2), "Moderate": (t / 2, t), "High": (t, (1 + t) / 2), "Critical": ((1 + t) / 2, 1.0)}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Align an uploaded table to the training feature set. Never silently invents data:
    missing columns become NaN and are median-imputed by the model pipeline (reported to the user)."""
    art = load_artefacts()
    feats = art["pre"]["features"]
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    present = [f for f in feats if f in d.columns]
    if not present:
        raise ValueError("None of the expected financial columns were found. Download the input template to see them.")
    X = d.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    inf = np.isinf(X.to_numpy(dtype=float))
    X = X.replace([np.inf, -np.inf], np.nan)
    info = {"missing_columns": [f for f in feats if f not in d.columns],
            "n_present": len(present), "n_expected": len(feats), "cells_imputed": int(X.isna().sum().sum()),
            "inf_replaced": int(inf.sum()),
            "ignored_columns": [c for c in d.columns if c not in feats]}
    return X, info


def predict(df: pd.DataFrame, model_key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (results table, aligned feature matrix, info)."""
    art = load_artefacts()
    X, info = prepare_features(df)
    thr = art["pre"]["thresholds"][model_key]
    proba = art["models"][model_key].predict_proba(X)[:, 1]
    if "Company ID" in df.columns:
        ids = df["Company ID"].astype(str).values
    else:
        ids = np.array([f"Company-{i + 1}" for i in range(len(df))])
    out = pd.DataFrame({"Company ID": ids, "Bankruptcy Probability": proba,
                        "Predicted Class": np.where(proba >= thr, "Bankrupt", "Non-Bankrupt"),
                        "Risk Category": [risk_level(p, thr) for p in proba],
                        "Model": cfg.MODELS[model_key]["name"], "Decision Threshold": thr})
    X.index = out.index
    return out, X, info


def explain_company(model_key: str, x_row: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    art = load_artefacts()
    return explainability.local_explanation(model_key, art["models"][model_key], x_row, art["pre"]["stats"],
                                            art["pre"].get("svm_occlusion_candidates"), top_k)


def observations(expl: pd.DataFrame) -> list[str]:
    """Plain-language, data-driven observations (compares each company ratio with healthy/bankrupt medians)."""
    notes = []
    for _, r in expl.iterrows():
        h, b, v = r["healthy_median"], r["bankrupt_median"], r["value"]
        closer = "bankrupt" if abs(v - b) < abs(v - h) else "non-bankrupt"
        direction = "raises" if r["contribution"] > 0 else "lowers"
        notes.append(f"{r['feature']} = {v:.4g} is closer to the typical {closer} company "
                     f"(non-bankrupt median {h:.4g}, bankrupt median {b:.4g}); this {direction} the model's risk estimate.")
    return notes


# ---------------------------------------------------------------- prediction history (SQLite)
def _conn():
    cfg.REPORTS_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(cfg.HISTORY_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, company TEXT,
                   model TEXT, probability REAL, predicted TEXT, risk TEXT, source TEXT)""")
    return con


def save_history(results: pd.DataFrame, source: str) -> int:
    con = _conn()
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(ts, r["Company ID"], r["Model"], float(r["Bankruptcy Probability"]), r["Predicted Class"], r["Risk Category"], source)
            for _, r in results.iterrows()]
    con.executemany("INSERT INTO history (ts, company, model, probability, predicted, risk, source) VALUES (?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()
    return len(rows)


def read_history(limit: int | None = None) -> pd.DataFrame:
    con = _conn()
    q = "SELECT ts AS Time, company AS Company, model AS Model, probability AS Probability, predicted AS Prediction, risk AS Risk, source AS Source FROM history ORDER BY id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q, con); con.close()
    return df


def clear_history() -> None:
    con = _conn(); con.execute("DELETE FROM history"); con.commit(); con.close()
