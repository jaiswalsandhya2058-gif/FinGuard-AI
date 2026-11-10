"""Cached data loaders for the Streamlit app."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config as cfg
from src import data_preprocessing as dp
from src import prediction


@st.cache_resource(show_spinner="Loading trained models ...")
def artefacts() -> dict:
    return prediction.load_artefacts()


@st.cache_data(show_spinner=False)
def dataset() -> pd.DataFrame:
    return dp.load_raw()


@st.cache_data(show_spinner=False)
def eda() -> dict:
    with open(cfg.EDA_FILE) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def importance() -> pd.DataFrame:
    return pd.read_csv(cfg.IMPORTANCE_FILE)


@st.cache_data(show_spinner=False)
def test_predictions() -> pd.DataFrame:
    return pd.read_csv(cfg.TEST_PRED_FILE)


@st.cache_data(show_spinner=False)
def correlation(features: tuple) -> pd.DataFrame:
    return dataset()[list(features)].corr()


def metrics_table() -> pd.DataFrame:
    """One row per model: tuned-threshold test metrics, CV PR-AUC and timings."""
    ev = artefacts()["evaluation"]["results"]
    rows = []
    for k, r in ev.items():
        m = r["evaluation"]["tuned"]
        rows.append({"key": k, "Model": cfg.MODELS[k]["name"], "Short": cfg.MODELS[k]["short"], "Accuracy": m["accuracy"],
                     "Precision": m["precision"], "Recall": m["recall"], "F1-Score": m["f1"], "ROC-AUC": r["evaluation"]["roc_auc"],
                     "PR-AUC": r["evaluation"]["pr_auc"], "CV PR-AUC": r["cv_pr_auc"], "CV PR-AUC std": r["cv_pr_auc_std"],
                     "Threshold": r["threshold"], "Train time (s)": r["fit_time_sec"], "Tuning time (s)": r["tuning_time_sec"]})
    return pd.DataFrame(rows)
