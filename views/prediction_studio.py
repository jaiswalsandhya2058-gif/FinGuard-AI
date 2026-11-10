"""PAGE 2 - Bankruptcy Prediction Studio (single company form + batch CSV)."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import config as cfg
from src import app_data, prediction, ui, visualization as viz

N_FORM = 12          # most influential indicators shown directly in the form
MAX_BATCH = 20000


def _key(f: str) -> str:
    return f"in::{f}"


def _bounds(feats):
    df = app_data.dataset()
    return df[feats].min(), df[feats].max()


def _load_example(kind: str, feats: list[str]) -> None:
    tp = app_data.test_predictions()
    ids = tp.loc[tp["y_true"] == (1 if kind == "bankrupt" else 0), "row_id"].to_numpy()
    row = app_data.dataset().loc[int(np.random.choice(ids)), feats] if kind != "median" else None
    stats = app_data.artefacts()["pre"]["stats"]
    for f in feats:
        st.session_state[_key(f)] = float(row[f]) if row is not None else float(stats[f]["median"])
    st.session_state["example_note"] = {"bankrupt": "Loaded a real BANKRUPT company from the held-out test set.",
                                       "healthy": "Loaded a real NON-BANKRUPT company from the held-out test set.",
                                       "median": "Reset every indicator to the training median."}[kind]


def _model_picker(key_suffix: str) -> str:
    art = app_data.artefacts()
    keys = list(cfg.MODELS)
    default = keys.index(art["best"])
    return st.selectbox("Prediction model", keys, index=default, key=f"model_{key_suffix}",
                        format_func=lambda k: cfg.MODELS[k]["name"] + ("  (best model)" if k == art["best"] else ""))


def _result_view(res: dict) -> None:
    r, expl = res["result"], res["expl"]
    risk, p, thr = r["Risk Category"], r["Bankruptcy Probability"], r["Decision Threshold"]
    color = cfg.RISK_COLORS[risk]
    c1, c2 = st.columns([1, 1.15])
    with c1:
        ui.plot(viz.gauge(p, thr, risk), "gauge")
        st.caption("Needle = predicted probability. White marker = this model's decision threshold. Coloured bands = risk categories.")
    with c2:
        cls_color = cfg.PALETTE["red"] if r["Predicted Class"] == "Bankrupt" else cfg.PALETTE["green"]
        st.markdown(f"""<div class="result-hero" style="border-color:{color}88">
            <div class="k-label" style="color:var(--muted);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Predicted class</div>
            <div class="big" style="color:{cls_color}">{r['Predicted Class']}</div>
            <div style="margin:10px 0 14px">{ui.badge(risk + ' risk', color)} &nbsp; {ui.badge(cfg.MODELS[res['model']]['short'], cfg.MODEL_COLORS[res['model']])}</div>
            <div style="color:var(--muted);font-size:.9rem;line-height:1.7">Bankruptcy probability: <b style="color:var(--text)">{p * 100:.2f}%</b><br>
            Decision threshold: <b style="color:var(--text)">{thr * 100:.1f}%</b><br>Company: <b style="color:var(--text)">{r['Company ID']}</b></div></div>""",
                    unsafe_allow_html=True)
    ui.section("Why this prediction? Top contributing features")
    ui.plot(viz.contribution_bar(expl.head(8)), "contrib")
    with st.expander("Plain-language observations"):
        for n in prediction.observations(expl.head(5)):
            st.markdown(f"- {n}")


def _single_tab() -> None:
    art = app_data.artefacts()
    feats, stats = art["pre"]["features"], art["pre"]["stats"]
    for f in feats:
        st.session_state.setdefault(_key(f), float(stats[f]["median"]))
    lo, hi = _bounds(feats)

    c0, c1 = st.columns([1, 2])
    with c0:
        model_key = _model_picker("single")
        company = st.text_input("Company name / ID", value="Company-A", key="company_name")
    with c1:
        st.markdown("**Quick start** - fill the form with a real company from the held-out test set (never seen during training):")
        b = st.columns(3)
        b[0].button("Load bankrupt example", on_click=_load_example, args=("bankrupt", feats), width="stretch")
        b[1].button("Load healthy example", on_click=_load_example, args=("healthy", feats), width="stretch")
        b[2].button("Reset to medians", on_click=_load_example, args=("median", feats), width="stretch")
        if st.session_state.get("example_note"):
            st.caption(st.session_state["example_note"])

    imp = app_data.importance()
    m = imp[(imp["model"] == model_key) & (imp["method"].isin(["Mean |SHAP|"]) | imp["method"].str.startswith("Perm"))]
    top = m.nlargest(N_FORM, "importance")["feature"].tolist()
    rest = [f for f in feats if f not in top]

    def field(f: str):
        span = max(hi[f] - lo[f], 1e-9)
        step = float(max(min((stats[f]["p99"] - stats[f]["p01"]) / 200, span / 20), 1e-6))
        st.number_input(f, min_value=float(lo[f]), max_value=float(hi[f]), step=step, format="%.6g", key=_key(f),
                        help=f"Training median {stats[f]['median']:.4g} | typical range (1st-99th pct) {stats[f]['p01']:.4g} to {stats[f]['p99']:.4g}")

    ui.section(f"Key financial indicators (top {N_FORM} for {cfg.MODELS[model_key]['short']})")
    cols = st.columns(3)
    for i, f in enumerate(top):
        with cols[i % 3]:
            field(f)
    with st.expander(f"All remaining indicators ({len(rest)}) - advanced"):
        cols = st.columns(3)
        for i, f in enumerate(rest):
            with cols[i % 3]:
                field(f)

    save = st.checkbox("Save this prediction to history", value=True)
    if st.button("Run bankruptcy prediction", type="primary"):
        row = pd.DataFrame([{f: st.session_state[_key(f)] for f in feats}])
        row.insert(0, "Company ID", company or "Company-A")
        with st.spinner("Scoring with the trained model ..."):
            out, X, info = prediction.predict(row, model_key)
            expl = prediction.explain_company(model_key, X.iloc[[0]], top_k=12)
        outside = [f for f in feats if not (stats[f]["p01"] <= st.session_state[_key(f)] <= stats[f]["p99"])]
        res = {"result": out.iloc[0].to_dict(), "expl": expl, "model": model_key, "outside": outside}
        st.session_state["single"] = res
        st.session_state["report_ctx"] = {"source": "single", "model": model_key, "results": out, "X": X}
        if save:
            prediction.save_history(out, "manual")
    res = st.session_state.get("single")
    if res:
        ui.section("Prediction result")
        _result_view(res)
        if res["outside"]:
            ui.callout(f"<b>Input check:</b> {len(res['outside'])} indicator(s) lie outside the typical 1st-99th percentile range seen in training "
                       "(e.g. " + ", ".join(res["outside"][:3]) + "). The model is less reliable for unusual values.", "warn")
        if st.button("Open full company risk report"):
            st.switch_page(st.session_state["_pages"]["company-report"])


def _batch_tab() -> None:
    st.markdown("Upload a CSV with one company per row. Column names must match the dataset's financial indicators "
                "(an optional **Company ID** column is used as the label; an optional **Bankrupt?** column enables an accuracy check).")
    d = st.columns([1, 1, 2])
    d[0].download_button("Download input template", cfg.TEMPLATE_CSV.read_bytes(), "finguard_input_template.csv", "text/csv", width="stretch")
    d[1].download_button("Download sample companies", cfg.SAMPLE_CSV.read_bytes(), "finguard_sample_input.csv", "text/csv", width="stretch")
    d[2].caption("The sample file contains 25 real held-out test companies (10 bankrupt, 15 healthy) with labels removed.")
    c0, c1 = st.columns([1, 2])
    with c0:
        model_key = _model_picker("batch")
    with c1:
        up = st.file_uploader("Upload companies (CSV)", type=["csv"], key="batch_upload")
    use_sample = st.checkbox("Use the bundled sample file instead of uploading", value=False)
    if st.button("Run batch prediction", type="primary"):
        try:
            df = pd.read_csv(cfg.SAMPLE_CSV) if use_sample else (pd.read_csv(up) if up else None)
            if df is None:
                st.warning("Please upload a CSV file (or tick the sample option).")
            elif len(df) > MAX_BATCH:
                st.error(f"File has {len(df):,} rows; the limit is {MAX_BATCH:,}.")
            else:
                with st.spinner(f"Scoring {len(df):,} companies ..."):
                    out, X, info = prediction.predict(df, model_key)
                labels = None
                dfc = df.copy(); dfc.columns = dfc.columns.str.strip()
                if cfg.TARGET in dfc.columns:
                    labels = pd.to_numeric(dfc[cfg.TARGET], errors="coerce").fillna(0).astype(int).to_numpy()
                    out.insert(3, "Actual (uploaded)", np.where(labels == 1, "Bankrupt", "Non-Bankrupt"))
                st.session_state["batch"] = {"out": out, "info": info, "model": model_key, "labels": labels}
                st.session_state["report_ctx"] = {"source": "batch", "model": model_key, "results": out, "X": X}
                prediction.save_history(out, "batch")
        except Exception as e:  # surface a readable validation message
            st.error(f"Could not process the file: {e}")
    b = st.session_state.get("batch")
    if not b:
        return
    out, info = b["out"], b["info"]
    ui.section("Batch results")
    if info["missing_columns"]:
        ui.callout(f"<b>{len(info['missing_columns'])} of {info['n_expected']} expected columns were missing</b> and were filled with training medians "
                   f"({info['cells_imputed']:,} cells). Predictions are less reliable. Missing e.g.: {', '.join(info['missing_columns'][:4])}.", "warn")
    n_b = int((out["Predicted Class"] == "Bankrupt").sum())
    ui.kpi_row([("Companies scored", f"{len(out):,}", cfg.MODELS[b['model']]['name'], cfg.PALETTE["blue"]),
                ("Flagged bankrupt", f"{n_b:,}", f"{n_b / len(out) * 100:.1f}% of batch", cfg.PALETTE["red"]),
                ("Highest risk", f"{out['Bankruptcy Probability'].max() * 100:.1f}%", str(out.loc[out['Bankruptcy Probability'].idxmax(), 'Company ID']), cfg.PALETTE["amber"]),
                ("Average probability", f"{out['Bankruptcy Probability'].mean() * 100:.1f}%", "across the batch", cfg.PALETTE["cyan"])])
    c1, c2 = st.columns([1, 1.6])
    with c1:
        ui.plot(viz.risk_category_bar(out["Risk Category"].value_counts(), "Risk categories"), "batch_risk")
    with c2:
        ui.plot(viz.risk_hist(out["Bankruptcy Probability"].to_numpy(), None, float(out["Decision Threshold"].iloc[0])), "batch_hist")
    if b["labels"] is not None:
        y, pred = b["labels"], (out["Predicted Class"] == "Bankrupt").astype(int).to_numpy()
        ui.callout(f"<b>Accuracy check against your uploaded labels</b> - precision {precision_score(y, pred, zero_division=0):.2f}, "
                   f"recall {recall_score(y, pred, zero_division=0):.2f}, F1 {f1_score(y, pred, zero_division=0):.2f}. "
                   "(Meaningful only if these companies were not used for training.)", "info")
    st.dataframe(out.sort_values("Bankruptcy Probability", ascending=False), width="stretch", hide_index=True,
                 column_config={"Bankruptcy Probability": st.column_config.ProgressColumn("Bankruptcy Probability", min_value=0, max_value=1, format="%.3f"),
                                "Decision Threshold": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "finguard_predictions.csv", "text/csv", type="primary")
    st.caption("Results were saved to prediction history. Open **Company Risk Report** to inspect any company in detail.")


def render() -> None:
    ui.page_header("Bankruptcy Prediction Studio", "Score one company manually or a whole portfolio from CSV using any of the three trained models.", "Predict")
    t1, t2 = st.tabs(["Single company", "Batch upload (CSV)"])
    with t1:
        _single_tab()
    with t2:
        _batch_tab()
    ui.disclaimer()
