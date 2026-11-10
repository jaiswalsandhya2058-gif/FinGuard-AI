"""PAGE 5 - Company Risk Report."""
import streamlit as st

import config as cfg
from src import prediction, report_generator as rg, ui, visualization as viz


def render() -> None:
    ui.page_header("Company Risk Report", "A detailed, downloadable explanation of a single prediction: probability, risk category, drivers, observations and limitations.", "Report")
    ctx = st.session_state.get("report_ctx")
    if not ctx:
        ui.empty_state("No prediction available yet.<br>Run a prediction in the <b>Prediction Studio</b> (single or batch) and the report will appear here.")
        if st.button("Go to Prediction Studio", type="primary"):
            st.switch_page(st.session_state["_pages"]["prediction-studio"])
        ui.disclaimer()
        return
    results, X, key = ctx["results"].reset_index(drop=True), ctx["X"].reset_index(drop=True), ctx["model"]
    if len(results) > 1:
        order = results.sort_values("Bankruptcy Probability", ascending=False)
        pick = st.selectbox("Company", order.index, format_func=lambda i: f"{results.loc[i, 'Company ID']}  -  {results.loc[i, 'Bankruptcy Probability'] * 100:.1f}%  ({results.loc[i, 'Risk Category']})")
    else:
        pick = 0
    r = results.loc[pick].to_dict()
    with st.spinner("Computing feature contributions ..."):
        expl = prediction.explain_company(key, X.iloc[[pick]], top_k=12)
    notes = prediction.observations(expl.head(6))
    color = cfg.RISK_COLORS[r["Risk Category"]]

    ui.kpi_row([("Prediction", r["Predicted Class"], f"Company: {r['Company ID']}", cfg.PALETTE["red"] if r["Predicted Class"] == "Bankrupt" else cfg.PALETTE["green"]),
                ("Bankruptcy probability", f"{r['Bankruptcy Probability'] * 100:.1f}%", f"threshold {r['Decision Threshold'] * 100:.1f}%", cfg.PALETTE["blue"]),
                ("Risk category", r["Risk Category"], "probability band (project convention)", color),
                ("Algorithm", cfg.MODELS[key]["short"], cfg.MODELS[key]["name"], cfg.MODEL_COLORS[key])])
    c1, c2 = st.columns([1, 1.5])
    with c1:
        ui.section("Risk gauge")
        ui.plot(viz.gauge(r["Bankruptcy Probability"], r["Decision Threshold"], r["Risk Category"]), "rep_gauge")
        b = prediction.risk_bands(r["Decision Threshold"])
        st.caption(" | ".join(f"{k}: {a * 100:.0f}-{c * 100:.0f}%" for k, (a, c) in b.items()))
    with c2:
        ui.section("Important contributing features")
        ui.plot(viz.contribution_bar(expl.head(10), title=None), "rep_contrib")
    ui.section("Financial risk observations")
    for n in notes:
        st.markdown(f"- {n}")
    ui.section("Model limitations")
    for l in rg.LIMITATIONS:
        st.markdown(f"- {l}")
    ui.section("Download")
    d = st.columns([1, 1, 3])
    name = str(r["Company ID"]).replace("/", "-").replace(" ", "_")
    d[0].download_button("Report (PDF)", rg.report_pdf(str(r["Company ID"]), r, expl, notes), f"risk_report_{name}.pdf", "application/pdf", type="primary", width="stretch")
    d[1].download_button("Report (CSV)", rg.report_dataframe(str(r["Company ID"]), r, expl, notes).to_csv(index=False).encode(), f"risk_report_{name}.csv", "text/csv", width="stretch")
    ui.disclaimer()
