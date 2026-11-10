"""FinGuard AI - Bankruptcy Prediction & Financial Risk Analytics Platform (Streamlit entry point)."""
import streamlit as st

import config as cfg

st.set_page_config(page_title="FinGuard AI | Bankruptcy Risk Intelligence", page_icon=":material/shield:",
                   layout="wide", initial_sidebar_state="expanded")

from views import company_report, dashboard, methodology, model_intelligence, prediction_studio, risk_analytics  # noqa: E402
from src import app_data, ui, visualization as viz  # noqa: E402
from src.prediction import ArtefactsMissing  # noqa: E402

st.session_state["theme"] = "light" if st.session_state.get("light_mode") else "dark"
viz.set_theme(st.session_state["theme"])
ui.inject_theme()

PAGES = [
    st.Page(dashboard.render, title="Executive Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True),
    st.Page(prediction_studio.render, title="Prediction Studio", icon=":material/query_stats:", url_path="prediction-studio"),
    st.Page(model_intelligence.render, title="Model Intelligence", icon=":material/neurology:", url_path="model-intelligence"),
    st.Page(risk_analytics.render, title="Financial Risk Analytics", icon=":material/monitoring:", url_path="risk-analytics"),
    st.Page(company_report.render, title="Company Risk Report", icon=":material/summarize:", url_path="company-report"),
    st.Page(methodology.render, title="About & Methodology", icon=":material/menu_book:", url_path="methodology"),
]
st.session_state["_pages"] = {p.url_path: p for p in PAGES}
nav = st.navigation(PAGES, position="hidden")

with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-logo"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6l8-3z"/><path d="M8.5 13l2.5-2.5 2 2 3-3.5"/></svg></div>
    <div><div class="brand-name">FinGuard AI</div><div class="brand-sub">Risk Intelligence</div></div></div>
    <div class="nav-label">Workspace</div>""", unsafe_allow_html=True)
    for p in PAGES:
        st.page_link(p, label=p.title, icon=p.icon)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.toggle("Light mode", key="light_mode")
    st.markdown('<div class="side-foot">FinGuard AI is a machine-learning project.</div>', unsafe_allow_html=True)

try:
    app_data.artefacts()
except ArtefactsMissing as e:
    ui.page_header("Models not trained yet", "The application needs the trained model files before it can start.")
    ui.callout(f"{e}<br><br>Open a terminal in the project folder and run: <code>python train.py</code>", "bad")
    st.stop()

nav.run()
