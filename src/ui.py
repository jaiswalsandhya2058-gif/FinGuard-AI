"""Shared UI helpers: theme injection, KPI cards, header, tables, callouts."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import config as cfg

THEMES = {
    "dark": {"bg": "#0A1128", "side": "#0C1430", "card": "#111A36", "card2": "#16214A", "border": "#243055", "text": "#E6ECFA",
             "muted": "#8FA0BF", "input": "#0E1733", "hover": "rgba(59,130,246,.10)", "glow": "rgba(59,130,246,.16)",
             "shadow": "rgba(0,0,0,.45)", "accent": "#3B82F6"},
    "light": {"bg": "#F3F6FC", "side": "#FFFFFF", "card": "#FFFFFF", "card2": "#F6F8FD", "border": "#DAE2F0", "text": "#0F172A",
              "muted": "#5A6B8C", "input": "#FFFFFF", "hover": "rgba(37,99,235,.07)", "glow": "rgba(99,102,241,.10)",
              "shadow": "rgba(30,41,90,.14)", "accent": "#2563EB"},
}


def theme_name() -> str:
    return st.session_state.get("theme", "dark")


def inject_theme() -> None:
    t = THEMES[theme_name()]
    css_vars = ":root{" + "".join(f"--{k}:{v};" for k, v in t.items()) + "}"
    css = (cfg.ASSETS_DIR / "custom.css").read_text(encoding="utf-8")
    extra = ""
    if theme_name() == "light":
        # Streamlit's base theme is dark; recolour native widgets that ignore CSS variables.
        extra = """
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { color-scheme: light; }
        [data-testid="stDataFrame"] { filter: invert(1) hue-rotate(180deg); }
        [data-testid="stDataFrame"] canvas { opacity: .98; }
        [data-baseweb="popover"] *, [data-baseweb="menu"] * { color: #0F172A !important; }
        [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #fff !important; }
        [data-testid="stFileUploaderDropzone"] button { color: #0F172A !important; }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #5A6B8C !important; }
        [data-testid="stHeader"] * { color: #0F172A; }
        """
    st.markdown(f"<style>{css_vars}{css}{extra}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "FinGuard AI") -> None:
    st.markdown(f'<div class="ph"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{html.escape(title)}</div>', unsafe_allow_html=True)


def kpi_html(label: str, value: str, sub: str = "", accent: str = "#3B82F6") -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="k-label">{html.escape(label)}</div>'
            f'<div class="k-value">{html.escape(str(value))}</div><div class="k-sub">{html.escape(sub)}</div></div>')


def kpi_row(items: list[tuple], cols: int | None = None) -> None:
    """items: (label, value, sub, accent)"""
    cs = st.columns(cols or len(items))
    for c, it in zip(cs, items):
        c.markdown(kpi_html(*it), unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return (f'<span class="badge" style="color:{color};border-color:{color}66;background:{color}1F">{html.escape(text)}</span>')


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f'<div class="empty">{text}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">{html.escape(cfg.DISCLAIMER)}</div>', unsafe_allow_html=True)


def html_table(df: pd.DataFrame, formats: dict | None = None, num_cols: list[str] | None = None) -> None:
    """Small, theme-aware HTML table (used for compact summaries)."""
    formats = formats or {}
    num_cols = num_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        tds = ""
        for c in df.columns:
            v = r[c]
            s = formats[c](v) if c in formats else (f"{v:,.4f}" if isinstance(v, float) else str(v))
            raw = s if str(s).startswith("<span") else html.escape(str(s))
            tds += f'<td class="{"num" if c in num_cols else ""}">{raw}</td>'
        body += f"<tr>{tds}</tr>"
    st.markdown(f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def card(key: str, title: str, sub: str = ""):
    """Bordered card container with a title (use as `with ui.card(...):`)."""
    c = st.container(key=f"card-{key}")
    c.markdown(f'<h4 style="margin:0">{html.escape(title)}</h4><div style="color:var(--muted);font-size:.84rem;margin-bottom:4px">{html.escape(sub)}</div>', unsafe_allow_html=True)
    return c


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"
