"""Plotly figure factory. Every figure is themed through `style()`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg

_THEME = {"dark": True}
BLUE, CYAN, PURPLE, RED, GREEN, AMBER = (cfg.PALETTE[k] for k in ("blue", "cyan", "purple", "red", "green", "amber"))
METRIC_COLORS = [BLUE, CYAN, PURPLE, GREEN, AMBER]


def set_theme(name: str) -> None:
    _THEME["dark"] = name == "dark"


def style(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    dark = _THEME["dark"]
    fg, grid = ("#C9D4EC", "rgba(143,160,191,.16)") if dark else ("#334155", "rgba(51,65,85,.12)")
    fig.update_layout(height=height, template="plotly_dark" if dark else "plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, sans-serif", color=fg, size=12),
                      margin=dict(l=10, r=10, t=44 if title else 16, b=10), showlegend=legend,
                      title=dict(text=title, font=dict(size=14), x=0.01) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#0A1128" if dark else "#FFFFFF", font_color="#E6ECFA" if dark else "#0F172A"))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont_color=fg, title_font_color=fg)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont_color=fg, title_font_color=fg)
    return fig


def class_donut(n_safe: int, n_bankrupt: int) -> go.Figure:
    fig = go.Figure(go.Pie(labels=["Non-Bankrupt", "Bankrupt"], values=[n_safe, n_bankrupt], hole=.68, sort=False,
                           marker=dict(colors=[BLUE, RED], line=dict(color="rgba(0,0,0,0)", width=0)), textinfo="percent",
                           hovertemplate="%{label}: %{value:,} companies<extra></extra>"))
    fig.add_annotation(text=f"<b>{n_safe + n_bankrupt:,}</b><br><span style='font-size:11px'>companies</span>", showarrow=False, font=dict(size=20))
    return style(fig, 330)


def model_bars(df: pd.DataFrame, metrics: list[str], height: int = 360) -> go.Figure:
    fig = go.Figure()
    for m, c in zip(metrics, METRIC_COLORS):
        fig.add_bar(name=m, x=df["Short"], y=df[m], marker_color=c, text=df[m].map(lambda v: f"{v:.2f}"), textposition="outside",
                    hovertemplate="%{x}<br>" + m + ": %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=.28, bargroupgap=.06)
    fig.update_yaxes(range=[0, 1.12], title=None)
    return style(fig, height)


def radar(df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    fig = go.Figure()
    for _, r in df.iterrows():
        vals = [r[m] for m in metrics] + [r[metrics[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=metrics + [metrics[0]], name=r["Short"], fill="toself", opacity=.55,
                                      line=dict(color=cfg.MODEL_COLORS[r["key"]], width=2)))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="rgba(143,160,191,.2)"),
                                 angularaxis=dict(gridcolor="rgba(143,160,191,.2)")))
    return style(fig, 380)


def single_bar(x, y, colors, title: str, fmt: str = "{:.2f}", height: int = 320, horizontal: bool = False) -> go.Figure:
    txt = [fmt.format(v) for v in y]
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y, marker_color=colors, text=txt, textposition="outside",
                           orientation="h" if horizontal else "v"))
    fig.update_layout(bargap=.45)
    return style(fig, height, legend=False, title=title)


def roc_fig(results: dict) -> go.Figure:
    fig = go.Figure()
    for k, r in results.items():
        e = r["evaluation"]
        fig.add_scatter(x=e["roc_curve"]["fpr"], y=e["roc_curve"]["tpr"], mode="lines", name=f"{cfg.MODELS[k]['short']} (AUC {e['roc_auc']:.3f})",
                        line=dict(color=cfg.MODEL_COLORS[k], width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748B"), name="Chance", showlegend=False)
    fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
    return style(fig, 400, title="ROC curve (test set)")


def pr_fig(results: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    for k, r in results.items():
        e = r["evaluation"]
        fig.add_scatter(x=e["pr_curve"]["recall"], y=e["pr_curve"]["precision"], mode="lines",
                        name=f"{cfg.MODELS[k]['short']} (PR-AUC {e['pr_auc']:.3f})", line=dict(color=cfg.MODEL_COLORS[k], width=3))
    fig.add_hline(y=prevalence, line_dash="dash", line_color="#64748B", annotation_text="No-skill baseline", annotation_position="bottom right")
    fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1.02])
    return style(fig, 400, title="Precision-Recall curve (test set)")


def calibration_fig(results: dict) -> go.Figure:
    fig = go.Figure()
    for k, r in results.items():
        c = r["evaluation"]["calibration"]
        fig.add_scatter(x=c["predicted"], y=c["observed"], mode="lines+markers", name=cfg.MODELS[k]["short"], line=dict(color=cfg.MODEL_COLORS[k], width=2.5))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748B"), showlegend=False)
    fig.update_xaxes(title="Mean predicted probability"); fig.update_yaxes(title="Observed bankruptcy rate")
    return style(fig, 400, title="Calibration (quantile bins)")


def confusion_fig(cm, title: str) -> go.Figure:
    cm = np.array(cm)
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    z = cm / cm.sum(axis=1, keepdims=True)
    text = [[f"<b>{cm[i, j]:,}</b><br><span style='font-size:10px'>{labels[i][j]}</span>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=["Non-Bankrupt", "Bankrupt"], y=["Non-Bankrupt", "Bankrupt"], text=text, texttemplate="%{text}",
                               colorscale=[[0, "#111A36"], [1, BLUE]], showscale=False, hovertemplate="Row-normalised: %{z:.1%}<extra></extra>", xgap=3, ygap=3))
    fig.update_xaxes(title="Predicted"); fig.update_yaxes(title="Actual", autorange="reversed")
    return style(fig, 320, legend=False, title=title)


def threshold_sweep(y_true, proba, current: float) -> tuple[go.Figure, pd.DataFrame]:
    ths = np.linspace(0.02, 0.98, 49)
    rows = []
    for t in ths:
        pred = proba >= t
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append((t, p, r, 2 * p * r / (p + r) if p + r else 0.0))
    d = pd.DataFrame(rows, columns=["t", "Precision", "Recall", "F1"])
    fig = go.Figure()
    for m, c in zip(["Precision", "Recall", "F1"], [CYAN, GREEN, PURPLE]):
        fig.add_scatter(x=d["t"], y=d[m], mode="lines", name=m, line=dict(color=c, width=3))
    fig.add_vline(x=current, line_dash="dash", line_color=AMBER, annotation_text="selected threshold")
    fig.update_xaxes(title="Decision threshold"); fig.update_yaxes(range=[0, 1.02])
    return style(fig, 380, title="Precision / recall trade-off"), d


def importance_bar(df: pd.DataFrame, color: str, title: str) -> go.Figure:
    d = df.sort_values("importance")
    fig = go.Figure(go.Bar(x=d["importance"] * 100, y=d["feature"].str.slice(0, 48), orientation="h", marker=dict(color=d["importance"], colorscale=[[0, "#1E3A8A"], [1, color]]),
                           hovertemplate="%{y}<br>%{x:.2f}% of total importance<extra></extra>"))
    fig.update_xaxes(title="Share of total importance (%)")
    return style(fig, max(380, 24 * len(d) + 80), legend=False, title=title)


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    short = [c[:34] for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short, zmin=-1, zmax=1, colorscale="RdBu_r", xgap=1, ygap=1,
                               colorbar=dict(thickness=12, len=.8), hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(tickangle=50, tickfont=dict(size=9)); fig.update_yaxes(tickfont=dict(size=9))
    return style(fig, 640, legend=False)


def gauge(prob: float, threshold: float, risk: str) -> go.Figure:
    col = cfg.RISK_COLORS[risk]
    bands = [(0, threshold / 2, GREEN), (threshold / 2, threshold, AMBER), (threshold, (1 + threshold) / 2, "#F97316"), ((1 + threshold) / 2, 1, RED)]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=44, color=col), valueformat=".1f"),
                                 gauge=dict(axis=dict(range=[0, 100], tickwidth=0, ticksuffix="%"), bar=dict(color=col, thickness=.28),
                                            bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                            steps=[dict(range=[a * 100, b * 100], color=c + "40") for a, b, c in bands],
                                            threshold=dict(line=dict(color="#E6ECFA" if _THEME["dark"] else "#0F172A", width=3), thickness=.85, value=threshold * 100))))
    return style(fig, 270, legend=False)


def contribution_bar(expl: pd.DataFrame, title: str = "Top contributing features") -> go.Figure:
    d = expl.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["contribution"], y=d["feature"].str.slice(0, 46), orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in d["contribution"]],
                           customdata=np.c_[d["value"], d["healthy_median"], d["bankrupt_median"]],
                           hovertemplate="%{y}<br>value=%{customdata[0]:.4g}<br>non-bankrupt median=%{customdata[1]:.4g}<br>bankrupt median=%{customdata[2]:.4g}<br>contribution=%{x:+.4f}<extra></extra>"))
    fig.update_xaxes(title=f"{expl['unit'].iloc[0]}  (red raises risk, green lowers it)")
    return style(fig, max(340, 34 * len(d) + 90), legend=False, title=title)


def risk_hist(proba: np.ndarray, y_true: np.ndarray | None, threshold: float) -> go.Figure:
    fig = go.Figure()
    if y_true is None:
        fig.add_histogram(x=proba, nbinsx=40, marker_color=BLUE, name="Companies")
    else:
        fig.add_histogram(x=proba[y_true == 0], nbinsx=40, marker_color=BLUE, name="Non-Bankrupt", opacity=.8)
        fig.add_histogram(x=proba[y_true == 1], nbinsx=40, marker_color=RED, name="Bankrupt", opacity=.85)
        fig.update_layout(barmode="overlay")
    fig.add_vline(x=threshold, line_dash="dash", line_color=AMBER, annotation_text="decision threshold")
    fig.update_yaxes(type="log", title="Companies (log scale)"); fig.update_xaxes(title="Predicted bankruptcy probability", range=[0, 1])
    return style(fig, 340)


def risk_category_bar(counts: pd.Series, title: str | None = None) -> go.Figure:
    counts = counts.reindex(cfg.RISK_LEVELS).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[cfg.RISK_COLORS[k] for k in counts.index], text=counts.values.astype(int), textposition="outside"))
    fig.update_layout(bargap=.4)
    return style(fig, 320, legend=False, title=title)


def class_box(df: pd.DataFrame, feature: str, clip: tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    for cls, name, c in [(0, "Non-Bankrupt", BLUE), (1, "Bankrupt", RED)]:
        v = df.loc[df[cfg.TARGET] == cls, feature].clip(*clip)
        fig.add_violin(y=v, name=name, box_visible=True, meanline_visible=True, line_color=c, fillcolor=c + "55", points=False)
    fig.update_yaxes(title=feature[:60])
    return style(fig, 400, legend=False)
