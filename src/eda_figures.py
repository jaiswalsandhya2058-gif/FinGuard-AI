"""Static Matplotlib/Seaborn figures for the EDA gallery (transparent background so they suit both themes)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

FG = "#8FA0BF"


def _style():
    sns.set_theme(style="ticks", rc={"axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
                                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                                     "axes.edgecolor": "#3A4766", "grid.color": "#2A3556", "font.size": 10})


def _save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES_DIR / name, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)


def make_eda_figures(df: pd.DataFrame, audit: dict) -> None:
    _style()
    y = df[cfg.TARGET]
    X = df.drop(columns=[cfg.TARGET])

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 3.6))
    cnt = y.value_counts().sort_index()
    sns.barplot(x=["Non-Bankrupt", "Bankrupt"], y=cnt.values, palette=[cfg.PALETTE["blue"], cfg.PALETTE["red"]], ax=ax)
    for i, v in enumerate(cnt.values):
        ax.text(i, v, f"{v:,}\n({v / len(y) * 100:.1f}%)", ha="center", va="bottom")
    ax.set_ylabel("Companies"); ax.set_ylim(0, cnt.max() * 1.18); sns.despine()
    _save(fig, "class_distribution.png")

    # 2. Correlation heatmap of the 20 most predictive features
    top = list(audit["top_single_feature_auc"])[:20]
    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(X[top].corr(), cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .7},
                xticklabels=[t[:28] for t in top], yticklabels=[t[:34] for t in top])
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7); plt.setp(ax.get_yticklabels(), fontsize=7)
    _save(fig, "correlation_heatmap.png")

    # 3. Outlier box-plots (robust z-score) for the 10 features with most IQR outliers
    names = list(audit["outlier_share_top"])[:10]
    z = (X[names] - X[names].median()) / (X[names].quantile(.75) - X[names].quantile(.25)).replace(0, np.nan)
    z = z.clip(-30, 30)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.boxplot(data=z, orient="h", ax=ax, color=cfg.PALETTE["purple"], fliersize=1.5, linewidth=.8)
    ax.set_yticklabels([n[:34] for n in names], fontsize=8); ax.set_xlabel("Robust z-score ((x - median) / IQR), clipped at ±30")
    sns.despine(); _save(fig, "outlier_boxplots.png")

    # 4. Class-conditional distributions for top-6 predictive features
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, f in zip(axes.ravel(), list(audit["top_single_feature_auc"])[:6]):
        lo, hi = X[f].quantile(.005), X[f].quantile(.995)
        sns.kdeplot(X.loc[y == 0, f].clip(lo, hi), ax=ax, color=cfg.PALETTE["blue"], fill=True, alpha=.3, label="Non-Bankrupt", warn_singular=False)
        sns.kdeplot(X.loc[y == 1, f].clip(lo, hi), ax=ax, color=cfg.PALETTE["red"], fill=True, alpha=.3, label="Bankrupt", warn_singular=False)
        ax.set_title(f[:36], fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    axes[0, 0].legend(fontsize=8, frameon=False); plt.tight_layout(); _save(fig, "feature_distributions.png")


def make_model_figures(results: dict, extras: dict) -> None:
    """Static evaluation + explainability figures produced after training."""
    import shap
    _style()
    names = {k: cfg.MODELS[k]["short"] for k in results}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, r in results.items():
        axes[0].plot(r["evaluation"]["roc_curve"]["fpr"], r["evaluation"]["roc_curve"]["tpr"], color=cfg.MODEL_COLORS[k],
                     label=f"{names[k]} (AUC {r['evaluation']['roc_auc']:.3f})", lw=2)
        axes[1].plot(r["evaluation"]["pr_curve"]["recall"], r["evaluation"]["pr_curve"]["precision"], color=cfg.MODEL_COLORS[k],
                     label=f"{names[k]} (PR-AUC {r['evaluation']['pr_auc']:.3f})", lw=2)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#556", lw=1); axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    for a in axes: a.legend(frameon=False, fontsize=8)
    sns.despine(); _save(fig, "roc_pr_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (k, r) in zip(axes, results.items()):
        sns.heatmap(np.array(r["evaluation"]["tuned"]["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-B", "Bankrupt"], yticklabels=["Non-B", "Bankrupt"])
        ax.set_title(names[k]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); _save(fig, "confusion_matrices.png")

    for k, ex in extras.items():
        if "shap_values" in ex:
            plt.figure(figsize=(8, 6.5))
            shap.summary_plot(ex["shap_values"], ex["sample"], max_display=15, show=False, plot_size=None)
            _save(plt.gcf(), f"shap_summary_{k}.png")
