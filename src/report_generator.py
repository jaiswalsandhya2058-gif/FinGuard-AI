"""Company risk report builders (CSV + PDF via ReportLab)."""
from __future__ import annotations

import io

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config as cfg

LIMITATIONS = [
    "Trained on one historical dataset (6,819 companies, 220 bankrupt); it may not generalise to other markets or periods.",
    "Only ~3% of training examples are bankrupt, so precision is modest: a flagged company is a candidate for review, not a verdict.",
    "The model sees financial ratios only - no management quality, industry cycles, macro conditions or fraud signals.",
    "Feature contributions describe the model's behaviour (association), not causes of bankruptcy.",
]


def report_dataframe(company: str, result: dict, expl: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    rows = [("Company", company), ("Prediction", result["Predicted Class"]),
            ("Bankruptcy probability", f"{result['Bankruptcy Probability']:.4f}"),
            ("Risk category", result["Risk Category"]), ("Algorithm", result["Model"]),
            ("Decision threshold", f"{result['Decision Threshold']:.3f}")]
    rows += [(f"Top feature {i + 1}", f"{r.feature} | value={r.value:.6g} | contribution={r.contribution:+.4f} ({r.unit})")
             for i, r in enumerate(expl.itertuples())]
    rows += [(f"Observation {i + 1}", n) for i, n in enumerate(notes)]
    rows += [(f"Limitation {i + 1}", l) for i, l in enumerate(LIMITATIONS)]
    rows.append(("Disclaimer", cfg.DISCLAIMER))
    return pd.DataFrame(rows, columns=["Field", "Value"])


def report_pdf(company: str, result: dict, expl: pd.DataFrame, notes: list[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"FinGuard AI Risk Report - {company}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#0A1128"), alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1D4ED8"), spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    risk_col = colors.HexColor(cfg.RISK_COLORS[result["Risk Category"]])
    el = [Paragraph("FinGuard AI - Company Risk Report", h1),
          Paragraph(f"Company: <b>{company}</b> &nbsp;|&nbsp; Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}", body), Spacer(1, 6)]
    summary = Table([["Prediction", result["Predicted Class"]], ["Bankruptcy probability", f"{result['Bankruptcy Probability'] * 100:.1f}%"],
                     ["Risk category", result["Risk Category"]], ["Algorithm", result["Model"]],
                     ["Decision threshold", f"{result['Decision Threshold']:.3f}"]], colWidths=[55 * mm, 110 * mm])
    summary.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
                                 ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")), ("TEXTCOLOR", (1, 2), (1, 2), risk_col),
                                 ("FONTNAME", (1, 0), (1, 2), "Helvetica-Bold"), ("PADDING", (0, 0), (-1, -1), 5)]))
    el += [summary, Paragraph("Top contributing features", h2)]
    data = [["Feature", "Value", "Contribution"]] + [[Paragraph(r.feature, small), f"{r.value:.4g}", f"{r.contribution:+.4f}"] for r in expl.itertuples()]
    t = Table(data, colWidths=[95 * mm, 35 * mm, 35 * mm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A1128")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")),
                           ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    el += [t, Paragraph(f"Contribution unit: {expl['unit'].iloc[0]}. Positive values push the estimate towards bankruptcy.", small),
           Paragraph("Financial risk observations", h2)] + [Paragraph("&bull; " + n.replace("&", "&amp;"), body) for n in notes]
    el += [Paragraph("Model limitations", h2)] + [Paragraph("&bull; " + l, body) for l in LIMITATIONS]
    el += [Spacer(1, 10), Paragraph(cfg.DISCLAIMER, small)]
    doc.build(el)
    return buf.getvalue()
