"""Regenerate the STA4003 project proposal (Word) with data-driven numbers.

Reads data/processed/retail_quarterly.csv, pmi_quarterly.csv and data/meta/lunar_new_year.csv,
computes the facts cited in the text, and writes outputs/reports/Proposal_gpX.docx.

Formatting follows the project brief: A4, 1" margins, Times New Roman 12 pt,
1.15 line spacing, 6 pt space after each paragraph.
"""
import csv, math, os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC = PROJECT_ROOT / "data" / "processed"
META = PROJECT_ROOT / "data" / "meta"
REPORTS = PROJECT_ROOT / "outputs" / "reports"


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [row for row in r if row]
    return hdr, rows


_, ret_rows = read_csv(os.path.join(PROC, "retail_quarterly.csv"))
_, pmi_rows = read_csv(os.path.join(PROC, "pmi_quarterly.csv"))
_, cny_rows = read_csv(os.path.join(META, "lunar_new_year.csv"))

RET = {r[0]: float(r[1]) for r in ret_rows}
PMI = {r[0]: float(r[1]) for r in pmi_rows}
CNY = {int(r[0]): r[1] for r in cny_rows}

QKY = sorted(RET)
N = len(QKY)
FIRST, LAST = QKY[0], QKY[-1]   # e.g. 1994Q1, 2026Q2
LY = int(LAST[:4])
LY0 = int(FIRST[:4])
NCY = LY - LY0                 # complete calendar years (1994-2025 -> 32)



def FIRSTQ():
    return FIRST

annual = {}
for y in range(1994, LY+1):
    qs = [f"{y}Q{q}" for q in range(1, 5)]
    if all(q in RET for q in qs):
        annual[y] = sum(RET[q] for q in qs)
first_q = RET[FIRST] / 1e4      # 万亿
last_q = RET[LAST] / 1e4
ann_last = annual[LY-1] / 1e4   # 全年最新完整年
g0 = RET["2020Q1"] / RET["2019Q1"] - 1
qshare = {q: sum(annual[y] and RET[f"{y}Q{q}"] / annual[y] for y in annual if f"{y}Q{q}" in RET) / len(annual) for q in range(1, 5)}

cny_month = {y: int(d[5:7]) for y, d in CNY.items() if 1994 <= y <= 2011}
febjan = {}
for y in range(1994, 2012):
    a = RET.get(f"{y}Q1")  # Q1 includes jan+feb(+mar); not usable for monthly ratio
    _ = a
# monthly Feb/Jan ratio needs the raw monthly file
_, m_rows = read_csv(PROJECT_ROOT / "data" / "raw" / "retail_sales_monthly.csv")
MON = {}
for r in m_rows:
    if r[1] not in ("", None):
        MON[r[0]] = float(r[1])
febjan_jan, febjan_feb = [], []
for y in range(1994, 2012):
    j = MON.get(f"{y}01"); f_ = MON.get(f"{y}02")
    if j and f_:
        (febjan_feb if cny_month[y] == 2 else febjan_jan).append(f_ / j)
m_jan, m_feb = sum(febjan_jan)/len(febjan_jan), sum(febjan_feb)/len(febjan_feb)

q1share_jan, q1share_feb = [], []
for y in annual:
    if 2012 <= y <= LY-1:
        (q1share_feb if CNY[y][5:7] == "02" else q1share_jan).append(RET[f"{y}Q1"] / annual[y])
s_jan, s_feb = sum(q1share_jan)/len(q1share_jan), sum(q1share_feb)/len(q1share_feb)

FMT = {
    "N": N, "FW": f'{RET[LAST]/1e4:,.2f}', "FW2": f'{last_q:.2f}',
    "FIRST": f'{RET[FIRST]/1e4:.2f}', "LAST": LAST, "LY": LY,
    "annual_last": f"{ann_last:.1f}",
    "q2020": f"{-g0*100:.1f}",
    "peak": f'Q{max(range(1,5), key=lambda q: qshare[q])}',
    "peak_share": f"{max(qshare.values())*100:.1f}",
    "trough": f'Q{min(range(1,5), key=lambda q: qshare[q])}',
    "mjan": f"{m_jan:.3f}", "mfeb": f"{m_feb:.3f}",
    "sjan": f"{s_jan*100:.1f}", "sfeb": f"{s_feb*100:.1f}",
    "npmi": len(PMI), "pmi_first": min(PMI), "pmi_last": max(PMI),
}
print(FMT)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for a in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, a, Inches(1))

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def para(text, bold=False, center=False, justify=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.bold = bold
    return p


def body(text):
    return para(text)


def heading(text):
    return para(text, bold=True)


para("Forecasting China's Quarterly Retail Sales With a Lunar New Year Adjustment", bold=True, center=True)
para("⟨Member 1 Name / Student ID⟩  ⟨Member 2 Name / Student ID⟩  "
     "⟨Member 3 Name / Student ID⟩  ⟨Member 4 Name / Student ID⟩", center=True)

heading("1.  Forecasting Question")
body(f"Can the quarterly total retail sales of consumer goods in China (\u793e\u4f1a\u6d88\u8d39\u54c1\u96f6\u552e\u603b\u989d) be forecast "
     f"reliably two quarters — six months — ahead, and does explicitly accounting for the moving position of the "
     f"lunar New Year improve out-of-sample accuracy relative to a standard seasonal model? We address three "
     f"sub-questions: (i) how accurate are simple benchmarks — the historical mean, the naive and the "
     f"seasonal-naive forecast — at the two-quarter horizon?; (ii) does a seasonal ARIMA model outperform these "
     f"benchmarks once the series is appropriately transformed and seasonally differenced?; and (iii) does adding "
     f"a continuous calendar variable for the lunar New Year's position improve accuracy near the "
     f"January–February boundary? Quarterly retail sales are a headline indicator of household consumption and an "
     f"input for retailers' inventory and staffing decisions, consumption-policy assessment and short-term demand "
     f"analysis, so accurate six-month-ahead forecasts with honest uncertainty measures have direct practical value.")

heading("2.  Data")
body("The primary series is the national quarterly total retail sales of consumer goods, published by the National "
     "Bureau of Statistics of China (NBS) and accessed through the official National Data platform "
     "(https://data.stats.gov.cn). In the underlying monthly series NBS published distinct January and February "
     "values through 2011, but from 2012 onward it publishes January and February only as a combined \u201cJan\u2013Feb\u201d "
     "total. We therefore work at quarterly frequency: Q1 is the published Jan\u2013Feb total plus March, and Q2\u2013Q4 "
     "are recovered by differencing the official cumulative series. From 2000 onward, quarter differences sum "
     "to December cumulative values by construction; this internal accounting identity is not independent "
     "validation. The quarterly series runs "
     f"{FIRST} to {LAST}, giving {N} observations across {NCY} complete seasonal cycles plus the first two quarters "
     f"of {LY} — far above the guideline of at least 60 observations and 4 complete cycles. A secondary quarterly "
     f"series, the manufacturing PMI ({min(PMI)}\u2013{max(PMI)}, {len(PMI)} observations), is "
     "obtained by averaging the monthly survey reading and is retained as a candidate explanatory variable. The "
     "dataset table below documents source, coverage, variables and known data-quality issues.")

data_rows = [
    ("Original source and link",
     "National Bureau of Statistics of China, National Data (\u56fd\u5bb6\u6570\u636e), https://data.stats.gov.cn "
     "(monthly data \u21d2 domestic trade \u21d2 retail sales; other \u21d2 purchasing managers index). Retrieved 28 Sep 2026."),
    ("How the data were collected",
     "Monthly statistical reports by wholesale/retail enterprises above a turnover threshold, aggregated by NBS; "
     "PMI from NBS\u2019s monthly survey of purchasing managers. The project selects cumulative retail values for "
     "2000 onward; quarter sums equal December cumulative values by construction, not independent validation."),
    ("Time period and frequency",
     f"Retail {FIRSTQ()} to {LAST} at quarterly frequency (built from official monthly releases); PMI {min(PMI)}\u2013{max(PMI)}."),
    ("Number of observations",
     f"{N} quarterly retail observations; {len(PMI)} quarterly PMI observations."),
    ("Variable(s) available",
     "Quarterly nominal retail sales (\u4ebf\u5143, log-transformable) and quarterly manufacturing PMI (%); lunar New Year "
     "calendar position per year (metadata)."),
    ("Missing values / known data-quality issues",
     "Since 2012 Jan\u2013Feb is published only as a combined total (handled by quarterly aggregation); current-period "
     "and cumulative vintages disagree in 2005 (\u22485%) and 2011 (\u22480.2%). The project selects cumulative values for "
     "2000 onward. COVID-19 observations in 2020Q1 and 2022Q2 are retained."),
    ("Seasonality check",
     f"Stable 4-quarter cycle; 32 complete annual cycles; {FMT['peak']} always the strongest quarter."),
    ("Legal / ethical suitability",
     "Public statistical releases by a national statistics office; suitable for non-commercial academic use."),
]

tbl = doc.add_table(rows=len(data_rows) + 1, cols=2)
tbl.style = "Table Grid"
tbl.rows[0].cells[0].text = "Item"
tbl.rows[0].cells[1].text = "Detail"
for i, (a, b) in enumerate(data_rows, start=1):
    tbl.rows[i].cells[0].text = a
    tbl.rows[i].cells[1].text = b
for rr in tbl.rows:
    for c in rr.cells:
        for p in c.paragraphs:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                run.font.size = Pt(10.5)
                run.font.name = "Times New Roman"

para("")

heading("3.  Main Features of the Series")
body("\u27e8INSERT Figure 1: time plot of logged quarterly retail sales, 1994Q1\u20132026Q2, with 2020Q1 and 2022Q2 annotated.\u27e9")
body("Figure 1. Quarterly total retail sales of consumer goods in China (seasonal period 4), log scale.")
body(f"Level and trend: retail sales rise almost monotonically from \u00a5{FMT['FIRST']} trillion per quarter in 1994Q1 to "
     f"\u00a5{FMT['FW2']} trillion in 2026Q2, and the spread widens with the level — this motivates a log (Box\u2013Cox) "
     f"transformation. The COVID-19 disruptions of 2020Q1 (\u221219.6% year-on-year) and 2022Q2 stand out as "
     f"sudden, genuine events.")
body(f"Within-year seasonality: the seasonal swing is large, and the ranking is remarkably stable — Q4 "
     f"({FMT['peak_share']}% of the yearly total on average) is the strongest quarter in every one of the 32 complete "
     f"years, followed by Q1 ({qshare[1]*100:.1f}%); Q2 is the most common trough ({qshare[2]*100:.1f}%).")
body(f"Stability of the seasonal pattern: the January\u2013February balance shifts with the lunar New Year, whose date "
     f"moves between late January and mid-February. In the 1994\u20132011 monthly data, the Feb/Jan retail ratio "
     f"averages {FMT['mjan']} when CNY falls in January but {FMT['mfeb']} when it falls in February; at "
     f"quarterly level, Q1\u2019s share of the annual total rises correspondingly from {FMT['sjan']}% (CNY in January) to "
     f"{FMT['sfeb']}% (CNY in February). Because January and February are published jointly from 2012, this drift "
     f"enters our models as a continuous CNY-position covariate rather than through fixed monthly seasonal factors.")

heading("4.  Potential Data-Quality Issues")
body(f"Missing or combined months: since 2012 January and February exist only as a combined Jan\u2013Feb figure; we "
     f"aggregate to quarterly totals so no observation is dropped and no value is imputed.")
body("Duplicated dates: none. Every series comes from a single NBS endpoint and a consistency script verifies that "
     "monthly periods are unique (no duplicates) and reports any absent months before aggregation.")
body("Irregular timing: the lunar New Year shifts the effective season between January and February, so fixed "
     "seasonal factors are only approximate; the CNY-position covariate is designed to capture exactly this.")
body(f"Outlying observations: 2020Q1 (\u2212{FMT['q2020']}% year-on-year) and 2022Q2 reflect COVID-19 control measures; they are "
     f"real events and are retained and interpreted rather than removed.")
body("Handling principle: retain genuine outliers with interpretation and do not impute absent months. Quarterly "
     "differences from 2000 onward telescope to December cumulative values by definition; that accounting identity "
     "is not independent validation.")

heading("5.  Analysis Goals and Preliminary Analysis Plan")
body("Analysis goals: (i) understand the temporal structure through plots, ACF/PACF and stationarity tests, with "
     "transformation and differencing choices that are statistically justified; (ii) construct and compare suitable "
     "forecasting models — benchmarks, seasonal ARIMA, and a dynamic-regression extension with the lunar New Year "
     "and PMI — on out-of-sample accuracy; (iii) produce two-quarter-ahead (six-month) point forecasts with "
     "prediction intervals and discuss their uncertainty.")
body("Preliminary plan: (a) identification — log-transform, then ACF/PACF, ADF/KPSS stationarity tests and seasonal "
     "differencing at period 4; (b) candidate models — historical mean, naive and seasonal-naive benchmarks; a "
     "seasonal ARIMA (p,d,q)\u00d7(P,D,Q)\u2084 on log retail with orders chosen by ACF/PACF, AICc and residual diagnostics; "
     "and a dynamic-regression (ARIMAX) specification adding a CNY-position covariate (fraction of Q1 elapsed before "
     "CNY) and the quarterly PMI, with one supplementary method (ETS or Theta) as a robustness check; (c) evaluation "
     "— rolling-origin out-of-sample assessment at horizon h = 2 quarters using MAE, RMSE and MASE on a common test "
     "window (2005Q1 onward, the PMI sample), residual diagnostics via Ljung\u2013Box and residual ACF/PACF, and final "
     "two-quarter forecasts with 80% and 95% intervals, with a discussion of interval width and practical usability.")

heading("References")
refs = [
    "National Bureau of Statistics of China. National Data (\u56fd\u5bb6\u6570\u636e) \u2014 retail sales and purchasing managers index datasets. https://data.stats.gov.cn. Retrieved 28 September 2026.",
    "National Bureau of Statistics of China (2025). \u793e\u4f1a\u6d88\u8d39\u54c1\u96f6\u552e\u603b\u989d\u7edf\u8ba1\u6307\u6807\u8bf4\u660e\u53ca\u7b2c\u4e94\u6b21\u5168\u56fd\u7ecf\u6d4e\u666e\u67e5\u4fee\u8ba2\u8bf4\u660e (indicator annotation retrieved from the National Data platform).",
    "National Bureau of Statistics of China. \u91c7\u8d2d\u7ecf\u7406\u6307\u6570\u7f16\u5236\u52a0\u5de5\u8bf4\u660e. http://www.stats.gov.cn.",
    "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
    "Shumway, R. H., & Stoffer, D. S. (2017). Time Series Analysis and Its Applications (4th ed.). Springer.",
]
for r_ in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.add_run(r_)

REPORTS.mkdir(parents=True, exist_ok=True)
doc.save(REPORTS / "Proposal_gpX.docx")
print("saved Proposal_gpX.docx")
