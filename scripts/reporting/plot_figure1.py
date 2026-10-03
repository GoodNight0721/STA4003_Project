"""Generate Figure 1 from the processed quarterly retail-sales series."""
import csv, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_quarterly():
    with open(PROJECT_ROOT / "data" / "processed" / "retail_quarterly.csv",
              encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        dates, vals = [], []
        for row in r:
            if not row:
                continue
            q = row[0]  # e.g. 1994Q1
            y, qn = int(q[:4]), int(q[5])
            dates.append(date(y, 1 + 3 * (qn - 1), 1))
            vals.append(math.log(float(row[1])))
    return dates, vals


def main():
    dates, vals = read_quarterly()

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.plot(dates, vals, lw=1.3, color="#1f6fb2")
    ax.set_title("Quarterly total retail sales of consumer goods in China (log scale)",
                 fontsize=11, pad=8)
    ax.set_ylabel("log(quarterly retail sales)", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linewidth=0.6)

    marks = {
        date(2020, 1, 1): "2020Q1\n(COVID-19, -19.6% y/y)",
        date(2022, 4, 1): "2022Q2\n(Omicron lockdown)",
    }
    for d, lab in marks.items():
        i = dates.index(d)
        ax.annotate(lab, xy=(dates[i], vals[i]),
                    xytext=(dates[i], vals[i] - 0.25),
                    ha="center", fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
        ax.plot([dates[i]], [vals[i]], "o", ms=4, color="#d14f2b")

    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / "figures" / "Figure1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
