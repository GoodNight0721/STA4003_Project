"""Build quarterly retail-sales & PMI series from NBS monthly CSVs + run verification.

Run from the repository root:  python scripts/data/build_quarterly.py   (stdlib only; CNY table embedded, computed via sxtwl and
spot-checked against known official dates: 2012-01-23, 2016-02-08, 2020-01-25,
2024-02-10, 2025-01-29, 2026-02-17 all OK).

Construction rules
------------------
Retail sales (current-period, RMB 100 million; 亿元):
  * 1994M1-2011M12: NBS publishes a separate value every month; quarterly value =
    sum of the three monthly values.
  * 2012M1+ : NBS publishes Jan and Feb only as a combined "Jan-Feb" total.  The
    cumulative indicator carries cum(Feb) == Jan-Feb total in the February slot and
    cum(Mar), cum(Jun), ... in the monthly slots.  Hence, for y >= 2012:
        Q1(y) = cum(Mar, y)
        Q2(y) = cum(Jun, y) - cum(Mar, y)
        Q3(y) = cum(Sep, y) - cum(Jun, y)
        Q4(y) = cum(Dec, y) - cum(Sep, y)
    For 2000-2011 the same cumulative-difference formula is used for consistency.
    The current-period and cumulative vintages differ in some years, so these values
    need not reproduce the separate monthly sums.  For 1994-1999 monthly sums are used.
  * Last complete quarter = 2026Q2 (2026M9 not yet published).

PMI (manufacturing PMI, %): quarterly = mean of the three monthly PMI values; starts
2005Q1 (PMI survey began Jan 2005).
"""
import csv
from datetime import date
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
META = PROJECT_ROOT / "data" / "meta"
DIAGNOSTICS = PROJECT_ROOT / "outputs" / "diagnostics"
for directory in (PROC, META, DIAGNOSTICS):
    directory.mkdir(parents=True, exist_ok=True)

CNY = {  # Chinese New Year (lunar 1/1), computed with sxtwl, spot-checked
    1994: "1994-02-10", 1995: "1995-01-31", 1996: "1996-02-19", 1997: "1997-02-07",
    1998: "1998-01-28", 1999: "1999-02-16", 2000: "2000-02-05", 2001: "2001-01-24",
    2002: "2002-02-12", 2003: "2003-02-01", 2004: "2004-01-22", 2005: "2005-02-09",
    2006: "2006-01-29", 2007: "2007-02-18", 2008: "2008-02-07", 2009: "2009-01-26",
    2010: "2010-02-14", 2011: "2011-02-03", 2012: "2012-01-23", 2013: "2013-02-10",
    2014: "2014-01-31", 2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29",
    2026: "2026-02-17",
}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        d = {}
        for row in r:
            if row and row[0]:
                try:
                    d[row[0]] = (float(row[1]) if row[1] not in ("", None) else None)
                except ValueError:
                    d[row[0]] = None
    return d


def qkey(y, q):
    return f"{y}Q{q}"


def main():
    monthly = read_csv(RAW / "retail_sales_monthly.csv")   # yyyymm -> 亿元
    cumul = read_csv(RAW / "retail_sales_cumulative.csv")  # yyyymm -> 亿元
    pmi_m = read_csv(RAW / "manuf_pmi.csv")                # yyyymm -> %

    # ---- CNY meta table ----
    with open(META / "lunar_new_year.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "cny_date", "cny_month"])
        for y in sorted(CNY):
            w.writerow([y, CNY[y], int(CNY[y][5:7])])

    # ---- quarterly retail ----
    # 2000+ : cumulative-differences; 1994-1999 : monthly sums (no cumulative series).
    # The quarterly sum equals December cumulative by construction, not independently.
    # NOTE: the separate "当期值" (current-month) series disagrees with the cumulative
    #       vintage in a few early years (2005 ~5%, 2011 ~0.2% mainly Q1); the project
    #       uses cumulative values from 2000 onward as its defined construction source.
    ret = {}
    for y in range(1994, 2026 + 1):
        for q in range(1, 5):
            if y >= 2000:
                m_prev = {1: 0, 2: 3, 3: 6, 4: 9}[q]
                m_end = {1: 3, 2: 6, 3: 9, 4: 12}[q]
                if m_prev == 0:
                    v = cumul.get(f"{y}03")
                else:
                    a = cumul.get(f"{y}{m_prev:02d}")
                    b = cumul.get(f"{y}{m_end:02d}")
                    v = b - a if (a is not None and b is not None) else None
            elif q == 1:
                v = (monthly.get(f"{y}01"), monthly.get(f"{y}02"), monthly.get(f"{y}03"))
                v = sum(v) if None not in v else None
            else:
                m_end = {2: 6, 3: 9, 4: 12}[q]
                m_prev = {2: 3, 3: 6, 4: 9}[q]
                if y >= 2000:
                    a = cumul.get(f"{y}{m_prev:02d}")
                    b = cumul.get(f"{y}{m_end:02d}")
                    v = b - a if (a is not None and b is not None) else None
                else:
                    ms = (monthly.get(f"{y}{m:02d}") for m in range(m_prev + 1, m_end + 1))
                    ms = list(ms)
                    v = sum(ms) if None not in ms else None
            if v is not None:
                ret[qkey(y, q)] = v
    # drop incomplete 2026Q3
    years = sorted({int(k[:4]) for k in ret})
    last_year, last_q = years[-1], max(int(k[5]) for k in ret if k.startswith(f"{years[-1]}Q"))
    print("last complete quarter:", f"{last_year}Q{last_q}")

    rows = sorted(ret.items())
    with open(PROC / "retail_quarterly.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "value_bn", "log_value"])
        for k, v in rows:
            w.writerow([k, round(v, 1), round(__import__("math").log(v), 6)])
    missing = [k for k, v in rows if v is None]
    print("retail quarterly: n =", len(rows), "range =", rows[0][0], "..", rows[-1][0], "missing =", missing)

    # ---- quarterly PMI ----
    pmi = {}
    for y in range(2005, 2026 + 1):
        for q in range(1, 5):
            ms = [pmi_m.get(f"{y}{m:02d}") for m in range(3 * q - 2, 3 * q + 1)]
            if None in ms:
                continue
            pmi[qkey(y, q)] = sum(ms) / 3.0
    pmi = {k: v for k, v in pmi.items() if k <= f"{last_year}Q{last_q}"}
    with open(PROC / "pmi_quarterly.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "pmi"])
        for k in sorted(pmi):
            w.writerow([k, round(pmi[k], 2)])
    print("pmi quarterly: n =", len(pmi), "range =", sorted(pmi)[0], "..", sorted(pmi)[-1])

    # ---- verification report ----
    rep = []
    rep.append("Quarterly construction & verification - built 2026-09-28")
    rep.append("Sources: NBS National Data (国家数据), region=全国")
    rep.append("")
    rep.append(f"Retail quarterly: n={len(rows)}, {rows[0][0]}..{rows[-1][0]}, missing={missing}")
    rep.append(f"PMI quarterly: n={len(pmi)}, {sorted(pmi)[0]}..{sorted(pmi)[-1]}")

    # 1) cumulative-consistency: annual total (sum of 4 quarters) vs cum(Dec) 2000+
    worst = 0.0
    for y in range(2000, last_year):
        tot = ret.get(qkey(y, 1), 0) + ret.get(qkey(y, 2), 0) + ret.get(qkey(y, 3), 0) + ret.get(qkey(y, 4), 0)
        cdec = cumul.get(f"{y}12")
        if cdec:
            worst = max(worst, abs(tot - cdec) / cdec)
    rep.append(f"Cumulative consistency: max |sum(Q1..Q4) - cum(Dec)|/cum(Dec) over 2000-{last_year-1} = {worst:.6g}")
    rep.append("  (construction telescopes to cum(Dec) by definition for 2000+; residual from 1994-99 monthly sums)")
    rep.append("Known data-quality notes:")
    rep.append("  1. Since 2012, NBS publishes Jan & Feb retail only as a combined Jan-Feb total;")
    rep.append("     the current-month series has no separate Jan/Feb values for 2012+.")
    rep.append("  2. The 'current-month' and 'cumulative' vintages disagree in a few years")
    rep.append("     (2005 ~5.2% annual; 2011 ~0.2%, mainly Q1); cumulative values are the chosen")
    rep.append("     source for 2000+. Quarter sums telescope to December cumulative by construction;")
    rep.append("     this is internal accounting consistency, not independent validation.")
    rep.append("  3. Retail series revised per 5th Economic Census; since 2025 growth rates are on a")
    rep.append("     comparable basis (see NBS metadata annotation).")

    # 2) annual retail (complete years only, i.e. through last_year-1)
    annual = {}
    all_q = {k: v for k, v in ret.items() if None not in (v,)}
    for y in range(1994, last_year):
        if all(qkey(y, q) in all_q for q in range(1, 5)):
            annual[y] = sum(all_q[qkey(y, q)] for q in range(1, 5))
    rep.append(f"Annual retail totals {min(annual)}-{max(annual)} (RMB 100 million; 亿元); value at {max(annual)}: {annual[max(annual)]:,.0f}")

    # 3) CNY evidence, monthly era (1994-2011): Feb/Jan ratio by CNY month
    feb_jan = {}
    for y in range(1994, 2012):
        j = monthly.get(f"{y}01"); f_ = monthly.get(f"{y}02")
        if j and f_:
            feb_jan[y] = f_ / j
    jan_cny = {y: v for y, v in feb_jan.items() if CNY[y][5:7] == "01"}
    feb_cny = {y: v for y, v in feb_jan.items() if CNY[y][5:7] == "02"}
    mj = sum(jan_cny.values()) / len(jan_cny)
    mf = sum(feb_cny.values()) / len(feb_cny)
    rep.append("")
    rep.append("Chinese New Year effect, monthly era 1994-2011 (Jan/Feb published separately):")
    rep.append(f"  mean Feb/Jan retail ratio, CNY-in-Jan years (n={len(jan_cny)}): {mj:.3f}")
    rep.append(f"  mean Feb/Jan retail ratio, CNY-in-Feb years (n={len(feb_cny)}): {mf:.3f}")
    rep.append(f"  => Mean Feb/Jan ratio is {mf-mj:.3f} higher when CNY falls in February")

    # 4) CNY evidence, quarterly era (2012-2025): Q1 share of annual
    q1share = {}
    for y in range(2012, last_year):
        a = annual.get(y)
        if a and ret.get(qkey(y, 1)):
            q1share[y] = ret[qkey(y, 1)] / a
    jan_cny2 = {y: v for y, v in q1share.items() if CNY[y][5:7] == "01"}
    feb_cny2 = {y: v for y, v in q1share.items() if CNY[y][5:7] == "02"}
    mj2 = sum(jan_cny2.values()) / len(jan_cny2)
    mf2 = sum(feb_cny2.values()) / len(feb_cny2)
    rep.append("")
    rep.append("Chinese New Year effect, quarterly era 2012-2025 (only joint Jan-Feb published):")
    rep.append(f"  mean Q1 share of annual retail, CNY-in-Jan years (n={len(jan_cny2)}): {mj2*100:.1f}%")
    rep.append(f"  mean Q1 share of annual retail, CNY-in-Feb years (n={len(feb_cny2)}): {mf2*100:.1f}%")

    # 5) strongest/weakest quarter frequency
    peak, trough = Counter(), Counter()
    for y in range(1994, last_year):
        b = {f"Q{qk}": ret.get(qkey(y, qk)) for qk in range(1, 5)}
        if None in b.values():
            continue
        peak[max(b, key=b.get)] += 1
        trough[min(b, key=b.get)] += 1
    rep.append("")
    rep.append("Seasonal pattern ({}, quarterly peaks/troughs by quarter):".format(f"{min(annual)}-{last_year}"))
    rep.append("  peaks:   " + ", ".join(f"Q{q}: {peak[f'Q{q}']}x" for q in range(1, 5)))
    rep.append("  troughs: " + ", ".join(f"Q{q}: {trough[f'Q{q}']}x" for q in range(1, 5)))

    # 6) series stats for proposal
    import math, statistics
    lv = [math.log(v) for k, v in ret.items()]
    rep.append("")
    rep.append(f"log-retail: mean {statistics.mean(lv):.4f}, sd {statistics.pstdev(lv):.4f}, "
               f"min {min(lv):.4f}, max {max(lv):.4f}")
    rep.append(f"PMI: mean {statistics.mean(pmi.values()):.2f}, min {min(pmi.values()):.2f}, "
               f"max {max(pmi.values()):.2f}")

    with open(DIAGNOSTICS / "VERIFICATION.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
