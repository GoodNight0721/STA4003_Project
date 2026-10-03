import urllib.request, json, csv
from datetime import date
from collections import Counter
from pathlib import Path

BASE = "https://data.stats.gov.cn/dg/website/publicrelease/web/external"
HDRS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://data.stats.gov.cn/dg/website/page.html",
    "Content-Type": "application/json;charset=UTF-8",
    "Accept": "application/json, text/plain, */*",
}
ROOT = "fc982599aa684be7969d7b90b1bd0e84"  # 月度数据 root
KEY = "000000000000"  # 全国

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
META = PROJECT_ROOT / "data" / "meta"
RAW.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)


def post_series(cid, indic_id, start="199401", end="202608"):
    payload = {
        "cid": cid,
        "id": indic_id,
        "da": KEY,
        "dt": "",
        "dts": [f"{start}MM-{end}MM"],
        "rootId": ROOT,
    }
    req = urllib.request.Request(
        BASE + "/getEsDataByIndicatorIdAndDa",
        data=json.dumps(payload).encode("utf-8"),
        headers=HDRS,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def clean(rows):
    out = []
    for x in rows:
        dt = x["dt"]  # e.g. 199401MM
        yyyym = dt[:6]
        try:
            v = float(x["v"]) if x.get("v") not in (None, "") else None
        except ValueError:
            v = None
        out.append((yyyym, v, x.get("dt_name", ""), x.get("unit", "")))
    out.sort()
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def missing_and_dupes(rows):
    months = [r[0] for r in rows if r[1] is not None]
    dups = [m for m, c in Counter(months).items() if c > 1]
    gaps = []
    if months:
        y0, m0 = int(months[0][:4]), int(months[0][4:])
        y1, m1 = int(months[-1][:4]), int(months[-1][4:])
        cursor = date(y0, m0, 1)
        end = date(y1, m1, 1)
        seen = set(months)
        while cursor <= end:
            key = cursor.strftime("%Y%m")
            if key not in seen:
                gaps.append(key)
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)
    return dups, gaps


def main():
    report = []
    today = date.today().isoformat()

    # ---- 社会消费品零售总额: 当期值 / 累计值 / 同比 ----
    spec = [
        ("d0cb882c7f27443ab6b3ef9421901961", "1142a3a03e9045959e606a21822641ac", "消费零售额-当期值(亿元)", "retail_sales_monthly.csv", "社会消费品零售总额当期值"),
        ("d0cb882c7f27443ab6b3ef9421901961", "260a1794443b43dd93a59928b12f38af", "消费零售额-累计值(亿元)", "retail_sales_cumulative.csv", "社会消费品零售总额累计值"),
        ("d0cb882c7f27443ab6b3ef9421901961", "aaac57d54d2e465d91bc9f3ea1a8618e", "消费零售额-同比(%)", "retail_sales_yoy.csv", "社会消费品零售总额同比增长"),
    ]
    for cid, ind, tag, fn, txt in spec:
        try:
            out = post_series(cid, ind, "199401", "202608")
            rows = clean(out.get("data") or [])
            write_csv(RAW / fn, ["yyyym", "value", "dt_name", "unit"], rows)
            vals = [r for r in rows if r[1] is not None]
            dups, gaps = missing_and_dupes(vals)
            first, last = (vals[0][0], vals[-1][0]) if vals else ("-", "-")
            report.append(f"[{tag}] rows={len(rows)} nonmissing={len(vals)} range={first}..{last} duplicates={dups} gaps={gaps}")
            print("saved", fn, "| n =", len(rows), "| nonmissing =", len(vals), "| range =", first, "..", last, "| gaps =", gaps)
        except Exception as e:
            report.append(f"[{tag}] ERROR {e!r}")
            print("ERR", fn, repr(e))

    # ---- 制造业PMI ----
    try:
        out = post_series("93ffbb1aa85740d3aa2618371508b606", "a09aa989bdcf4cffa2021795722eb916", "200501", "202608")
        rows = clean(out.get("data") or [])
        write_csv(RAW / "manuf_pmi.csv", ["yyyym", "value", "dt_name", "unit"], rows)
        vals = [r for r in rows if r[1] is not None]
        dups, gaps = missing_and_dupes(vals)
        first, last = (vals[0][0], vals[-1][0]) if vals else ("-", "-")
        report.append(f"[制造业PMI] rows={len(rows)} nonmissing={len(vals)} range={first}..{last} duplicates={dups} gaps={gaps}")
        print("saved manuf_pmi.csv | n =", len(rows), "| nonmissing =", len(vals), "| range =", first, "..", last, "| gaps =", gaps)
    except Exception as e:
        report.append(f"[制造业PMI] ERROR {e!r}")
        print("ERR pmi", repr(e))

    # ---- 核验报告 ----
    report.insert(0, f"NBS data fetch report - retrieved {today}")
    report.insert(1, "Source: National Bureau of Statistics of China, National Data (国家数据)")
    report.insert(2, "API: data.stats.gov.cn/dg/website/publicrelease/web/external/getEsDataByIndicatorIdAndDa, region=全国")
    with open(META / "FETCH_REPORT.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n--- report ---")
    print("\n".join(report))


if __name__ == "__main__":
    main()
