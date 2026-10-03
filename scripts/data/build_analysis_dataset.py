"""Build the canonical, one-row-per-quarter analysis input."""

import csv

if __package__:
    from .validate_data import (
        ANALYSIS_COLUMNS,
        ANALYSIS_PATH,
        PROJECT_ROOT,
        cny_position_fraction,
        load_validated_inputs,
        quarter_sequence,
    )
else:  # Support `python scripts/data/build_analysis_dataset.py` from the repository root.
    from validate_data import (
        ANALYSIS_COLUMNS,
        ANALYSIS_PATH,
        PROJECT_ROOT,
        cny_position_fraction,
        load_validated_inputs,
        quarter_sequence,
    )


def format_optional(value, decimals=2):
    return "" if value is None else f"{value:.{decimals}f}"


def build_analysis_rows(bundle):
    quarters = quarter_sequence("1994Q1", "2026Q2")
    rows = []
    for index, quarter in enumerate(quarters):
        year = int(quarter[:4])
        quarter_num = int(quarter[-1])
        retail = bundle["retail_processed"][quarter]
        cny_date = bundle["cny"][year]
        pmi = bundle["pmi_processed"].get(quarter)
        pmi_lag1 = bundle["pmi_processed"].get(quarters[index - 1]) if index >= 1 else None
        pmi_lag2 = bundle["pmi_processed"].get(quarters[index - 2]) if index >= 2 else None
        rows.append(
            {
                "quarter": quarter,
                "year": year,
                "quarter_num": quarter_num,
                "retail_bn": f"{retail['retail']:.1f}",
                "log_retail": f"{retail['log_retail']:.6f}",
                "is_q1": int(quarter_num == 1),
                "cny_date": cny_date.isoformat(),
                "cny_month": cny_date.month,
                "cny_position_fraction": f"{cny_position_fraction(cny_date):.12f}",
                "pmi": format_optional(pmi),
                "pmi_lag1": format_optional(pmi_lag1),
                "pmi_lag2": format_optional(pmi_lag2),
            }
        )
    return rows


def write_analysis_dataset(rows, path=ANALYSIS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    _, bundle = load_validated_inputs()
    rows = build_analysis_rows(bundle)
    write_analysis_dataset(rows)
    print(f"Built {len(rows)} rows: {ANALYSIS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
