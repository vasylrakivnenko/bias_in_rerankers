#!/usr/bin/env python3
"""Build `data/name_gender.csv` from the SSA national baby-name data.

Input : data/raw/ssa_names/yob????.txt  (from https://www.ssa.gov/oact/babynames/names.zip)
Output: data/name_gender.csv

Restricted to birth years 1940-2005: people born in that window make up
essentially the whole current U.S. working-age population, so the gender
association of a name on a resume is the association it had for that cohort,
not the one it has for newborns (Ashley, Avery, Riley and friends have moved a
long way since 1940).

Columns:
    name         given name as SSA spells it (title case)
    pct_female   100 * female births / all births with that name in the window
    n            total births with that name in the window (both sexes)
    peak_decade  decade of that name's peak births -- lets the realistic-pool
                 study match a substituted name on era as well as on frequency

Only names with >= 95% one-gender usage are kept (pct_female >= 95 or <= 5),
and only names with n >= 100 in the window, so a substituted name is both
unambiguously gendered and actually plausible on a resume.

Run:  venv/bin/python data/build_name_gender.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "ssa_names"
OUT = ROOT / "data" / "name_gender.csv"

YEAR_MIN, YEAR_MAX = 1940, 2005
MIN_N = 100
MIN_PURITY = 95.0


def main() -> int:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"F": 0, "M": 0})
    by_decade: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    files = sorted(SRC.glob("yob*.txt"))
    if not files:
        print(f"ERROR: no yob*.txt under {SRC}; see data/PROVENANCE.md")
        return 1

    used = 0
    for path in files:
        year = int(path.stem[3:])
        if not (YEAR_MIN <= year <= YEAR_MAX):
            continue
        used += 1
        decade = year - year % 10
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                name, sex, num = line.rstrip("\n").split(",")
                num = int(num)
                counts[name][sex] += num
                by_decade[name][decade] += num

    rows = []
    for name, c in counts.items():
        n = c["F"] + c["M"]
        if n < MIN_N:
            continue
        pct_f = 100.0 * c["F"] / n
        if not (pct_f >= MIN_PURITY or pct_f <= 100.0 - MIN_PURITY):
            continue
        peak = max(by_decade[name].items(), key=lambda kv: kv[1])[0]
        rows.append((name, round(pct_f, 2), n, peak))

    rows.sort(key=lambda r: -r[2])
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "pct_female", "n", "peak_decade"])
        w.writerows(rows)

    f = sum(1 for r in rows if r[1] >= MIN_PURITY)
    print(f"read {used} yob files ({YEAR_MIN}-{YEAR_MAX}); {len(counts)} distinct names")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} names "
          f"({f} female-typed, {len(rows) - f} male-typed)")
    print("top 10 by births:")
    for r in rows[:10]:
        print(f"  {r[0]:<12} pct_female={r[1]:>6}  n={r[2]:>9,}  peak={r[3]}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
