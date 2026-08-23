#!/usr/bin/env python3
"""Parse BLS CPS Table 11 from its HTML into a tidy CSV.

Input : data/raw/cpsaat11_bls_2025.html  (the official BLS page, saved verbatim)
Output: data/bls/cpsaat11_2025.csv

Table 11 is "Employed people by detailed occupation, sex, race, and Hispanic or
Latino ethnicity", annual averages. The HTML carries the full table; the six
data columns are, in order: total employed (thousands), then percent of that
total who are women / White / Black or African American / Asian / Hispanic or
Latino. A "-" means BLS suppressed the estimate (base below their publication
threshold); it is written through to the CSV as "-" and never imputed.

`indent_level` preserves the table's own hierarchy (0/1 = major groups,
2/3 = detailed occupations), so aggregate rows can be excluded from analysis.

See data/PROVENANCE.md -- www.bls.gov returns HTTP 403 to every scripted
request, so the HTML has to be fetched by hand or from the Internet Archive.

Run:  venv/bin/python data/build_bls_table11.py
"""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "cpsaat11_bls_2025.html"
DST = ROOT / "data" / "bls" / "cpsaat11_2025.csv"

COLUMNS = ["total_employed_thousands", "pct_women", "pct_white",
           "pct_black_or_african_american", "pct_asian", "pct_hispanic_or_latino"]


def text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. See data/PROVENANCE.md for how to get it.")
        return 1

    s = SRC.read_text(encoding="utf-8", errors="replace")
    body = s[s.index("<tbody>"):s.index("</tbody>")]

    rows = []
    for chunk in re.split(r"<tr[^>]*>", body)[1:]:
        chunk = chunk.split("</tr>")[0]
        th = re.search(r"<th[^>]*>(.*?)</th>", chunk, re.S)
        if not th:
            continue
        lvl = re.search(r'class="sub(\d)"', th.group(1))
        vals = [text(v) for v in re.findall(r"<td[^>]*>(.*?)</td>", chunk, re.S)]
        rows.append([int(lvl.group(1)) if lvl else -1, text(th.group(1))]
                    + (vals + [""] * len(COLUMNS))[:len(COLUMNS)])

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["indent_level", "occupation"] + COLUMNS)
        w.writerows(rows)

    detailed = sum(1 for r in rows if r[0] >= 2)
    suppressed = sum(1 for r in rows if r[3] == "-")
    print(f"wrote {DST.relative_to(ROOT)}: {len(rows)} rows "
          f"({detailed} detailed occupations, {suppressed} with % women suppressed)")
    print(f"sanity check -- {rows[0][1]}: {rows[0][2]} thousand employed, "
          f"{rows[0][3]}% women")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
