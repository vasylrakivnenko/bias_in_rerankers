#!/usr/bin/env python3
"""Build `data/category_to_soc.csv`: livecareer resume category -> O*NET SOC.

The mapping below is hand-made (24 rows). The SOC code and the confidence
judgement are typed here; the O*NET title and description are read from
`data/onet/Occupation Data.txt`, and the BLS row title / % women are read from
`data/bls/cpsaat11_2025.csv`, so no text or number is transcribed by hand.

`match_confidence` matters for the realistic-pool study (REVIEW_TODO.md G2):

  high   - the category names an occupation and the resumes in it are that job
  medium - the category names an occupation but the resumes vary around it
  low    - the category names an *industry*, not an occupation; the SOC is the
           modal job among those resumes, and category relevance is noisy

The corpus categories are noisy in an ordinary way -- a resume filed under CHEF
may be a nursing student's, one under AUTOMOBILE an ETL developer's. That noise
is a property of the corpus, not of this mapping, and the realistic-pool study
should treat category as a weak relevance label, not ground truth.

Run:  venv/bin/python data/build_category_to_soc.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONET = ROOT / "data" / "onet" / "Occupation Data.txt"
BLS = ROOT / "data" / "bls" / "cpsaat11_2025.csv"
RESUMES = ROOT / "data" / "resumes" / "resumes.csv"
OUT = ROOT / "data" / "category_to_soc.csv"

# category -> (O*NET-SOC code, confidence, BLS CPS Table 11 row title or "", note)
MAPPING = {
    "ACCOUNTANT":             ("13-2011.00", "high",   "Accountants and auditors", ""),
    "ADVOCATE":               ("23-1011.00", "medium", "Lawyers",
                               "mixed: attorneys, patient/customer advocates, HR advocates"),
    "AGRICULTURE":            ("11-9013.00", "medium", "Farmers, ranchers, and other agricultural managers",
                               "many are extension/advisory rather than farm operators"),
    "APPAREL":                ("41-2031.00", "low",    "Retail salespersons",
                               "industry label; modal job is apparel retail associate/cashier"),
    "ARTS":                   ("25-3021.00", "low",    "",
                               "industry label; modal job is arts educator/instructor or arts administrator"),
    "AUTOMOBILE":             ("49-3023.00", "medium", "Automotive service technicians and mechanics",
                               "industry label; includes drivers and service managers"),
    "AVIATION":               ("49-3011.00", "low",    "Aircraft mechanics and service technicians",
                               "industry label; mechanics, avionics technicians, supply and safety staff"),
    "BANKING":                ("13-2072.00", "low",    "",
                               "industry label; branch managers, analysts, specialists"),
    "BPO":                    ("43-4051.00", "medium", "Customer service representatives",
                               "business-process outsourcing; call-centre and operations roles"),
    "BUSINESS-DEVELOPMENT":   ("41-3091.00", "medium",
                               "Sales representatives of services, except advertising, insurance, financial services, and travel", ""),
    "CHEF":                   ("35-1011.00", "high",   "Chefs and head cooks", ""),
    "CONSTRUCTION":           ("11-9021.00", "medium", "Construction managers",
                               "modal header is construction manager; also labourers and inspectors"),
    "CONSULTANT":             ("13-1111.00", "medium", "Management analysts",
                               "includes IT and HR consultants"),
    "DESIGNER":               ("27-1024.00", "medium", "Other designers",
                               "modal header is graphic designer; also interior, floral, CAD, instructional"),
    "DIGITAL-MEDIA":          ("13-1161.00", "medium", "Market research analysts and marketing specialists",
                               "modal header is digital marketing manager/specialist"),
    "ENGINEERING":            ("17-2199.00", "medium", "Engineers, all other",
                               "spans disciplines; many are technicians or engineering managers"),
    "FINANCE":                ("13-2051.00", "medium", "Financial and investment analysts", ""),
    "FITNESS":                ("39-9031.00", "high",   "Exercise trainers and group fitness instructors", ""),
    "HEALTHCARE":             ("11-9111.00", "medium", "Medical and health services managers",
                               "industry label; administrative rather than clinical roles predominate"),
    "HR":                     ("13-1071.00", "high",   "Human resources workers", ""),
    "INFORMATION-TECHNOLOGY": ("15-1299.00", "medium", "Computer occupations, all other",
                               "modal header is information technology specialist/manager"),
    "PUBLIC-RELATIONS":       ("27-3031.00", "high",   "",
                               "PR specialists, directors and interns"),
    "SALES":                  ("41-4012.00", "medium", "Retail salespersons",
                               "account managers and field sales; BLS row is the nearest CPS category"),
    "TEACHER":                ("25-2021.00", "medium", "Elementary and middle school teachers",
                               "mostly K-12 classroom teachers"),
}


def main() -> int:
    onet = {r["O*NET-SOC Code"]: r for r in
            csv.DictReader(ONET.open(encoding="utf-8"), delimiter="\t")}
    bls = {r["occupation"]: r["pct_women"] for r in
           csv.DictReader(BLS.open(encoding="utf-8"))}

    csv.field_size_limit(sys.maxsize)
    counts = Counter(r["category"] for r in
                     csv.DictReader(RESUMES.open(encoding="utf-8")))

    bad = [c for c, (s, *_ ) in MAPPING.items() if s not in onet]
    bad += [f"{c}:BLS" for c, (_, _, b, _) in MAPPING.items() if b and b not in bls]
    if bad:
        print("ERROR: unresolved references:", bad)
        return 1
    if set(counts) != set(MAPPING):
        print("ERROR: category mismatch:", set(counts) ^ set(MAPPING))
        return 1

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_resumes", "soc_code", "onet_title",
                    "match_confidence", "bls_title", "bls_pct_women",
                    "note", "onet_description"])
        for cat in sorted(MAPPING):
            soc, conf, bls_title, note = MAPPING[cat]
            o = onet[soc]
            pct = bls.get(bls_title, "") if bls_title else ""
            w.writerow([cat, counts[cat], soc, o["Title"], conf, bls_title,
                        "" if pct in ("-", "") else pct, note, o["Description"]])

    print(f"wrote {OUT.relative_to(ROOT)}: {len(MAPPING)} categories, "
          f"{sum(counts.values())} resumes")
    print("confidence:", dict(Counter(v[1] for v in MAPPING.values())))
    print("bls % women filled for",
          sum(1 for c, (_, _, b, _) in MAPPING.items() if b and bls.get(b) not in (None, "-")),
          "of", len(MAPPING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
