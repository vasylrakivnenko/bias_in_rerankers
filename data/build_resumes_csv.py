#!/usr/bin/env python3
"""Slim the livecareer resume dump down to what the study needs.

Input : data/raw/livecareer_Resume_full.csv  (56 MB: ID, Resume_str, Resume_html, Category)
Output: data/resumes/resumes.csv             (15 MB: id, category, resume_text)

Drops the Resume_html column (unused, ~70% of the bytes) and collapses runs of
whitespace in Resume_str so the text is one clean line per resume.

Download the input first -- see data/PROVENANCE.md:
    curl -L -o data/raw/livecareer_Resume_full.csv \\
      https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv

Run:  venv/bin/python data/build_resumes_csv.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "livecareer_Resume_full.csv"
DST = ROOT / "data" / "resumes" / "resumes.csv"


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. See data/PROVENANCE.md for the download.")
        return 1

    csv.field_size_limit(sys.maxsize)
    DST.parent.mkdir(parents=True, exist_ok=True)

    cats: Counter[str] = Counter()
    with SRC.open(newline="", encoding="utf-8") as fh, \
            DST.open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["id", "category", "resume_text"])
        for row in csv.DictReader(fh):
            cats[row["Category"]] += 1
            w.writerow([row["ID"], row["Category"], " ".join(row["Resume_str"].split())])

    print(f"wrote {DST.relative_to(ROOT)}: {sum(cats.values())} resumes, "
          f"{len(cats)} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
