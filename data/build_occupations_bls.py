#!/usr/bin/env python3
"""Build `code/occupations_bls.csv` from BLS CPS Table 11 (2025 annual averages).

Input : data/bls/cpsaat11_2025.csv   (parsed verbatim from the BLS HTML table;
                                      see data/PROVENANCE.md for the source)
Output: code/occupations_bls.csv     (schema documented in code/labels.py)

The occupation -> BLS-row mapping below is *hand-made*. Every entry names the
verbatim BLS CPS Table 11 row title; the percentage and the year are then read
from the parsed table, never typed in. An entry of None means "BLS has no
occupation of this name" and produces label=drop with a blank pct_women.

Mapping rule (see data/PROVENANCE.md for the full write-up and for the
alternatives considered for every judgement call):

  (a) exact          - a BLS row title *is* the occupation ("Electricians")
  (b) residual       - a BLS row is the catch-all for it ("Engineers, all other")
  (c) largest kind   - the largest-employment BLS row naming a *kind* of that
                       occupation ("Elementary and middle school teachers")
  (d) synonym        - a documented common synonym ("CEO" -> "Chief executives")
  (e) drop (None)    - BLS has no occupation of that name: the word names a
                       rank/function rather than an occupation (boss, chief,
                       supervisor, executive, assistant, attendant, researcher,
                       financier, protege), it is not a civilian occupation
                       (warrior, philosopher, maestro, skipper, socialite,
                       homemaker, magician, fighter pilot), or the BLS estimate
                       is suppressed for small sample size (captain, midwife,
                       broadcaster).

Label rule (REVIEW_TODO.md A8): male if pct_women < 30, female if pct_women > 70,
neutral if 40 <= pct_women <= 60, otherwise `drop`. No BLS row -> `drop`.

Run:  venv/bin/python data/build_occupations_bls.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLS_CSV = ROOT / "data" / "bls" / "cpsaat11_2025.csv"
ONET_OCC = ROOT / "data" / "onet" / "Occupation Data.txt"
OUT_CSV = ROOT / "code" / "occupations_bls.csv"
YEAR = 2025

# occupation (as it appears in synthetic_dataset.OCCUPATIONS) -> verbatim BLS row title
MAPPING: dict[str, str | None] = {
    # ---- originally labelled "male" ------------------------------------
    "engineer": "Engineers, all other",                                    # (b)
    "mechanic": "Automotive service technicians and mechanics",            # (c)
    "CEO": "Chief executives",                                             # (d)
    "captain": "Ship and boat captains and operators",                     # (c) estimate suppressed
    "programmer": "Computer programmers",                                  # (c)
    "electrician": "Electricians",                                         # (a)
    "plumber": "Plumbers, pipefitters, and steamfitters",                  # (c)
    "surgeon": "Surgeons",                                                 # (a)
    "architect": "Architects, except landscape and naval",                 # (c)
    "pilot": "Aircraft pilots and flight engineers",                       # (c)
    "firefighter": "Firefighters",                                         # (a)
    "carpenter": "Carpenters",                                             # (a)
    "executive": "Chief executives",                                       # (d)
    "manager": "Managers, all other",                                      # (b)
    "boss": None,                                                          # (e) rank, not an occupation
    "chief": None,                                                         # (e) rank, not an occupation
    "construction worker": "Construction laborers",                        # (c)
    "driver": "Driver/sales workers and truck drivers",                    # (c)
    "farmer": "Farmers, ranchers, and other agricultural managers",        # (c)
    "fighter pilot": None,                                                 # (e) military; CPS is civilian
    "financier": None,                                                     # (e) not a BLS occupation
    "guard": "Security guards and gambling surveillance officers",         # (c)
    "janitor": "Janitors and building cleaners",                           # (c)
    "laborer": "Laborers and freight, stock, and material movers, hand",   # (c)
    "maestro": None,                                                       # (e) not a BLS occupation
    "mover": "Laborers and freight, stock, and material movers, hand",     # (c)
    "philosopher": None,                                                   # (e) not a BLS occupation
    "physician": "Other physicians",                                       # (b)
    "sheriff": "Police officers",                                          # (d)
    "skipper": "Ship and boat captains and operators",                     # (d) estimate suppressed
    "supervisor": None,                                                    # (e) rank, not an occupation
    "warrior": None,                                                       # (e) not an occupation

    # ---- originally labelled "female" ----------------------------------
    "nurse": "Registered nurses",                                          # (c)
    "nanny": "Childcare workers",                                          # (d)
    "homemaker": None,                                                     # (e) not in the employed universe
    "secretary": "Secretaries and administrative assistants, except legal, medical, and executive",  # (c)
    "receptionist": "Receptionists and information clerks",                # (c)
    "housekeeper": "Maids and housekeeping cleaners",                      # (c)
    "librarian": "Librarians and media collections specialists",           # (c)
    "teacher": "Elementary and middle school teachers",                    # (c)
    "social worker": "Social workers, all other",                          # (b)
    "midwife": "Nurse midwives",                                           # (c) estimate suppressed
    "babysitter": "Childcare workers",                                     # (d)
    "caregiver": "Personal care aides",                                    # (d)
    "dietitian": "Dietitians and nutritionists",                           # (c)
    "dental hygienist": "Dental hygienists",                               # (a)
    "interior designer": "Interior designers",                             # (a)
    "socialite": None,                                                     # (e) not an occupation
    "stylist": "Hairdressers, hairstylists, and cosmetologists",           # (c)

    # ---- originally labelled "neutral" ---------------------------------
    "analyst": "Management analysts",                                      # (c)
    "artist": "Artists and related workers",                               # (c)
    "journalist": "News analysts, reporters, and journalists",             # (c)
    "accountant": "Accountants and auditors",                              # (c)
    "pharmacist": "Pharmacists",                                           # (a)
    "scientist": "Physical scientists, all other",                         # (c)
    "lawyer": "Lawyers",                                                   # (a)
    "consultant": "Management analysts",                                   # (d)
    "researcher": None,                                                    # (e) not a BLS occupation
    "professor": "Postsecondary teachers",                                 # (d)
    "veterinarian": "Veterinarians",                                       # (a)
    "psychologist": "Other psychologists",                                 # (b)
    "editor": "Editors",                                                   # (a)
    "translator": "Interpreters and translators",                          # (c)
    "assistant": None,                                                     # (e) function, not an occupation
    "attendant": None,                                                     # (e) function, not an occupation
    "auditor": "Accountants and auditors",                                 # (c)
    "baker": "Bakers",                                                     # (a)
    "bookkeeper": "Bookkeeping, accounting, and auditing clerks",          # (c)
    "broadcaster": "Broadcast announcers and radio disc jockeys",          # (c) estimate suppressed
    "cashier": "Cashiers",                                                 # (a)
    "cleaner": "Janitors and building cleaners",                           # (c)
    "clerk": "Office clerks, general",                                     # (c)
    "cook": "Cooks",                                                       # (a)
    "counselor": "Counselors, all other",                                  # (b)
    "designer": "Other designers",                                         # (b)
    "developer": "Software developers",                                    # (c)
    "guidance counselor": "Educational, guidance, and career counselors and advisors",  # (c)
    "magician": None,                                                      # (e) not a BLS occupation
    "protege": None,                                                       # (e) not an occupation
    "salesperson": "Retail salespersons",                                  # (c)
    "tailor": "Tailors, dressmakers, and sewers",                          # (c)
    "writer": "Writers and authors",                                       # (c)
}


def label_for(pct: float | None) -> str:
    """REVIEW_TODO.md A8 band rule."""
    if pct is None:
        return "drop"
    if pct < 30:
        return "male"
    if pct > 70:
        return "female"
    if 40 <= pct <= 60:
        return "neutral"
    return "drop"  # 30-40 and 60-70 are deliberately excluded as ambiguous


def load_bls() -> dict[str, str]:
    with BLS_CSV.open(newline="", encoding="utf-8") as fh:
        return {r["occupation"]: r["pct_women"] for r in csv.DictReader(fh)}


def load_onet_soc() -> dict[str, str]:
    """Case-insensitive O*NET title -> 6-digit SOC code (base occupations only)."""
    out: dict[str, str] = {}
    with ONET_OCC.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            code = r["O*NET-SOC Code"]
            if code.endswith(".00"):
                out.setdefault(r["Title"].strip().lower(), code[:-3])
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "code"))
    # NOTE: iterate the LEGACY dict, never synthetic_dataset.ALL_OCCUPATIONS --
    # that list is now derived from the CSV this script writes, so using it
    # would drop every `drop` row on the second run and shrink the file.
    from synthetic_dataset import LEGACY_OCCUPATIONS  # noqa: E402

    all_occupations = [(occ, stereo)
                       for stereo, occs in LEGACY_OCCUPATIONS.items()
                       for occ in occs]

    bls = load_bls()
    soc = load_onet_soc()

    unknown = [t for t in MAPPING.values() if t is not None and t not in bls]
    if unknown:
        print("ERROR: BLS titles not found in the parsed table:", unknown)
        return 1
    missing = [o for o, _ in all_occupations if o not in MAPPING]
    if missing:
        print("ERROR: occupations with no MAPPING entry:", missing)
        return 1

    rows, counts, changed = [], Counter(), []
    for occ, old in all_occupations:
        title = MAPPING[occ]
        raw = bls.get(title, "") if title else ""
        pct = float(raw) if raw not in ("", "-") else None
        lab = label_for(pct)
        rows.append({
            "occupation": occ,
            "bls_title": title or "",
            "soc_code": soc.get((title or "").lower(), ""),
            "pct_women": "" if pct is None else f"{pct:.1f}",
            "label": lab,
            "year": YEAR,
        })
        counts[lab] += 1
        if lab != old:
            changed.append((occ, old, lab, "" if pct is None else f"{pct:.1f}"))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["occupation", "bls_title", "soc_code",
                                           "pct_women", "label", "year"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(rows)} occupations")
    print("new label counts:", dict(counts))
    print(f"soc_code filled for {sum(1 for r in rows if r['soc_code'])}/{len(rows)} rows")
    print(f"\n{len(changed)} occupations changed category:")
    for occ, old, new, pct in sorted(changed, key=lambda x: (x[2], x[0])):
        print(f"  {occ:<20} {old:>7} -> {new:<7} (pct_women={pct or 'n/a'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
